// Spec 21b Task 6: load_qwen4exp on the card (box only; label `checkpoint;qwen4exp`), one device.
//   qwen4exp_load_checkpoint_test <snapshot> [layers] [mtp]       (B70_Q4_PLE: the PLE file, else <snapshot>-ple-int8/)
// Loads the checkpoint (a synthetic one from tools/quantize/qwen4exp/make_synth.py + ple_int8.py, or Intel's
// truncated to its first `layers` - development mode), prints the report (one line per device and one for
// the pinned PLE table: its bytes, the time to pin it, MemAvailable before / after - spec 22 P0.5's starting
// numbers), and asserts: zero unconsumed tensors; the part's bytes equal to q4_device_weight_bytes (the
// planner's figure); the RoPE table's size; device bytes read back against a fresh host repack at the edges a
// wrong stride breaks (the last layer's last expert's gate||up and down blocks at q4_gate_up_offset /
// q4_down_offset, the router's row `experts` - the shared gate - and its zero tail, the MTP head's last expert
// block); the PLE host ranges' page words equal the file's bytes at every 2 MiB page and sampled rows
// dequantise as the file's; the device pointer table holds the ranges' addresses. Exit 77 (SKIP) when the
// snapshot or its PLE file is not on this machine.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "check.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/qwen4exp_layout.h"
#include "loader/qwen4exp_loader.h"
#include "loader/qwen4exp_ple.h"
#include "loader/qwen4exp_repack.h"
#include "loader/safetensors.h"
#include "loader/snapshot.h"

namespace {
std::vector<uint8_t> readback(l0::CmdList& imm, const l0::Mem& m, size_t off, size_t bytes) {
  std::vector<uint8_t> v(bytes);
  imm.copy(v.data(), static_cast<const uint8_t*>(m.ptr()) + off, bytes);
  return v;
}
}  // namespace

int main(int argc, char** argv) {
  const std::string arg = argc > 1 ? argv[1] : "Intel/Qwen3.8-Flash-Next-W4A16-AutoRound";
  const uint32_t layers = argc > 2 ? uint32_t(std::stoul(argv[2])) : 0;
  const bool mtp = argc > 3 && std::string(argv[3]) == "mtp";
  std::string snap, ple;
  try {
    snap = loader::resolve_snapshot(arg);
    if (!loader::is_qwen4exp_checkpoint(snap)) throw std::runtime_error("not a qwen4_exp checkpoint");
    ple = loader::q4_ple_dir(snap);
  } catch (const std::exception& e) {
    std::printf("SKIP: no Qwen3.8-Flash-Next checkpoint and PLE file at '%s' (%s)\n", arg.c_str(), e.what());
    return 77;
  }
  const model::Qwen4ExpDesc pre = loader::qwen4exp_checkpoint_desc(snap, layers);
  l0::Context ctx(0);
  std::vector<l0::Context*> devs = {&ctx};
  const uint32_t max_len = 32768;
  loader::Q4LoadedModel m =
      loader::load_qwen4exp(devs, snap, max_len, model::Q4Placement::one(pre), loader::LmHeadForm::Int8, layers, mtp, ple);
  const model::Qwen4ExpDesc& d = m.desc;
  CHECK(d.layers == pre.layers && d.forms.dense == pre.forms.dense && d.forms.expert_group == pre.forms.expert_group);
  CHECK_EQ(m.unconsumed, size_t(0));
  CHECK_EQ(m.parts.size(), size_t(1));
  const loader::Q4DevicePart& P = m.parts[0];
  CHECK_EQ(P.layers.size(), size_t(d.layers));
  CHECK_EQ(P.bytes, loader::q4_device_weight_bytes(d, m.placement, 0, true, mtp));
  CHECK_EQ(P.rope_bytes, d.rope_table_bytes(max_len));
  CHECK(P.embed && P.lm_head && P.final_mixer && P.rope && bool(P.mtp) == mtp);

  // The edges against a fresh host repack of the same checkpoint.
  loader::SafetensorsSet set(snap);
  model::Qwen4ExpDesc full = loader::qwen4exp_checkpoint_desc(snap, 0);
  loader::Q4Checkpoint ck(full, set);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  const uint32_t last = d.layers - 1, E = d.experts;
  loader::Q4HostLayer h;
  ck.repack_layer(last, h);
  const loader::Q4Layer& L = P.layer(last);
  {
    const size_t gub = loader::q4_gate_up_block_bytes(d), dnb = loader::q4_down_block_bytes(d);
    const std::vector<uint8_t> gu = readback(imm, *L.gate_up, loader::q4_gate_up_offset(d, E - 1), gub);
    const std::vector<uint8_t> dn = readback(imm, *L.down, loader::q4_down_offset(d, E - 1), dnb);
    CHECK(std::memcmp(gu.data(), reinterpret_cast<const uint8_t*>(h.gate_up.data()) + loader::q4_gate_up_offset(d, E - 1), gub) == 0);
    CHECK(std::memcmp(dn.data(), reinterpret_cast<const uint8_t*>(h.down.data()) + loader::q4_down_offset(d, E - 1), dnb) == 0);
    // The router's 16-row tile holding row `experts` (the shared gate) and the zero rows after it.
    const size_t tile = size_t(d.hidden) * 16 * 2;
    const std::vector<uint8_t> rt = readback(imm, L.router->mem, size_t(E / 16) * tile, tile);
    CHECK(std::memcmp(rt.data(), reinterpret_cast<const uint8_t*>(h.router.tiles.data()) + size_t(E / 16) * tile, tile) == 0);
    const std::vector<uint8_t> sh = readback(imm, *L.shared, 0, h.shared.size());
    CHECK(std::memcmp(sh.data(), h.shared.data(), h.shared.size()) == 0);
  }
  if (mtp) {
    loader::Q4HostLayer hm;
    ck.repack_mtp(hm);
    const size_t gub = loader::q4_gate_up_block_bytes(d);
    const std::vector<uint8_t> gu = readback(imm, *P.mtp->gate_up, loader::q4_gate_up_offset(d, E - 1), gub);
    CHECK(std::memcmp(gu.data(), reinterpret_cast<const uint8_t*>(hm.gate_up.data()) + loader::q4_gate_up_offset(d, E - 1), gub) == 0);
  }
  // The pinned PLE table: every range's page words equal the file's bytes there, sampled rows dequantise
  // alike, and the device pointer table holds the ranges' addresses.
  {
    const loader::Q4PleHost f(ple);
    const uint32_t H = f.heads();
    size_t w = 0;
    for (uint32_t r = 0; r < 2 * H; ++r) {
      const uint8_t* src = r < H ? reinterpret_cast<const uint8_t*>(f.q(r)) : static_cast<const uint8_t*>(f.s(r - H));
      const size_t n = r < H ? f.q_bytes(r) : f.s_bytes(r - H);
      for (size_t p = 0; p * loader::kQ4PlePage + 8 <= n; ++p) {
        uint64_t want = 0;
        std::memcpy(&want, src + p * loader::kQ4PlePage, 8);
        CHECK(w < m.ple.page_words.size() && m.ple.page_words[w++] == want);
      }
    }
    CHECK_EQ(w, m.ple.page_words.size());
    for (uint32_t hh = 0; hh < H; ++hh) {
      const uint64_t r = f.rows(hh) - 1;
      CHECK(std::memcmp(m.ple.q[hh]->as<int8_t>() + r * f.dim(), f.q(hh) + r * f.dim(), f.dim()) == 0);
    }
    std::vector<uint64_t> ptrs(2 * H);
    imm.copy(ptrs.data(), m.ple.ptrs->ptr(), ptrs.size() * 8);
    for (uint32_t r = 0; r < H; ++r) {
      CHECK(ptrs[r] == reinterpret_cast<uint64_t>(m.ple.q[r]->ptr()));
      CHECK(ptrs[H + r] == reinterpret_cast<uint64_t>(m.ple.s[r]->ptr()));
    }
    CHECK(m.ple.tag_checksum != 0 && m.ple.bytes == f.bytes());
  }
  std::printf("qwen4exp_load_checkpoint_test OK: %u layers, dense %s, experts g%u%s, %zu B of weights, the PLE table "
              "%zu B pinned in %.1f s (MemAvailable %.1f -> %.1f GB)\n",
              d.layers, model::q4_form_name(d.forms.dense), d.forms.expert_group, mtp ? ", MTP head" : "", P.bytes,
              m.ple.bytes, m.ple.seconds, m.ple.mem_before / 1e9, m.ple.mem_after / 1e9);
  return 0;
}
