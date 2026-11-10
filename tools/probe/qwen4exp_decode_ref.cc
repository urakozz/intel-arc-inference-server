// Bounded teacher-forced decode capture for tools/oracle/qwen4exp_decode_diagnose.py.
// argv: checkpoint ids output-directory [layers=4] [rows=8]. No generation or prefill.
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "common/bf16.h"
#include "l0/cmdlist.h"
#include "runtime/qwen4exp_rig.h"

namespace {
template<class T> void save(const std::string& path, const std::vector<T>& v) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
  if (!f) throw std::runtime_error("write failed: " + path);
}
}

int main(int argc, char** argv) {
  if (argc < 4) return 2;
  qwen4exp_rig::Options o;
  o.layers = argc > 4 ? uint32_t(std::stoul(argv[4])) : 4;
  o.max_len = 256;  // loader's minimum context quantum
  o.int8_head = true;
  o.debug_tap = true;
  auto ids = qwen4exp_rig::read_ids(argv[2], 248320);
  ids.resize(std::min(ids.size(), argc > 5 ? size_t(std::stoul(argv[5])) : size_t(8)));
  if (ids.empty() || ids.size() > o.max_len) return 2;
  const std::string out = argv[3];
  std::filesystem::create_directories(out);
  qwen4exp_rig::Rig rig;
  const std::string snap = qwen4exp_rig::qwen4exp_snapshot(argv[1]);
  if (snap.empty()) return 77;
  qwen4exp_rig::build(rig, snap, o);
  auto& eng = *rig.eng;
  const auto& d = eng.model().desc;
  loader::Q4PleHost ple(loader::q4_ple_dir(snap));
  std::ofstream meta(out + "/manifest.json");
  meta << "{\"layers\":" << d.layers << ",\"rows\":" << ids.size()
       << ",\"max_len\":" << o.max_len << ",\"hidden\":" << d.hc_n()
       << ",\"kv\":" << d.kv_n() << ",\"idx\":" << d.idx_dim
       << ",\"gdn_elems\":" << runtime::qwen4exp::gdn_state_bytes_per_layer(d) / 4
       << ",\"conv_rows\":" << d.conv_rows() << ",\"ple_layer\":" << d.ple_layer
       << ",\"ple_dim\":" << d.ple_dim << ",\"moe_inter\":" << d.moe_inter
       << ",\"model_hidden\":" << d.hidden << ",\"qsa\":[";
  for (uint32_t l = 0; l < d.layers; ++l) meta << (l ? "," : "") << d.is_qsa(l);
  meta << "]}";
  meta.close();
  save(out + "/prompt.u32", ids);
  for (size_t r = 0; r < ids.size(); ++r) {
    eng.ingest({ids[r]});
    const std::string base = out + "/r" + std::to_string(r);
    save(base + ".H.bf16", eng.read_debug_H());
    save(base + ".routes.u32", eng.read_routes());
    save(base + ".logits.f32", eng.read_logits());
    const auto rows = eng.read_ple_ids();
    save(base + ".ple_ids.u64", rows);
    std::vector<int8_t> qs, pinned_q;
    std::vector<float> scales, pinned_s;
    std::vector<uint16_t> dequant, pinned;
    const auto& table = eng.model().ple;
    for (uint32_t h = 0; h < d.ple_heads; ++h) {
      const uint64_t local = rows[h] - ple.offsets[h];
      const int8_t* q = ple.q(h) + local * d.ple_dim;
      qs.insert(qs.end(), q, q + d.ple_dim);
      const float s = ple.scale() == loader::Q4PleScale::Bf16
          ? common::bf16_to_f32(static_cast<const uint16_t*>(ple.s(h))[local])
          : static_cast<const float*>(ple.s(h))[local];
      scales.push_back(s);
      const size_t at = dequant.size();
      dequant.resize(at + d.ple_dim);
      ple.dequant_row(h, local, dequant.data() + at);
      const auto* pq = static_cast<const int8_t*>(table.q[h]->ptr()) + local * d.ple_dim;
      const float ps = table.scale == loader::Q4PleScale::Bf16
          ? common::bf16_to_f32(static_cast<const uint16_t*>(table.s[h]->ptr())[local])
          : static_cast<const float*>(table.s[h]->ptr())[local];
      pinned_q.insert(pinned_q.end(), pq, pq + d.ple_dim);
      pinned_s.push_back(ps);
      for (uint32_t j = 0; j < d.ple_dim; ++j) pinned.push_back(common::f32_to_bf16(float(pq[j]) * ps));
    }
    save(base + ".ple.q.i8", qs);
    save(base + ".ple.s.f32", scales);
    save(base + ".ple.pinned.q.i8", pinned_q);
    save(base + ".ple.pinned.s.f32", pinned_s);
    save(base + ".ple.bf16", dequant);
    save(base + ".ple.pinned.bf16", pinned);
    for (uint32_t l = 0; l < d.layers; ++l) save(base + ".L" + std::to_string(l) + ".state", eng.read_layer_state(l));
  }
  // Full expert-zero gate/up and down packed device blocks: validates layout and exact dequant products.
  auto imm = l0::CmdList::immediate(*rig.c0);
  const auto& l0 = eng.model().parts[0].layer(0);
  std::vector<uint32_t> gu(loader::q4_gate_up_block_bytes(d) / 4), dn(loader::q4_down_block_bytes(d) / 4);
  imm.copy(gu.data(), l0.gate_up->ptr(), gu.size() * 4);
  imm.copy(dn.data(), l0.down->ptr(), dn.size() * 4);
  save(out + "/L0.expert0.gate_up.u32", gu);
  save(out + "/L0.expert0.down.u32", dn);
  std::printf("decode diagnostic captured %zu teacher-forced rows, %u layers\n", ids.size(), d.layers);
}
