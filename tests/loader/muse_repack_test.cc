// Spec 23b Task 3: the Muse-Glimmer loader's host half (loader/muse_repack.h), host only.
//
// The test WRITES its own fixtures (tests/loader/muse_fixture.h) and reads them back through
// loader::SafetensorsSet exactly as load_muse_glimmer does:
//   1. a 1-layer checkpoint at the REAL widths (a test-only vocabulary of 1024) in the gptq spelling, 3 vision
//      tensors: Review Focus 1 - the fused GEMVs' column order (q || gate || k || v at 0 / 4095 / 4096 / 8191 /
//      8192 / 8447 / 8448 / 8703, gate || up at 0 / 19967 / 19968 / 39935, o_proj and down, + random columns):
//      every word and scale of a sampled column is its source's, every weight (q - 8) x scale of the source row;
//      the norms' fp32 copies bit-exact widenings; the bytes = loader::muse_layer_bytes; Review Focus 2 - the
//      consumed set = muse_expected_names(gptq), the 3 vision names counted, nothing unconsumed;
//   2. small-width checkpoints (refusals, each by name): an extra text tensor, an unknown non-text tensor, a
//      missing one, a qzeros word != 0x77777777 (tensor and index), a non-identity g_idx, desc_act true, group 32,
//      bits 8, asym, an undeclared desc_act beside g_idx, one linear in the other spelling; the autoround spelling
//      accepted;
//   3. when present (environment): MUSE_TINY_INT4 = 23a's tiny `int4` variant (muse_make_tiny.py --variant int4:
//      the tiny's widths, 73 vision tensors) and MUSE_SYNTH_DIR = make_synth.py's real-width checkpoint (ckpt2 or
//      ckpt4: the served constructor, check_muse_config) - every layer repacked and read back the same way.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/json.h"
#include "loader/muse_layout.h"
#include "loader/muse_repack.h"
#include "loader/safetensors.h"
#include "model/muse_glimmer.h"
#include "muse_fixture.h"

namespace {

namespace fs = std::filesystem;
namespace mf = muse_fixture;
using loader::MuseCheckpoint;
using loader::MuseHostLayer;
using loader::MuseHostLinear;
using loader::MuseQuantForm;
using model::MuseGlimmerDesc;
using model::MuseLinearId;

template <class F>
void throws_naming(const std::vector<std::string>& what, F fn) {
  bool ok = false;
  std::string msg;
  try {
    fn();
  } catch (const std::exception& e) {
    msg = e.what();
    ok = true;
    for (const std::string& w : what)
      if (msg.find(w) == std::string::npos) ok = false;
  }
  if (!ok) {
    std::fprintf(stderr, "expected a refusal naming");
    for (const std::string& w : what) std::fprintf(stderr, " '%s'", w.c_str());
    std::fprintf(stderr, "; got: %s\n", msg.empty() ? "(no throw)" : msg.c_str());
  }
  CHECK(ok);
  std::printf("  refused: %.200s\n", msg.c_str());
}

bool same_f32(float a, float b) { return std::memcmp(&a, &b, 4) == 0; }

// A layout-1 tile's word / scale at (k-octet r, column c) - common::repack_int4_layout1's layout.
uint32_t l1_word(const uint32_t* t, uint32_t K, uint32_t r, uint32_t c) {
  return t[(size_t(c / 16) * (K / 64) + r / 8) * 136 + (r % 8) * 16 + c % 16];
}
uint16_t l1_scale(const uint32_t* t, uint32_t K, uint32_t g, uint32_t c) {
  const uint32_t w = t[(size_t(c / 16) * (K / 64) + g) * 136 + 128 + (c % 16) / 2];
  return uint16_t(c % 2 == 0 ? w & 0xFFFFu : w >> 16);
}

// The fused host linear's column c against its source part, word by word, scale by scale and weight by weight.
void check_column(const loader::SafetensorsSet& set, const MuseGlimmerDesc& d, uint32_t l, MuseLinearId id,
                  const MuseHostLinear& h, uint32_t c) {
  const model::MuseLinear ln = d.linear(id);
  const uint32_t K = ln.shape.K, N = ln.shape.N;
  uint32_t part = 0, j = c;
  while (j >= ln.part_n[part]) j -= ln.part_n[part++];
  const std::string base = MuseGlimmerDesc::layer_prefix(l) + ln.parts[part];
  const loader::TensorInfo qi = set.tensors().at(base + ".qweight"), si = set.tensors().at(base + ".scales");
  const uint32_t Np = ln.part_n[part];
  CHECK_EQ(uint32_t(qi.shape[1]), Np);
  const uint32_t* qw = reinterpret_cast<const uint32_t*>(set.data(qi));
  const uint16_t* sc = reinterpret_cast<const uint16_t*>(set.data(si));
  for (uint32_t r = 0; r < K / 8; ++r) {
    const uint32_t src = qw[size_t(r) * Np + j];
    const uint32_t dst = ln.shape.layout == 0 ? h.words[size_t(r) * N + c] : l1_word(h.words.data(), K, r, c);
    CHECK_EQ(dst, src);
  }
  for (uint32_t g = 0; g < K / 64; ++g) {
    const uint16_t src = sc[size_t(g) * Np + j];
    const uint16_t dst = ln.shape.layout == 0 ? h.scales[size_t(g) * N + c] : l1_scale(h.words.data(), K, g, c);
    CHECK_EQ(dst, src);
  }
  // every weight of the column, (q - 8) x scale of the source row
  for (uint32_t k = 0; k < K; ++k) {
    const uint32_t ws = qw[size_t(k / 8) * Np + j];
    const uint32_t wd = ln.shape.layout == 0 ? h.words[size_t(k / 8) * N + c] : l1_word(h.words.data(), K, k / 8, c);
    const uint16_t ss = sc[size_t(k / 64) * Np + j];
    const uint16_t sd = ln.shape.layout == 0 ? h.scales[size_t(k / 64) * N + c] : l1_scale(h.words.data(), K, k / 64, c);
    const float a = (float((ws >> (4 * (k % 8))) & 0xF) - 8.0f) * common::f16_to_f32(ss);
    const float b = (float((wd >> (4 * (k % 8))) & 0xF) - 8.0f) * common::f16_to_f32(sd);
    CHECK(same_f32(a, b));
  }
}

// The boundaries of every part of a fused linear (first and last column of each), plus `extra` pseudo-random ones.
std::vector<uint32_t> sample_columns(const model::MuseLinear& ln, uint32_t extra) {
  std::vector<uint32_t> c;
  uint32_t at = 0;
  for (uint32_t n : ln.part_n) {
    c.push_back(at);
    c.push_back(at + n - 1);
    at += n;
  }
  uint64_t s = 12345 + uint64_t(ln.id);
  for (uint32_t i = 0; i < extra; ++i) {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    c.push_back(uint32_t((s >> 33) % ln.shape.N));
  }
  return c;
}

void check_norm(const loader::SafetensorsSet& set, const std::string& name, const std::vector<float>& h) {
  const loader::TensorInfo t = set.tensors().at(name);
  const uint16_t* w = reinterpret_cast<const uint16_t*>(set.data(t));
  CHECK_EQ(h.size(), size_t(t.shape[0]));
  for (size_t i = 0; i < h.size(); ++i) CHECK(same_f32(h[i], common::bf16_to_f32(w[i])));
}

// Every layer of `ck` repacked and read back against the shards; the top-level tensors; 0 unconsumed.
void check_all(const MuseCheckpoint& ck, uint32_t extra_cols, const char* what) {
  const MuseGlimmerDesc& d = ck.desc();
  const loader::SafetensorsSet set(ck.snapshot());
  CHECK(!ck.unconsumed().empty());   // nothing read yet
  MuseHostLayer h;
  for (uint32_t l = 0; l < d.layers; ++l) {
    ck.layer(l, h);
    const loader::MuseLayerBytes want = loader::muse_layer_bytes(d);
    CHECK_EQ(h.qgkv.bytes(), want.qgkv);
    CHECK_EQ(h.oproj.bytes(), want.oproj);
    CHECK_EQ(h.gate_up.bytes(), want.gate_up);
    CHECK_EQ(h.down.bytes(), want.down);
    CHECK_EQ(h.bytes(), want.total());
    CHECK_EQ(h.norm_block().size() * 4, want.norms);
    const MuseHostLinear* hl[] = {&h.qgkv, &h.oproj, &h.gate_up, &h.down};
    const std::vector<MuseLinearId> ids = d.layer_linears();
    for (size_t i = 0; i < ids.size(); ++i)
      for (uint32_t c : sample_columns(d.linear(ids[i]), extra_cols)) check_column(set, d, l, ids[i], *hl[i], c);
    const std::string p = MuseGlimmerDesc::layer_prefix(l);
    check_norm(set, p + "input_layernorm.weight", h.in_norm);
    check_norm(set, p + "post_attention_layernorm.weight", h.post_attn);
    check_norm(set, p + "pre_feedforward_layernorm.weight", h.pre_ff);
    check_norm(set, p + "post_feedforward_layernorm.weight", h.post_ff);
    const std::vector<float> nb = h.norm_block();
    CHECK(same_f32(nb[d.norm_off_pre_ff()], h.pre_ff[0]) && same_f32(nb[d.norm_off_post_ff() + d.hidden - 1], h.post_ff.back()));
  }
  const std::string lm = MuseGlimmerDesc::text_prefix();
  CHECK(ck.embed() == reinterpret_cast<const uint16_t*>(set.data(set.tensors().at(lm + "embed_tokens.weight"))) ||
        std::memcmp(ck.embed(), set.data(set.tensors().at(lm + "embed_tokens.weight")), size_t(d.hidden) * 2) == 0);
  CHECK(std::memcmp(ck.lm_head(), set.data(set.tensors().at("lm_head.weight")), size_t(d.hidden) * 2) == 0);
  check_norm(set, lm + "norm.weight", ck.final_norm());
  const std::vector<std::string> left = ck.unconsumed();
  for (size_t i = 0; i < left.size() && i < 5; ++i) std::fprintf(stderr, "unconsumed: %s\n", left[i].c_str());
  CHECK(left.empty());
  // Review Focus 2: the consumed set is exactly the expected names; the rest of the file is the vision side.
  CHECK_EQ(ck.text_tensors(), loader::muse_expected_names(d, ck.form()).size());
  CHECK_EQ(ck.text_tensors() + ck.vision_skipped() + ck.skipped_layer_tensors(), ck.tensors());
  std::printf("  %s: %u layers (%u full), %s spelling, %zu text tensors read (%zu more skipped by --layers), "
              "%zu vision skipped by name, 0 unconsumed\n",
              what, d.layers, d.full_layers(), loader::muse_quant_form_name(ck.form()), ck.text_tensors(),
              ck.skipped_layer_tensors(), ck.vision_skipped());
}

// The tiny model's descriptor, from its text_config (check_muse_config refuses its widths by design).
MuseGlimmerDesc desc_from(const fs::path& snap) {
  std::ifstream f(snap / "config.json");
  std::stringstream s;
  s << f.rdbuf();
  const common::json::Value c = common::json::parse(s.str());
  const common::json::Value& t = c.at("text_config");
  MuseGlimmerDesc d;
  d.layers = uint32_t(t.at("num_hidden_layers").num());
  d.hidden = uint32_t(t.at("hidden_size").num());
  d.inter = uint32_t(t.at("intermediate_size").num());
  d.q_heads = uint32_t(t.at("num_attention_heads").num());
  d.kv_heads = uint32_t(t.at("num_key_value_heads").num());
  d.head_dim = uint32_t(t.at("head_dim").num());
  d.vocab = uint32_t(t.at("vocab_size").num());
  d.window = uint32_t(t.at("sliding_window").num());
  for (const common::json::Value& v : t.at("layer_types").arr()) d.full.push_back(v.str() == "full_attention");
  return d;
}

// A small descriptor for the refusal fixtures (the tiny model's shape: hidden 64, 4 q / 2 kv x 16, FF 128).
MuseGlimmerDesc small_desc(uint32_t layers = 2) {
  MuseGlimmerDesc d = model::muse_glimmer();
  d.layers = layers;
  d.hidden = 64;
  d.q_heads = 4;
  d.kv_heads = 2;
  d.head_dim = 16;
  d.inter = 128;
  d.vocab = 64;
  d.full.assign(layers, false);
  if (layers > 3) d.full[3] = true;
  return d;
}

// muse_make_tiny.py writes ONE model.safetensors without an index (the loader reads indexed snapshots only): a
// temporary directory with the file and config.json linked and an index written from the header.
fs::path indexed(const fs::path& snap) {
  if (fs::exists(snap / "model.safetensors.index.json")) return snap;
  const fs::path d = mf::tmpdir("indexed");
  fs::create_directories(d);
  fs::create_symlink(fs::absolute(snap / "model.safetensors"), d / "model.safetensors");
  fs::create_symlink(fs::absolute(snap / "config.json"), d / "config.json");
  std::ifstream f(snap / "model.safetensors", std::ios::binary);
  uint64_t n = 0;
  f.read(reinterpret_cast<char*>(&n), 8);
  std::vector<uint8_t> h(size_t(8) + n);
  std::memcpy(h.data(), &n, 8);
  f.read(reinterpret_cast<char*>(h.data() + 8), std::streamsize(n));
  std::ofstream ix(d / "model.safetensors.index.json");
  ix << "{\"metadata\":{},\"weight_map\":{";
  bool first = true;
  for (const auto& [name, info] : loader::SafetensorsSet::parse_header(h.data(), h.size())) {
    (void)info;
    ix << (first ? "" : ",") << "\"" << name << "\":\"model.safetensors\"";
    first = false;
  }
  ix << "}}";
  return d;
}

fs::path write(const std::string& tag, const mf::Tensors& ts, const std::string& cfg) {
  const fs::path p = mf::tmpdir(tag);
  mf::write_checkpoint(p, ts, cfg);
  return p;
}

void refusals() {
  const MuseGlimmerDesc d = small_desc();
  const mf::Tensors good = mf::make(d, true);
  {
    // the good one (both spellings), the names both ways
    const fs::path p = write("good", good, mf::config(true));
    const MuseCheckpoint ck(d, p.string() + "/");
    CHECK(ck.form() == MuseQuantForm::Gptq);
    check_all(ck, 4, "small gptq");
    const fs::path q = write("autoround", mf::make(d, false), mf::config(false));
    const MuseCheckpoint ca(d, q.string() + "/");
    CHECK(ca.form() == MuseQuantForm::AutoRound);
    check_all(ca, 4, "small autoround");
    CHECK_EQ(loader::muse_expected_names(d, MuseQuantForm::Gptq).size(), size_t(3 + 2 * 36));
    CHECK_EQ(loader::muse_expected_names(d, MuseQuantForm::AutoRound).size(), size_t(3 + 2 * 28));
    fs::remove_all(p);
    fs::remove_all(q);
  }
  const std::string L0 = MuseGlimmerDesc::layer_prefix(0);
  auto refuse = [&](const std::string& tag, const mf::Tensors& ts, const std::string& cfg,
                    const std::vector<std::string>& names) {
    const fs::path p = write(tag, ts, cfg);
    throws_naming(names, [&] { MuseCheckpoint ck(d, p.string() + "/"); });
    fs::remove_all(p);
  };
  {
    mf::Tensors ts = good;
    ts[L0 + "self_attn.q_norm.weight"] = ts[L0 + "input_layernorm.weight"];   // a weighted q norm: not this model
    refuse("extra", ts, mf::config(true), {L0 + "self_attn.q_norm.weight"});
  }
  {
    mf::Tensors ts = good;
    ts["model.multi_modal_projector.weight"] = ts[L0 + "input_layernorm.weight"];
    refuse("stranger", ts, mf::config(true), {"model.multi_modal_projector.weight"});
  }
  {
    mf::Tensors ts = good;
    ts.erase(L0 + "mlp.down_proj.scales");
    refuse("missing", ts, mf::config(true), {L0 + "mlp.down_proj.scales"});
  }
  {
    mf::Tensors ts = good;
    reinterpret_cast<uint32_t*>(ts[L0 + "self_attn.k_proj.qzeros"].bytes.data())[1] = 0x77777776u;
    refuse("qzeros", ts, mf::config(true), {L0 + "self_attn.k_proj.qzeros", "[1]"});
  }
  {
    mf::Tensors ts = good;
    reinterpret_cast<int32_t*>(ts[L0 + "mlp.up_proj.g_idx"].bytes.data())[7] = 1;
    refuse("g_idx", ts, mf::config(true), {L0 + "mlp.up_proj.g_idx", "[7]"});
  }
  refuse("desc_act", good, mf::config(true, ",\"desc_act\":true"), {"desc_act=true"});
  refuse("group32", good, mf::config(true, ",\"group_size\":32"), {"g=32"});
  refuse("bits8", good, mf::config(true, ",\"bits\":8"), {"bits=8"});
  refuse("asym", good, mf::config(true, ",\"sym\":false"), {"sym=false"});
  // the auto-round spelling declares no desc_act: a g_idx beside it is a permutation nobody wrote down
  refuse("undeclared", good, mf::config(false), {"desc_act", "g_idx"});
  {
    // one linear in the other spelling: the names are no longer one form's
    mf::Tensors ts = mf::make(d, false);
    ts[L0 + "self_attn.o_proj.g_idx"] = good.at(L0 + "self_attn.o_proj.g_idx");
    refuse("mixed", ts, mf::config(true), {"g_idx"});
  }
  {
    // a bf16 linear where the format is int4: refused by name
    mf::Tensors ts = good;
    for (const char* s : {".qweight", ".scales", ".qzeros", ".g_idx"}) ts.erase(L0 + "mlp.gate_proj" + s);
    ts[L0 + "mlp.gate_proj.weight"] = mf::Tensor{"BF16", {d.inter, d.hidden}, std::vector<uint8_t>(size_t(d.inter) * d.hidden * 2)};
    refuse("bf16lin", ts, mf::config(true), {L0 + "mlp.gate_proj.weight"});
  }
}

}  // namespace

// resolve_muse_snapshot: a cache holding only the -GPTQ spelling answers for the -AutoRound repo id (and the other
// way round); another repo is not substituted.
void spelling_fallback() {
  const fs::path hf = mf::tmpdir("hf");
  const fs::path m = hf / "hub" / "models--Vishva007--Muse-Glimmer-30B-W4A16-AutoRound-GPTQ";
  fs::create_directories(m / "refs");
  fs::create_directories(m / "snapshots" / "0987fe04");
  std::ofstream(m / "refs" / "main") << "0987fe04";
  std::ofstream(m / "snapshots" / "0987fe04" / "config.json") << "{}";
  const char* old = std::getenv("HF_HOME");
  const std::string keep = old ? old : "";
  ::setenv("HF_HOME", hf.c_str(), 1);
  const std::string got = loader::resolve_muse_snapshot("Vishva007/Muse-Glimmer-30B-W4A16-AutoRound");
  CHECK(got.find("models--Vishva007--Muse-Glimmer-30B-W4A16-AutoRound-GPTQ/snapshots/0987fe04") != std::string::npos);
  CHECK(loader::resolve_muse_snapshot("Vishva007/Muse-Glimmer-30B-W4A16-AutoRound-GPTQ") == got);
  throws_naming({"Vishva007/Some-Other-Repo"}, [] { (void)loader::resolve_muse_snapshot("Vishva007/Some-Other-Repo"); });
  if (old)
    ::setenv("HF_HOME", keep.c_str(), 1);
  else
    ::unsetenv("HF_HOME");
  fs::remove_all(hf);
  std::printf("  either spelling: the -AutoRound repo id resolves to the cached -GPTQ snapshot\n");
}

int main() {
  spelling_fallback();
  // --- 1. the real widths, one layer, a test-only vocabulary ------------------------------------------------
  {
    MuseGlimmerDesc d = model::muse_glimmer();
    d.layers = 1;
    d.full.resize(1);
    d.vocab = 1024;
    const fs::path p = mf::tmpdir("real1");
    mf::write_checkpoint(p, mf::make(d, true), mf::config(true));
    const MuseCheckpoint ck(d, p.string() + "/");
    CHECK_EQ(ck.vision_skipped(), size_t(3));
    // Review Focus 1's columns, named: they are the parts' boundaries sample_columns() visits.
    const std::vector<uint32_t> qc = sample_columns(d.linear(MuseLinearId::QGKV), 0);
    CHECK((qc == std::vector<uint32_t>{0, 4095, 4096, 8191, 8192, 8447, 8448, 8703}));
    const std::vector<uint32_t> gc = sample_columns(d.linear(MuseLinearId::GateUp), 0);
    CHECK((gc == std::vector<uint32_t>{0, 19967, 19968, 39935}));
    check_all(ck, 64, "real widths, 1 layer");
    fs::remove_all(p);
  }
  // --- 2. refusals by name ----------------------------------------------------------------------------------
  refusals();
  // --- 3. 23a's tiny int4 variant and make_synth.py's real-width checkpoint, when present --------------------
  if (const char* t = std::getenv("MUSE_TINY_INT4"); t && fs::exists(fs::path(t) / "config.json")) {
    const MuseGlimmerDesc d = desc_from(t);
    const fs::path ix = indexed(t);
    {
      const MuseCheckpoint ck(d, ix.string() + "/");
      CHECK(ck.form() == MuseQuantForm::Gptq);
      check_all(ck, 16, "23a's tiny int4 variant");
      CHECK_EQ(ck.vision_skipped(), size_t(73));
    }
    if (ix != fs::path(t)) fs::remove_all(ix);
  } else {
    std::printf("  (no MUSE_TINY_INT4 checkpoint - muse_make_tiny.py --variant int4; that part skipped)\n");
  }
  bool synth = false;
  if (const char* s = std::getenv("MUSE_SYNTH_DIR"); s)
    for (const char* sub : {"ckpt2", "ckpt4"})
      if (fs::exists(fs::path(s) / sub / "config.json")) {
        const MuseCheckpoint ck((fs::path(s) / sub).string() + "/");   // the served constructor
        check_all(ck, 16, (std::string("make_synth.py ") + sub).c_str());
        synth = true;
        break;
      }
  if (!synth) std::printf("  (no make_synth.py checkpoint under MUSE_SYNTH_DIR - that part skipped)\n");
  std::printf("muse_repack_test OK\n");
  return 0;
}
