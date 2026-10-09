#include "loader/qwen4exp_ple.h"

#include <sys/stat.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "common/bf16.h"
#include "loader/quant.h"

// The host half of the PLE table (no Level Zero): the file, its checks, the host-memory rule, the
// directory. load_q4_ple, the pinning, is qwen4exp_ple_usm.cc (Level Zero) - split so the host tests link
// without a device library.
namespace loader {
namespace {

std::string with_slash(const std::string& d) { return d.empty() || d.back() == '/' ? d : d + "/"; }

std::vector<uint64_t> i64s(const SafetensorsSet& set, const std::string& dir, const std::string& name) {
  auto it = set.tensors().find(name);
  if (it == set.tensors().end())
    throw std::runtime_error("PLE file " + dir + ": no tensor '" + name + "' (tools/quantize/qwen4exp/ple_int8.py "
                             "writes it)");
  const TensorInfo& t = it->second;
  if (t.dtype != "I64" || t.shape.size() != 1)
    throw std::runtime_error("PLE file " + dir + ": '" + name + "' is " + t.dtype + " rank " +
                             std::to_string(t.shape.size()) + ", expected I64 rank 1");
  check_align(set.data(t), 8, name);
  const int64_t* v = reinterpret_cast<const int64_t*>(set.data(t));
  std::vector<uint64_t> out(t.shape[0]);
  for (size_t i = 0; i < out.size(); ++i) {
    if (v[i] < 0) throw std::runtime_error("PLE file " + dir + ": '" + name + "'[" + std::to_string(i) + "] < 0");
    out[i] = uint64_t(v[i]);
  }
  return out;
}

template <class A, class B>
std::string show(const A& v, const B& w) {
  std::string s = "[";
  for (size_t i = 0; i < v.size() && i < 4; ++i) s += (i ? ", " : "") + std::to_string(v[i]);
  s += v.size() > 4 ? ", ...] vs [" : "] vs [";
  for (size_t i = 0; i < w.size() && i < 4; ++i) s += (i ? ", " : "") + std::to_string(w[i]);
  return s + (w.size() > 4 ? ", ...]" : "]");
}

}  // namespace

Q4PleHost::Q4PleHost(const std::string& d) : dir(with_slash(d)) {
  set_ = std::make_unique<SafetensorsSet>(dir);
  const auto& ts = set_->tensors();
  for (uint32_t h = 0;; ++h) {
    const std::string qn = "ple.h" + std::to_string(h) + ".q", sn = "ple.h" + std::to_string(h) + ".s";
    const auto qi = ts.find(qn);
    if (qi == ts.end()) break;
    const auto si = ts.find(sn);
    if (si == ts.end()) throw std::runtime_error("PLE file " + dir + ": '" + qn + "' without '" + sn + "'");
    const TensorInfo& q = qi->second;
    const TensorInfo& s = si->second;
    if (q.dtype != "I8" || q.shape.size() != 2)
      throw std::runtime_error("PLE file " + dir + ": '" + qn + "' is " + q.dtype + " rank " +
                               std::to_string(q.shape.size()) + ", expected I8 [rows][dim]");
    if ((s.dtype != "F32" && s.dtype != "BF16") || s.shape.size() != 1 || s.shape[0] != q.shape[0])
      throw std::runtime_error("PLE file " + dir + ": '" + sn + "' is " + s.dtype + " of " +
                               std::to_string(s.shape.empty() ? 0 : s.shape[0]) + " rows, expected F32 or BF16 [" +
                               std::to_string(q.shape[0]) + "]");
    const Q4PleScale sc = s.dtype == "BF16" ? Q4PleScale::Bf16 : Q4PleScale::F32;
    if (h == 0) {
      scale_ = sc;
      dim_ = uint32_t(q.shape[1]);
    } else if (sc != scale_ || q.shape[1] != dim_) {
      throw std::runtime_error("PLE file " + dir + ": '" + qn + "' / '" + sn + "' (" + s.dtype + ", dim " +
                               std::to_string(q.shape[1]) + ") differ from head 0's (" + q4_ple_scale_name(scale_) +
                               ", dim " + std::to_string(dim_) + ") - one scale dtype and one width for every head");
    }
    check_align(set_->data(s), sc == Q4PleScale::Bf16 ? 2 : 4, sn);
    rows_.push_back(q.shape[0]);
    q_.push_back(reinterpret_cast<const int8_t*>(set_->data(q)));
    s_.push_back(set_->data(s));
  }
  if (rows_.empty())
    throw std::runtime_error("PLE file " + dir + ": no 'ple.h0.q' - not ple_int8.py's per-head int8 table");
  const std::vector<uint64_t> m = i64s(*set_, dir, "ple.layer_multipliers");
  if (m.size() != 3)
    throw std::runtime_error("PLE file " + dir + ": 'ple.layer_multipliers' has " + std::to_string(m.size()) +
                             " entries, expected 3 (bigram + trigram)");
  for (int i = 0; i < 3; ++i) multipliers[i] = m[i];
  sizes = i64s(*set_, dir, "ple.ngram_heads_vocab_sizes");
  offsets = i64s(*set_, dir, "ple.ngram_heads_offsets");
}

size_t Q4PleHost::bytes() const {
  size_t b = 0;
  for (uint32_t h = 0; h < heads(); ++h) b += q_bytes(h) + s_bytes(h);
  return b;
}

void Q4PleHost::dequant_row(uint32_t h, uint64_t r, uint16_t* out) const {
  const int8_t* q = q_.at(h) + r * dim_;
  float s;
  if (scale_ == Q4PleScale::Bf16)
    s = common::bf16_to_f32(reinterpret_cast<const uint16_t*>(s_.at(h))[r]);
  else
    s = reinterpret_cast<const float*>(s_.at(h))[r];
  for (uint32_t i = 0; i < dim_; ++i) out[i] = common::f32_to_bf16(float(q[i]) * s);
}

void check_q4_ple(const model::Qwen4ExpDesc& d, const Q4PleHost& h) {
  const std::string where = "PLE file " + h.dir + ": ";
  if (h.heads() != d.ple_heads)
    throw std::runtime_error(where + std::to_string(h.heads()) + " heads (ple.h0..), " + d.name + " has " +
                             std::to_string(d.ple_heads) + " (ngram_size - 1) x heads_per_ngram");
  if (h.dim() != d.ple_dim)
    throw std::runtime_error(where + "'ple.h0.q' rows are " + std::to_string(h.dim()) + " wide, " + d.name + "'s are " +
                             std::to_string(d.ple_dim) + " (ple_embed_dim / heads)");
  const std::array<uint64_t, 3> m = q4_ple_multipliers(d.vocab, d.ngram, 0, d.ple_seed);
  if (h.multipliers != m)
    throw std::runtime_error(where + "'ple.layer_multipliers' differs from the formula (vocab " +
                             std::to_string(d.vocab) + ", seed " + std::to_string(d.ple_seed) + "): " +
                             show(h.multipliers, m));
  const std::vector<uint64_t> sizes = q4_ple_primes(d.ple_base, d.ple_heads, 0), offs = q4_ple_offsets(sizes);
  if (h.sizes != sizes)
    throw std::runtime_error(where + "'ple.ngram_heads_vocab_sizes' differs from the primes after " +
                             std::to_string(d.ple_base) + " (ngram_vocab_size_base): " + show(h.sizes, sizes) +
                             " - a file made for another checkpoint?");
  if (h.offsets != offs)
    throw std::runtime_error(where + "'ple.ngram_heads_offsets' differs from the sizes' running sum: " +
                             show(h.offsets, offs));
  for (uint32_t i = 0; i < h.heads(); ++i)
    if (h.rows(i) != sizes[i])
      throw std::runtime_error(where + "'ple.h" + std::to_string(i) + ".q' has " + std::to_string(h.rows(i)) +
                               " rows, head " + std::to_string(i) + "'s vocabulary is " + std::to_string(sizes[i]));
}

void check_q4_ple_host_fit(size_t table_bytes, size_t mem_available) {
  if (mem_available == 0) return;
  if (table_bytes + kQ4PleHostMargin > mem_available)
    throw std::runtime_error("the PLE table is " + std::to_string(table_bytes) + " B (" +
                             std::to_string(table_bytes / 1000000000.0).substr(0, 6) + " GB) of pinned host memory, "
                             "but MemAvailable is " + std::to_string(mem_available) + " B (" +
                             std::to_string(mem_available / 1000000000.0).substr(0, 6) +
                             " GB) and the rule keeps a 16 GiB margin (the measured pinned cap, docs/probe-prefix-"
                             "cache-2026-09-27.md §1) - free host memory first");
}

size_t q4_mem_available() {
  std::ifstream f("/proc/meminfo");
  std::string key;
  size_t kb = 0;
  std::string unit;
  while (f >> key >> kb >> unit)
    if (key == "MemAvailable:") return kb * 1024;
  return 0;
}

std::string q4_ple_dir(const std::string& snapshot_dir) {
  std::string dir;
  if (const char* e = std::getenv("B70_Q4_PLE"); e && *e) {
    dir = with_slash(e);
  } else {
    std::string s = snapshot_dir;
    while (!s.empty() && s.back() == '/') s.pop_back();
    dir = s + "-ple-int8/";
  }
  struct stat st {};
  if (::stat((dir + "model.safetensors.index.json").c_str(), &st) != 0)
    throw std::runtime_error("no PLE int8 table at " + dir + " (B70_Q4_PLE overrides): the engine reads the n-gram "
                             "table as int8 rows from a one-time file beside the checkpoint (spec 21 decision 7) - "
                             "make it with: python3 tools/quantize/qwen4exp/ple_int8.py " + snapshot_dir + " " + dir);
  return dir;
}

}  // namespace loader
