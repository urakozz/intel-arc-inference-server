// Spec 14 §2: the loader's packed-int4 fold against the Python reference, on the
// shared synthetic fixture (tools/oracle/agnes_fold.py --write-fixture). Host-only.
//
//   1. fold_n(gate, gate_p), fold_n(up, up_p), fold_k(down, down_p) equal the
//      fixture's folded tensors byte for byte (qweight and scales);
//   2. the loader's gate||up assembly on the folded pair - cols_interleave16 then
//      repack_int4_layout0_cols, exactly load_linear's calls - equals the
//      fixture's `gate_up_l0` byte for byte;
//   3. the refusals: K mismatch on a column join, N mismatch on a row join, and a
//      row join off a g64 group boundary.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include "check.h"
#include "common/repack.h"
#include "loader/fold.h"
#include "loader/safetensors.h"

namespace {

struct Fixture {
  loader::MappedFile file;
  std::map<std::string, loader::TensorInfo> t;
  const uint8_t* base = nullptr;

  explicit Fixture(const std::string& path) : file(path) {
    for (auto& [name, info] : loader::SafetensorsSet::parse_header(file.data(), file.size()))
      t.emplace(name, info);
    uint64_t hlen = 0;
    std::memcpy(&hlen, file.data(), 8);
    base = file.data() + 8 + hlen;   // 8-aligned: safetensors pads the header
  }
  const loader::TensorInfo& info(const std::string& n) const {
    auto it = t.find(n);
    if (it == t.end()) throw std::runtime_error("fixture has no " + n);
    return it->second;
  }
  template <class T>
  const T* ptr(const std::string& n) const {
    return reinterpret_cast<const T*>(base + info(n).begin);
  }
  // One packed linear by role ("gate", "down_p", ...).
  loader::PackedInt4View view(const std::string& role) const {
    const loader::TensorInfo& q = info(role + ".qweight");
    return {uint32_t(q.shape[0]) * 8, uint32_t(q.shape[1]), ptr<uint32_t>(role + ".qweight"),
            ptr<uint16_t>(role + ".scales")};
  }
  size_t bytes(const std::string& n) const { return info(n).end - info(n).begin; }
};

void same_bytes(const Fixture& fx, const std::string& name, const void* got, size_t got_bytes) {
  CHECK_EQ(got_bytes, fx.bytes(name));
  if (std::memcmp(got, fx.ptr<uint8_t>(name), got_bytes) != 0) {
    std::fprintf(stderr, "agnes_fold_test: %s differs from the Python fold\n", name.c_str());
    std::exit(1);
  }
}

void same_packed(const Fixture& fx, const std::string& role, const loader::PackedInt4& p) {
  same_bytes(fx, role + ".qweight", p.qweight.data(), p.qweight.size() * 4);
  same_bytes(fx, role + ".scales", p.scales.data(), p.scales.size() * 2);
}

template <class F>
bool throws(F f) {
  try {
    f();
  } catch (const std::runtime_error&) {
    return true;
  }
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  const Fixture fx(argc > 1 ? argv[1] : "tests/loader/agnes_fold_fixture.safetensors");

  // 1. the three folds
  const loader::PackedInt4 gate = loader::fold_n(fx.view("gate"), fx.view("gate_p"));
  const loader::PackedInt4 up = loader::fold_n(fx.view("up"), fx.view("up_p"));
  const loader::PackedInt4 down = loader::fold_k(fx.view("down"), fx.view("down_p"));
  CHECK_EQ(gate.K, fx.view("gate").K);
  CHECK_EQ(gate.N, fx.view("gate").N + fx.view("gate_p").N);
  CHECK_EQ(down.K, fx.view("down").K + fx.view("down_p").K);
  CHECK_EQ(down.N, fx.view("down").N);
  same_packed(fx, "gate_fold", gate);
  same_packed(fx, "up_fold", up);
  same_packed(fx, "down_fold", down);

  // 2. gate'||up' as load_linear assembles it (Fuse::Interleave16, layout 0)
  const common::Part pg{gate.qweight.data(), gate.scales.data(), gate.N};
  const common::Part pu{up.qweight.data(), up.scales.data(), up.N};
  const std::vector<common::ColSource> cols = common::cols_interleave16(pg, pu);
  const uint32_t K = gate.K, N = 2 * gate.N;
  CHECK_EQ(cols.size(), size_t(N));
  std::vector<uint32_t> q(size_t(K / 8) * N);
  std::vector<uint16_t> s(size_t(K / 64) * N);
  common::repack_int4_layout0_cols(K, N, cols, q.data(), s.data());
  same_bytes(fx, "gate_up_l0.qweight", q.data(), q.size() * 4);
  same_bytes(fx, "gate_up_l0.scales", s.data(), s.size() * 2);

  // 3. refusals
  CHECK(throws([&] { loader::fold_n(fx.view("gate"), fx.view("down_p")); }));   // K 128 vs 64
  CHECK(throws([&] { loader::fold_k(fx.view("down"), fx.view("gate_p")); }));   // N 128 vs 64
  loader::PackedInt4View ragged = fx.view("down");
  ragged.K = 8;   // one packed row: not a whole g64 group
  CHECK(throws([&] { loader::fold_k(ragged, fx.view("down_p")); }));

  std::printf("agnes_fold_test OK: gate' %ux%u, up' %ux%u, down' %ux%u and gate'||up' (layout 0) "
              "byte-identical to tools/oracle/agnes_fold.py\n",
              gate.K, gate.N, up.K, up.N, down.K, down.N);
  return 0;
}
