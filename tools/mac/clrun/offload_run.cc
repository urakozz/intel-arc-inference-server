// offload_run - spec 22a (plan Task 2 Step 3): the table-addressed expert kernels (tools/probe/probe_offload_moe.cl,
// q4_moe.cl with its two address macros as a u64 pointer-table read) against q4_moe.cl's own plain-load build, on the
// Mac's OpenCL GPU at the real shapes (512 experts, top 10, hidden 2560, intermediate 640, the int4 shared expert).
// INDICATIVE ONLY: clrun.h says what this can and cannot show.
//
//   offload_run      no arguments
//
// OpenCL 1.2 passes no device pointer, so the table's addresses come from the device itself: a one-line kernel
// writes `(ulong)(intptr_t)` of the arena buffer it is given (read before and after the expert launches - an address
// that moved makes the run inconclusive, a FAIL). The arena is ONE buffer holding the table (u64 [1024]: gate||up
// pointers [0, 512), down pointers [512, 1024)) and the expert blocks, so every block a table entry names lies in a
// buffer the launch binds; the down launch is bound at a sub-buffer starting at table + 512 entries (4096 B) - the
// probe's `table + 512` (Apple maps a sub-buffer at an address of its own holding the arena's bytes from the origin
// on: the down launch reads its table there and dereferences the arena's addresses). Three layouts, each bitwise
// q4_moe.cl's direct build over 16 allocated experts and a route row of 10 of them (ids < 16) with the shared slot, h
// and y, and a control:
//   identity   expert e at arena slot e
//   permuted   expert e at arena slot perm[e] (a fixed shuffle; the table follows it)
//   host       the permuted layout in a CL_MEM_USE_HOST_PTR arena (page-aligned host memory: the zero-copy read)
//   control    the identity layout with the table entries of ids 3 and 15 swapped: h and y MUST differ (the check sees)
// DN_KS follows the device's work-group cap as qwen4exp_run's does (2 when 352 lanes fit, else 1): both builds use it.
#define CL_SILENCE_DEPRECATION
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

#include "clrun.h"

namespace {

constexpr uint32_t kE = 512, kK = 10, kSlots = 11, kD = 2560, kI = 640, kAlloc = 16;
constexpr uint32_t kTileU32 = 136;
constexpr size_t kGuWords = size_t(2 * kI / 16) * (kD / 64) * kTileU32;   // 435,200 (1,740,800 B)
constexpr size_t kDnWords = size_t(kD / 16) * (kI / 64) * kTileU32;       // 217,600 (870,400 B)
constexpr size_t kTableBytes = size_t(2) * kE * 8;                        // 8192
constexpr size_t kDownTable = size_t(kE) * 8;                             // the down launch's origin: 4096
constexpr uint32_t kRouteWords = 32, kIds = 0, kWeights = 16, kSharedGate = 26;

int failures = 0;

uint64_t splitmix(uint64_t& x) {
  uint64_t z = (x += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
std::vector<uint32_t> blocks(size_t words, uint64_t seed) {   // random nibbles, f16 scales in [2^-8, 2^-7)
  std::vector<uint32_t> w(words);
  uint64_t x = seed;
  for (size_t i = 0; i < words; ++i) {
    const uint64_t r = splitmix(x);
    w[i] = i % kTileU32 < 128 ? uint32_t(r) : (0x1C00u | uint32_t(r & 0x3FF)) | ((0x1C00u | uint32_t((r >> 10) & 0x3FF)) << 16);
  }
  return w;
}
uint32_t f32_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}
float rf(float f) { return clrun::bf16_to_f32(clrun::f32_to_bf16(f)); }

void check(cl_int e, const char* what) {
  if (e != CL_SUCCESS) throw clrun::Error(std::string(what) + " -> " + std::to_string(e));
}

// The device's view of a buffer's address: a one-kernel program written to a temporary file (clrun builds files).
struct AddrProbe {
  std::string path;
  std::unique_ptr<clrun::Program> prog;
  explicit AddrProbe(clrun::Device& dev) {
    char tmpl[] = "/tmp/offload_run_addr_XXXXXX";
    const int fd = mkstemp(tmpl);
    if (fd < 0) throw clrun::Error("mkstemp");
    const char* src =
        "__kernel void po_addr(__global const uint* a, __global ulong* out) {\n"
        "  if (get_global_id(0) == 0) out[0] = (ulong)(intptr_t)a;\n"
        "}\n";
    if (write(fd, src, std::strlen(src)) != ssize_t(std::strlen(src))) throw clrun::Error("write");
    close(fd);
    path = tmpl;
    prog = std::make_unique<clrun::Program>(dev, path, std::vector<std::string>{});
  }
  ~AddrProbe() { unlink(path.c_str()); }
  uint64_t of(clrun::Device& dev, cl_mem m) {
    clrun::Buffer out(dev, 8);
    prog->run("po_addr", {1}, {1}, m, out);
    return out.read<uint64_t>()[0];
  }
};

struct Case {
  const char* name;
  bool permuted, host, swap = false;
};

}  // namespace

int main() {
  try {
    clrun::Device dev;
    size_t cap = 0;
    clGetDeviceInfo(dev.id(), CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof cap, &cap, nullptr);
    const uint32_t dn_ks = cap >= 16 * kSlots * 2 ? 2 : 1;
    cl_uint align_bits = 0;
    clGetDeviceInfo(dev.id(), CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof align_bits, &align_bits, nullptr);
    std::printf("offload_run: device %s, CL_DEVICE_MAX_WORK_GROUP_SIZE %zu -> DN_KS=%u%s, sub-buffer alignment %u B\n",
                dev.name().c_str(), cap, dn_ks, dn_ks == 2 ? " (the B70 build's)" : " (the B70 build: 2)", align_bits / 8);
    if (align_bits == 0 || kDownTable % (align_bits / 8) != 0)
      throw clrun::Error("the down table's sub-buffer origin 4096 is not aligned for this device");
    const std::vector<std::string> defs = {"M=1", "MOE_E=512", "MOE_K=10", "HIDDEN=2560", "INTER=640", "UP_KS=4",
                                           "DN_KS=" + std::to_string(dn_ks), "SHARED_BF16=0"};
    clrun::Program direct(dev, "src/kernels/qwen4exp/q4_moe.cl", defs);
    clrun::Program table(dev, "tools/probe/probe_offload_moe.cl", defs);
    AddrProbe addr(dev);

    // 16 experts' blocks, the shared expert, x, one route row of ten of them
    std::vector<std::vector<uint32_t>> gu(kAlloc), dn(kAlloc);
    for (uint32_t e = 0; e < kAlloc; ++e) {
      gu[e] = blocks(kGuWords, 0x0FF10000ull + e);
      dn[e] = blocks(kDnWords, 0x0FF20000ull + e);
    }
    std::vector<uint32_t> gu_all, dn_all;
    for (uint32_t e = 0; e < kAlloc; ++e) {
      gu_all.insert(gu_all.end(), gu[e].begin(), gu[e].end());
      dn_all.insert(dn_all.end(), dn[e].begin(), dn[e].end());
    }
    const std::vector<uint32_t> sh_gu = blocks(kGuWords, 0x0FF30000), sh_dn = blocks(kDnWords, 0x0FF40000);
    std::vector<uint16_t> x(kD);
    uint64_t sx = 0x0FF50000;
    for (auto& v : x) v = clrun::f32_to_bf16((float(splitmix(sx) % 2001) - 1000.0f) / 2000.0f);
    const uint32_t ids[kK] = {3, 15, 0, 7, 12, 1, 9, 14, 5, 10};
    const float w[kK] = {0.21875f, 0.15625f, 0.125f, 0.109375f, 0.09375f, 0.0859375f, 0.078125f, 0.0625f, 0.046875f, 0.0390625f};
    std::vector<uint32_t> row(kRouteWords, 0);
    for (uint32_t k = 0; k < kK; ++k) {
      row[kIds + k] = ids[k];
      row[kWeights + k] = f32_bits(rf(w[k]));
    }
    row[kSharedGate] = f32_bits(rf(0.40625f));
    clrun::Buffer rb(dev, row), xb(dev, x), gub(dev, gu_all), dnb(dev, dn_all), sgu(dev, sh_gu), sdn(dev, sh_dn);
    const size_t wg_up = 256, groups_up = size_t(kSlots) * (2 * kI / 16 / 4), wg_dn = size_t(16) * kSlots * dn_ks;

    // the reference: q4_moe.cl's direct build (plain loads under the shim's emulation)
    clrun::Buffer h_ref_b(dev, size_t(kSlots) * kI * 2), y_ref_b(dev, size_t(kD) * 2);
    direct.run("q4_moe_gate_up", {groups_up * wg_up, 1}, {wg_up, 1}, rb, xb, gub, sgu, h_ref_b);
    direct.run("q4_moe_down", {(kD / 16) * wg_dn, 1}, {wg_dn, 1}, rb, h_ref_b, dnb, sdn, y_ref_b);
    const std::vector<uint16_t> h_ref = h_ref_b.read<uint16_t>(), y_ref = y_ref_b.read<uint16_t>();

    std::vector<uint32_t> perm(kAlloc);
    for (uint32_t e = 0; e < kAlloc; ++e) perm[e] = e;
    std::mt19937 rng(2210);
    std::shuffle(perm.begin(), perm.end(), rng);
    const size_t gu_off = kTableBytes, dn_off = gu_off + size_t(kAlloc) * kGuWords * 4;
    const size_t arena_bytes = dn_off + size_t(kAlloc) * kDnWords * 4;
    for (const Case& c : {Case{"identity", false, false}, Case{"permuted", true, false}, Case{"host", true, true},
                          Case{"control", false, false, true}}) {
      void* host_mem = nullptr;
      if (c.host && posix_memalign(&host_mem, 4096, arena_bytes) != 0) throw clrun::Error("posix_memalign");
      cl_int e = CL_SUCCESS;
      cl_mem arena = clCreateBuffer(dev.context(), CL_MEM_READ_WRITE | (c.host ? CL_MEM_USE_HOST_PTR : 0), arena_bytes,
                                    host_mem, &e);
      check(e, "clCreateBuffer (arena)");
      cl_buffer_region reg{kDownTable, arena_bytes - kDownTable};
      cl_mem arena_dn = clCreateSubBuffer(arena, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &reg, &e);
      check(e, "clCreateSubBuffer (table + 512)");
      const uint64_t base = addr.of(dev, arena), base_dn = addr.of(dev, arena_dn);
      // the arena: the blocks at their slots, the table naming them (ids >= 16 at slot 0: never read)
      std::vector<uint8_t> img(arena_bytes, 0);
      std::vector<uint64_t> t(size_t(2) * kE);
      for (uint32_t id = 0; id < kE; ++id) {
        const uint32_t ex = id < kAlloc ? id : 0, slot = c.permuted ? perm[ex] : ex;
        t[id] = base + gu_off + size_t(slot) * kGuWords * 4;
        t[kE + id] = base + dn_off + size_t(slot) * kDnWords * 4;
      }
      for (uint32_t ex = 0; ex < kAlloc; ++ex) {
        const uint32_t slot = c.permuted ? perm[ex] : ex;
        std::memcpy(img.data() + gu_off + size_t(slot) * kGuWords * 4, gu[ex].data(), kGuWords * 4);
        std::memcpy(img.data() + dn_off + size_t(slot) * kDnWords * 4, dn[ex].data(), kDnWords * 4);
      }
      if (c.swap) {
        std::swap(t[3], t[15]);
        std::swap(t[kE + 3], t[kE + 15]);
      }
      std::memcpy(img.data(), t.data(), kTableBytes);
      check(clEnqueueWriteBuffer(dev.queue(), arena, CL_TRUE, 0, arena_bytes, img.data(), 0, nullptr, nullptr),
            "clEnqueueWriteBuffer (arena)");
      clrun::Buffer hb(dev, size_t(kSlots) * kI * 2), yb(dev, size_t(kD) * 2);
      table.run("q4_moe_gate_up", {groups_up * wg_up, 1}, {wg_up, 1}, rb, xb, arena, sgu, hb);
      table.run("q4_moe_down", {(kD / 16) * wg_dn, 1}, {wg_dn, 1}, rb, hb, arena_dn, sdn, yb);
      const uint64_t base_after = addr.of(dev, arena);
      const std::vector<uint16_t> h = hb.read<uint16_t>(), y = yb.read<uint16_t>();
      size_t dh = 0, dy = 0;
      for (size_t i = 0; i < h.size(); ++i) dh += h[i] != h_ref[i];
      for (size_t i = 0; i < y.size(); ++i) dy += y[i] != y_ref[i];
      // The table's absolute addresses are the arena's: it must not move across the launches. (Apple's OpenCL maps a
      // sub-buffer at its own address - seen: arena + its size - holding the arena's bytes from the origin on; the
      // down launch reads its table there and dereferences the arena's addresses, as the probe's table + 512 does.)
      const bool stable = base_after == base;
      const bool ok = stable && (c.swap ? dh != 0 && dy != 0 : dh == 0 && dy == 0);
      std::printf("  %-4s table %-8s: h [11][640] %zu / %zu differ, y [2560] %zu / %zu differ; arena 0x%llx (after the "
                  "launches 0x%llx: %s), the down sub-buffer seen at 0x%llx\n",
                  ok ? "ok" : "FAIL", c.name, dh, h.size(), dy, y.size(), (unsigned long long)base,
                  (unsigned long long)base_after, stable ? "stable" : "MOVED - inconclusive", (unsigned long long)base_dn);
      if (!ok) ++failures;
      clReleaseMemObject(arena_dn);
      clReleaseMemObject(arena);
      std::free(host_mem);
    }
  } catch (const std::exception& e) {
    std::printf("offload_run: ERROR %s\n", e.what());
    return 1;
  }
  if (failures == 0) std::printf("offload_run: q4_moe table exact (identity, permuted, host-allocated: bitwise the direct build)\n");
  std::printf("offload_run: %s (%d disagreement%s)\n", failures ? "FAIL" : "OK", failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
