// 3c: production decode/verify anatomy; no engine or kernel changes.
// probe_longctx_decode SNAPSHOT KV [device=0] [round=0] [depths=4096,32768,81920,150000]
//                       [max_len=150272] [wall_calls=128] [profile_calls=32]
// Use a card lock for smoke, or a whole-box lock and an explicit device for timings.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/queue.h"
#include "loader/loader.h"
#include "loader/snapshot.h"
#include "runtime/capture.h"
#include "runtime/control.h"
#include "runtime/engine.h"

namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
std::vector<uint32_t> depths_of(const std::string& text) {
  std::vector<uint32_t> out;
  size_t start = 0;
  while (start < text.size()) {
    const size_t end = text.find(',', start);
    out.push_back(std::stoul(text.substr(start, end - start)));
    if (end == std::string::npos) break;
    start = end + 1;
  }
  if (out.empty() || out.front() == 0 || !std::is_sorted(out.begin(), out.end()) ||
      std::adjacent_find(out.begin(), out.end()) != out.end())
    throw std::runtime_error("depths must be positive and strictly increasing");
  return out;
}
// Host checkpoints keep genuine prefix state between depths. Copies are outside timing.
struct SavedMem {
  l0::Mem& target;
  std::vector<uint8_t> bytes;
  SavedMem(l0::CmdList& imm, l0::Mem& mem) : target(mem), bytes(mem.size()) {
    transfer(imm, bytes.data(), target.ptr());
  }
  void transfer(l0::CmdList& imm, void* dst, const void* src) {
    constexpr size_t chunk = 32ul << 20;
    for (size_t off = 0; off < bytes.size(); off += chunk)
      imm.copy(static_cast<uint8_t*>(dst) + off, static_cast<const uint8_t*>(src) + off,
               std::min(chunk, bytes.size() - off));
  }
  void restore(l0::CmdList& imm) { transfer(imm, target.ptr(), bytes.data()); }
};
struct Arm {
  uint32_t M;
  runtime::ProfileEvents prof;
  runtime::CapturedStep wall, instrumented;
  Arm(l0::Context& ctx, runtime::Engine& engine, uint32_t rows, uint32_t depth)
      : M(rows), prof(ctx), wall(capture(ctx, engine, rows, depth, nullptr)),
        instrumented(capture(ctx, engine, rows, depth, &prof)) {
    if (wall.labels != instrumented.labels || wall.kernel_count != prof.events.size())
      throw std::runtime_error("profile capture changed launches");
  }
  static runtime::CapturedStep capture(l0::Context& ctx, runtime::Engine& e, uint32_t M,
                                       uint32_t depth, runtime::ProfileEvents* prof) {
    if (M == 1) return runtime::build(ctx, e.model(), e.buffers(), nullptr, prof);
    const auto& md = *e.model().desc;
    const bool deep = runtime::verify_attn_kind(M, depth + 1,
        e.kv_cache() == runtime::KvCache::Int8, md.fa_q_heads, md.fa_kv_heads) == runtime::VerifyAttn::Dpas;
    const auto options = deep ? runtime::attn_forced(runtime::VerifyAttnForce::Dpas) : runtime::CaptureOptions{};
    return runtime::build_verify(ctx, e.model(), e.buffers(), *e.mtp_buffers(), M, prof, options);
  }
};

// Optional probe-only snapshots avoid repeating long prefill between paired
// executable builds. Engine's existing state/KV format is unchanged; metadata
// binds the cache to the model snapshot, prompt, capacity and cache form.
struct PrefixFiles {
  std::string stem, identity;
  PrefixFiles(const std::string& directory, const std::string& model, const char* kv,
              uint32_t capacity, uint32_t depth, const std::vector<uint32_t>& ids,
              runtime::Engine& engine) {
    if (directory.empty()) return;
    std::filesystem::create_directories(directory);
    stem = directory + "/" + kv + "-" + std::to_string(depth);
    uint64_t hash = 14695981039346656037ull;
    for (uint32_t id : ids) { hash ^= id; hash *= 1099511628211ull; }
    identity = "3f-prefix-v1\n" + loader::resolve_snapshot(model) + "\n" + kv + "\n" +
        std::to_string(capacity) + " " + std::to_string(depth) + " " + std::to_string(hash) + " " +
        std::to_string(engine.state_bytes()) + " " + std::to_string(engine.kv_bytes(depth)) + "\n";
  }
  void transfer(const std::string& suffix, l0::Mem& memory, bool write) {
    const std::string path = stem + suffix;
    if (write) {
      std::ofstream file(path + ".tmp", std::ios::binary | std::ios::trunc);
      file.write(memory.as<const char>(), std::streamsize(memory.size()));
      file.close();
      if (!file) throw std::runtime_error("cannot save prefix " + path);
      std::filesystem::rename(path + ".tmp", path);
    } else {
      if (std::filesystem::file_size(path) != memory.size())
        throw std::runtime_error("prefix size mismatch " + path);
      std::ifstream file(path, std::ios::binary);
      file.read(memory.as<char>(), std::streamsize(memory.size()));
      if (!file) throw std::runtime_error("cannot read prefix " + path);
    }
  }
  bool restore(l0::Context& ctx, runtime::Engine& engine, uint32_t depth) {
    if (stem.empty() || !std::filesystem::exists(stem + ".meta")) return false;
    std::ifstream file(stem + ".meta");
    const std::string found((std::istreambuf_iterator<char>(file)), {});
    if (found != identity) throw std::runtime_error("prefix metadata mismatch " + stem);
    l0::Mem state(ctx, l0::MemKind::Host, engine.state_bytes());
    l0::Mem kv(ctx, l0::MemKind::Host, engine.kv_bytes(depth));
    transfer(".state", state, false); transfer(".kv", kv, false);
    engine.load_kv(0, depth, kv.ptr()); engine.load_state(state.ptr(), depth);
    return true;
  }
  void save(l0::Context& ctx, runtime::Engine& engine, uint32_t depth) {
    if (stem.empty()) return;
    l0::Mem state(ctx, l0::MemKind::Host, engine.state_bytes());
    l0::Mem kv(ctx, l0::MemKind::Host, engine.kv_bytes(depth));
    engine.save_state(state.ptr()); engine.save_kv(0, depth, kv.ptr());
    transfer(".state", state, true); transfer(".kv", kv, true);
    std::ofstream file(stem + ".meta.tmp"); file << identity; file.close();
    if (!file) throw std::runtime_error("cannot save prefix metadata " + stem);
    std::filesystem::rename(stem + ".meta.tmp", stem + ".meta");
  }
};
}  // namespace

int main(int argc, char** argv) try {
  if (argc < 3 || argc > 10) throw std::runtime_error("usage: probe_longctx_decode SNAPSHOT KV [device] [round] [depths] [max_len] [wall_calls] [profile_calls] [prefix_directory]");
  const auto kv = runtime::parse_kv_cache(argv[2]);
  const uint32_t device = argc > 3 ? std::stoul(argv[3]) : 0;
  const uint32_t round = argc > 4 ? std::stoul(argv[4]) : 0;
  const auto depths = depths_of(argc > 5 ? argv[5] : "4096,32768,81920,150000");
  const uint32_t capacity = argc > 6 ? std::stoul(argv[6]) : 150272;
  const uint32_t calls = argc > 7 ? std::stoul(argv[7]) : 128;
  const uint32_t samples = argc > 8 ? std::stoul(argv[8]) : 32;
  const std::string prefix_directory = argc > 9 ? argv[9] : "";
  if (uint64_t(depths.back()) + std::max(calls, 4u) >= capacity || !calls || !samples)
    throw std::runtime_error("capacity must leave room for calls and four verify rows; counts must be positive");
  std::ifstream input("tests/golden/prompts/long32k.ids");
  std::vector<uint32_t> base;
  uint32_t id;
  while (input >> id) base.push_back(id);
  if (base.empty()) throw std::runtime_error("missing prompt ids");
  std::printf("# 3c kv=%s device=%u round=%u max_len=%u lm_head=int8 warmup=128 wall_calls=%u profile_calls=%u\n",
              argv[2], device, round, capacity, calls, samples);
  std::fflush(stdout);
  l0::Context ctx(device);
  runtime::Engine engine(ctx, loader::load(ctx, argv[1], capacity, true, loader::LmHeadForm::Int8),
                         capacity, false, kv);
  l0::Queue queue(ctx);
  l0::Fence fence(queue);
  auto imm = l0::CmdList::immediate(ctx);
  auto& b = engine.buffers();
  auto* ctl = b.control.as<runtime::Control>();
  auto* mtp = engine.mtp_buffers();
  auto* hctl = mtp->hctl.as<runtime::Control>();
  engine.prepare_prefill();   // identical scratch allocation for cold and restored arms
  for (uint32_t depth : depths) {
    PrefixFiles files(prefix_directory, argv[1], argv[2], capacity, depth, base, engine);
    std::vector<uint32_t> ids;
    for (uint32_t pos = engine.pos(); pos < depth; ++pos) ids.push_back(base[pos % base.size()]);
    const auto prefill_start = Clock::now();
    const bool restored = files.restore(ctx, engine, depth);
    if (!restored) { engine.prefill(ids); files.save(ctx, engine, depth); }
    std::printf("# populated depth=%u prefill_ms=%.3f pos=%u\n", depth, elapsed(prefill_start), engine.pos());
    if (!prefix_directory.empty()) std::printf("# prefix-cache depth=%u restored=%u\n", depth, unsigned(restored));
    std::printf("# memory %s\n", engine.memory_line().c_str());
    std::fflush(stdout);
    if (engine.pos() != depth || ctl->gdn_live != 0) throw std::runtime_error("unexpected prefix state");
    SavedMem state(imm, b.gdn_state), ring(imm, b.conv_ring), control(imm, b.control),
             head_hidden(imm, mtp->hh), head_control(imm, mtp->hctl);
    auto restore = [&] {
      state.restore(imm); ring.restore(imm); control.restore(imm);
      head_hidden.restore(imm); head_control.restore(imm);
    };
    std::unique_ptr<Arm> arms[2];
    arms[0] = std::make_unique<Arm>(ctx, engine, 1, depth);
    arms[1] = std::make_unique<Arm>(ctx, engine, 4, depth);
    auto submit = [&](Arm& arm, bool profile) {
      ctl->pos = depth;
      ctl->n_active = arm.M;
      for (uint32_t r = 0; r < arm.M; ++r) ctl->cur_token[r] = base[(depth + r) % base.size()];
      if (arm.M > 1) {
        hctl->pos = depth - 1;
        hctl->n_active = arm.M;
        for (uint32_t r = 0; r < arm.M; ++r) hctl->cur_token[r] = ctl->cur_token[r];
      }
      if (profile) for (auto& ev : arm.prof.events) ev.reset();
      queue.execute(profile ? arm.instrumented.list : arm.wall.list, &fence);
      fence.wait();
    };
    // Alternate paired chunks within every depth, and reverse their order by outer round.
    double wall_ms[2] = {}, total_us[2] = {}, prep_us[2] = {}, core_us[2] = {}, reduce_us[2] = {}, head_us[2] = {};
    uint32_t attn_launches[2] = {};
    for (unsigned phase = 0; phase < 2; ++phase) {
      for (unsigned order = 0; order < 2; ++order) {
        const unsigned a = (order + round + phase) % 2;
        auto& arm = *arms[a];
        restore();
        for (uint32_t w = 0; w < 128; ++w) submit(arm, false);
        if (phase == 0) {
          const auto start = Clock::now();
          for (uint32_t n = 0; n < calls; ++n) submit(arm, false);
          wall_ms[a] = elapsed(start) / calls;
        } else {
          for (uint32_t n = 0; n < samples; ++n) {
            submit(arm, true);
            uint32_t count = 0;
            for (size_t k = 0; k < arm.prof.events.size(); ++k) {
              const double us = arm.prof.events[k].duration_us();
              if (!(us >= 0) || !std::isfinite(us)) throw std::runtime_error("invalid timestamp");
              total_us[a] += us;
              const std::string& label = arm.instrumented.labels[k];
              const bool prep = label.find(" attn_prep") != std::string::npos;
              const bool core = label.find(" attn_decode") != std::string::npos || label.find(" attn_verify") != std::string::npos;
              const bool reduce = label.find(" attn_reduce") != std::string::npos;
              if (prep) prep_us[a] += us;
              if (core) core_us[a] += us;
              if (reduce) reduce_us[a] += us;
              if (prep || core || reduce) {
                ++count;
                if (label.rfind("L" + std::to_string(engine.model().desc->layers) + " ", 0) == 0) head_us[a] += us;
              }
            }
            if (n == 0) attn_launches[a] = count;
            else if (count != attn_launches[a]) throw std::runtime_error("attention launch count changed");
          }
        }
      }
    }
    for (unsigned a = 0; a < 2; ++a) {
      const auto& arm = *arms[a];
      if (!(wall_ms[a] > 0) || !(total_us[a] > 0) || !attn_launches[a]) throw std::runtime_error("incomplete measurements");
      std::printf("{\"kv\":\"%s\",\"round\":%u,\"depth\":%u,\"M\":%u,\"capacity\":%u,\"launches\":%zu,\"attn_launches\":%u,\"wall_ms\":%.9f,\"gpu_ms\":%.9f,\"prep_ms\":%.9f,\"core_ms\":%.9f,\"reduce_ms\":%.9f,\"head_attn_ms\":%.9f,\"rows_per_s\":%.6f}\n",
          argv[2], round, depth, arm.M, capacity, arm.wall.kernel_count, attn_launches[a], wall_ms[a],
          total_us[a] / samples / 1000, prep_us[a] / samples / 1000, core_us[a] / samples / 1000,
          reduce_us[a] / samples / 1000, head_us[a] / samples / 1000, arm.M * 1000.0 / wall_ms[a]);
      for (size_t k = 0; k < arm.wall.labels.size(); ++k)
        if (arm.wall.labels[k].find(" attn_") != std::string::npos)
          std::printf("# kernel M=%u %s\n", arm.M, arm.wall.labels[k].c_str());
    }
    restore();
    std::fflush(stdout);
  }
  std::puts("ALLDONE rc=0");
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "probe_longctx_decode: %s\n", e.what());
  return 1;
}
