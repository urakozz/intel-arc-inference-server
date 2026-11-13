// Spec 19 v2: production DFlash list costs, and engine taps on recorded continuations.
// probe_dflash_iter TARGET DRAFTER [--depth 4096] [--calls 16] [--rounds 3]
// probe_dflash_iter TARGET DRAFTER --engine-taps SOURCES.json OUT_DIR [--anchors 16]
// SOURCES: [{name, prompt_ids, cont_ids}]. Taps use dump_taps.py's safetensors format.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "l0/fence.h"
#include "l0/queue.h"
#include "loader/loader.h"
#include "runtime/dflash/dflash_drafter.h"
#include "server/spec_accept.h"

namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
namespace df = runtime::dflash;

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

template <class F> double timed(F&& f) {
  const auto start = Clock::now();
  f();
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void write_dump(const std::string& path, const Json& meta, const std::vector<int32_t>& ids,
                const std::vector<int32_t>& greedy, const std::vector<std::vector<uint16_t>>& taps,
                const std::vector<uint32_t>& layers, uint32_t hidden) {
  Json header;
  header["__metadata__"] = meta;
  std::vector<std::pair<const void*, size_t>> data;
  size_t offset = 0;
  auto tensor = [&](const std::string& name, const char* dtype, Json shape, const void* ptr, size_t bytes) {
    header[name] = {{"dtype", dtype}, {"shape", shape}, {"data_offsets", {offset, offset + bytes}}};
    data.emplace_back(ptr, bytes);
    offset += bytes;
  };
  tensor("ids", "I32", {ids.size()}, ids.data(), ids.size() * 4);
  tensor("greedy", "I32", {greedy.size()}, greedy.data(), greedy.size() * 4);
  for (size_t i = 0; i < layers.size(); ++i)
    tensor("tap.L" + std::to_string(layers[i]), "BF16", {greedy.size(), hidden}, taps[i].data(), taps[i].size() * 2);
  std::string text = header.dump();
  text.append((8 - text.size() % 8) % 8, ' ');
  const uint64_t length = text.size();
  std::ofstream out(path, std::ios::binary);
  out.exceptions(std::ios::failbit | std::ios::badbit);
  out.write(reinterpret_cast<const char*>(&length), 8);
  out.write(text.data(), text.size());
  for (const auto& [ptr, bytes] : data) out.write(static_cast<const char*>(ptr), bytes);
}

void engine_taps(runtime::Engine& e, df::DFlashDrafter& d, const Json& sources, const std::string& dir,
                 unsigned anchors) {
  std::filesystem::create_directories(dir);
  l0::CmdList imm = l0::CmdList::immediate(e.context());
  const uint32_t hidden = e.model().desc->hidden;
  const auto& layers = d.tap_layers();
  const size_t width = size_t(layers.size()) * hidden;
  l0::Mem saved(e.context(), l0::MemKind::Host, e.state_bytes());
  std::ofstream records(dir + "/engine.jsonl");
  std::ofstream live(dir + "/live.jsonl");
  records.exceptions(std::ios::failbit | std::ios::badbit);
  live.exceptions(std::ios::failbit | std::ios::badbit);
  records << Json{{"run", {{"format", loader::draft_format_name(d.weights().format)}, {"head", "int8"},
                           {"draft_vocab", 0}, {"candidate_head", bool(d.weights().candidate_head)}, {"vocab_used", e.model().desc->vocab_used}, {"v_ids", Json::array()}}}}.dump() << '\n';
  for (const auto& source : sources) {
    const std::string name = source.at("name");
    const auto prompt = source.at("prompt_ids").get<std::vector<uint32_t>>();
    const auto cont = source.at("cont_ids").get<std::vector<uint32_t>>();
    std::vector<int32_t> ids(prompt.begin(), prompt.end());
    ids.insert(ids.end(), cont.begin(), cont.end());
    const uint32_t n = ids.size(), np = prompt.size(), from = np > d.window() ? np - d.window() : 0;
    if (np == 0 || cont.size() < 8 || n + 8 > e.max_len())
      throw std::runtime_error(name + ": need a prompt, at least 8 continuation ids, and room in max_len");
    const auto eos = source.at("eos_ids").get<std::vector<uint32_t>>();
    e.reset();
    e.prefill(prompt);
    std::vector<uint32_t> generated;
    uint32_t iterations = 0;
    bool ended = false;
    while (generated.size() < 256 && !ended) {
      const uint32_t anchor = e.control()->cur_token[0];
      const auto out = d.draft(e, 7, 0.0f, 0);
      e.verify(7);
      const auto accepted = server::accept_greedy(e.verify_ids(), out.ids, 7);
      ++iterations;
      for (uint32_t j = 0; j <= accepted.accepted && generated.size() < 256; ++j) {
        const uint32_t token = j ? out.ids[j - 1] : anchor;
        generated.push_back(token);
        if (std::find(eos.begin(), eos.end(), token) != eos.end()) { ended = true; break; }
      }
      e.commit(accepted.accepted, accepted.next_token);
    }
    live << Json{{"name", name}, {"ids", generated}, {"iterations", iterations}, {"eos", ended}}.dump() << '\n';
    std::printf("DFLASH_EOS %s ids=%zu iterations=%u ids_per_iteration=%.5f eos=%u\n", name.c_str(),
                generated.size(), iterations, double(generated.size()) / iterations, unsigned(ended));
    std::vector<uint32_t> ps;
    for (unsigned i = 0; i < anchors; ++i) ps.push_back(np + uint64_t(i) * (n - 8 - np) / std::max(1u, anchors - 1));
    std::sort(ps.begin(), ps.end());
    ps.erase(std::unique(ps.begin(), ps.end()), ps.end());
    std::vector<std::vector<uint16_t>> taps(layers.size(), std::vector<uint16_t>(size_t(n - from) * hidden));
    std::vector<int32_t> greedy(n - from);
    std::vector<uint16_t> row(width);
    e.reset();
    if (from) e.prefill(std::vector<uint32_t>(ids.begin(), ids.begin() + from));
    for (uint32_t p = from; p < n; ++p) {
      e.control()->cur_token[0] = ids[p];
      if (std::binary_search(ps.begin(), ps.end(), p)) {
        e.save_state(saved.ptr());
        for (uint32_t k = 1; k <= 7; ++k) {
          e.load_state(saved.ptr(), p);
          e.control()->cur_token[0] = ids[p];
          const df::DFlashDraftOut out = d.draft(e, k, 0.0f, 0);
          e.verify(k);
          const std::vector<uint32_t> target(e.verify_ids(), e.verify_ids() + k + 1);
          const auto accept = server::accept_greedy(target.data(), out.ids, k);
          Json rec = {{"dump", name}, {"p", p}, {"anchor", ids[p]}, {"k", k},
                      {"ids", std::vector<uint32_t>(out.ids, out.ids + k)}, {"target", target},
                      {"accepted", accept.accepted}};
          for (uint32_t j = 0; j < k; ++j) {
            rec["cand"].push_back(std::vector<uint32_t>(out.cand[j], out.cand[j] + 16));
            rec["unary"].push_back(std::vector<float>(out.unary[j], out.unary[j] + 16));
            rec["scores"].push_back(std::vector<float>(out.scores[j], out.scores[j] + 16));
          }
          records << rec.dump() << '\n';
        }
        e.load_state(saved.ptr(), p);
        e.control()->cur_token[0] = ids[p];
      }
      e.verify(0);
      greedy[p - from] = e.verify_ids()[0];
      imm.copy(row.data(), d.taps().ptr(), row.size() * 2);
      for (size_t i = 0; i < layers.size(); ++i)
        std::copy_n(row.data() + i * hidden, hidden, taps[i].data() + size_t(p - from) * hidden);
      e.commit(0, p + 1 < n ? uint32_t(ids[p + 1]) : uint32_t(greedy.back()));
    }
    std::string file = name;
    for (size_t at = 0; (at = file.find('/', at)) != std::string::npos; at += 2) file.replace(at, 1, "__");
    std::string ls;
    for (auto layer : layers) ls += (ls.empty() ? "" : ",") + std::to_string(layer);
    write_dump(dir + "/" + file + ".taps.safetensors",
               {{"name", name}, {"n_prompt", std::to_string(np)}, {"n", std::to_string(n)},
                {"tap_from", std::to_string(from)}, {"layers", ls}, {"head_stats", "greedy_only"}},
               ids, greedy, taps, layers, hidden);
    std::printf("ENGINE_TAPS %s prompt=%u n=%u from=%u anchors=%zu\n", name.c_str(), np, n, from, ps.size());
    std::fflush(stdout);
  }
}

void costs(runtime::Engine& e, df::DFlashDrafter& d, uint32_t depth, unsigned calls, unsigned rounds) {
  if (!depth) throw std::runtime_error("cost depth must be positive");
  std::ifstream input("tests/golden/prompts/long32k.ids");
  std::vector<uint32_t> file;
  for (uint32_t id; input >> id;) file.push_back(id);
  if (file.empty()) throw std::runtime_error("no long32k.ids");
  std::vector<uint32_t> prompt(depth);
  for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = file[i % file.size()];
  if (depth > 1) e.prefill(std::vector<uint32_t>(prompt.begin(), prompt.end() - 1));
  else e.reset();
  l0::Mem saved(e.context(), l0::MemKind::Host, e.state_bytes());
  e.save_state(saved.ptr());
  // Recreate the last prompt row outside the clock, leaving its projection pending for a steady-state batch.
  auto restore = [&](bool defer = false) {
    e.load_state(saved.ptr(), depth - 1);
    e.control()->cur_token[0] = prompt.back();
    e.control()->n_active = 1;
    e.verify(0);
    e.commit(0, e.verify_ids()[0], defer);
  };
  auto plain = runtime::build(e.context(), e.model(), e.buffers());
  restore();
  e.verify(0);
  const uint32_t reference = e.verify_ids()[0];
  restore();
  e.replay_aux(plain.list);
  if (e.pos() != depth + 1 || e.control()->out_token[0] != reference)
    throw std::runtime_error("plain baseline did not consume one active row or match verify(1)");
  std::printf("DFLASH_CALIBRATION depth=%u plain row matches verify(1), pos=%u\n", depth, e.pos());
  l0::Queue queue(e.context());
  l0::Fence fence(queue);
  auto empty = l0::CmdList::regular(e.context());
  empty.close();
  std::vector<std::string> names = {"draft"};
  std::vector<std::function<void()>> arms = {[&]() { d.draft(e, 7, 0.0f, 0); }};
  for (uint32_t m = 1; m <= 8; ++m) {
    names.push_back("verify" + std::to_string(m));
    arms.push_back([&, m]() {
      for (uint32_t j = 1; j < m; ++j) e.control()->cur_token[j] = file[j % file.size()];
      e.verify(m - 1);
    });
  }
  names.push_back("projection");
  arms.push_back([&]() { e.replay_aux(d.projection(depth - 1, 1)); });
  names.push_back("submit_fence");
  arms.push_back([&]() { queue.execute(empty, &fence); fence.wait(); });
  names.push_back("iteration7");
  arms.push_back([&]() {
    const uint64_t before = e.queue_submissions();
    const auto out = d.draft_with([&](l0::CmdList& list) { e.verify(7, &list); }, 7, 0.0f, 0);
    const auto accept = server::accept_greedy(e.verify_ids(), out.ids, 7);
    e.commit(accept.accepted, accept.next_token, true);
    if (e.queue_submissions() != before + 1) throw std::runtime_error("DFlash iteration used more than one submit");
  });
  std::printf("DFLASH_ITER depth=%u calls=%u rounds=%u draft_launches=%zu projection_launches=%zu\n",
              depth, calls, rounds, d.draft_step()->kernel_count, d.project_step().kernel_count);
  for (size_t a = 0; a < arms.size(); ++a) {
    std::vector<double> arm_ms, plain_ms, ratios;
    for (unsigned r = 0; r <= rounds; ++r) {
      double av = 0, pv = 0;
      for (unsigned c = 0; c < calls; ++c) {
        auto one = [&](bool baseline) {
          restore(!baseline && names[a] == "iteration7");
          const double ms = timed([&]() { if (baseline) e.replay_aux(plain.list); else arms[a](); });
          (baseline ? pv : av) += ms;
        };
        one(r % 2 == 0);
        one(r % 2 != 0);
      }
      if (r) {
        arm_ms.push_back(av / calls);
        plain_ms.push_back(pv / calls);
        ratios.push_back(av / pv);
      }
    }
    std::printf("DFLASH_COST depth=%u arm=%s ms=%.6f plain_ms=%.6f ratio=%.6f\n", depth, names[a].c_str(),
                median(arm_ms), median(plain_ms), median(ratios));
    std::fflush(stdout);
  }
  // Profile separate lists: timestamp signals must not contaminate the wall-time medians above.
  runtime::ProfileEvents dp(e.context()), pp(e.context());
  runtime::CaptureOptions opt;
  opt.profile = &dp;
  auto draft = df::build_dflash_draft(e.context(), d.weights(), df::dflash_target(e.model()), d.buffers(),
                                     e.buffers().control, opt);
  opt.profile = &pp;
  auto project = df::build_dflash_project(e.context(), d.weights(), d.buffers(), opt);
  for (unsigned which = 0; which < 2; ++which) {
    auto& prof = which ? pp : dp;
    auto& list = which ? project : draft;
    std::map<std::string, std::vector<double>> families;
    std::vector<std::vector<double>> launches(prof.events.size());
    std::vector<double> device, walls;
    for (unsigned r = 0; r <= rounds; ++r) {
      restore();
      d.buffers().ctl()->k = 7;
      d.projection(depth - 1, 1);
      for (auto& event : prof.events) event.reset();
      const double wall = timed([&]() { e.replay_aux(list.list); });
      std::map<std::string, double> sums;
      for (size_t i = 0; i < prof.events.size(); ++i) {
        const std::string& label = list.labels[i];
        const auto start = label.find(' ') + 1, end = label.find(' ', start);
        const double ms = prof.events[i].duration_us() / 1000.0;
        sums[label.substr(start, end - start)] += ms;
        if (r) launches[i].push_back(ms);
      }
      if (r) {
        double sum = 0;
        for (const auto& [family, ms] : sums) { families[family].push_back(ms); sum += ms; }
        device.push_back(sum);
        walls.push_back(wall);
      }
    }
    for (const auto& [family, ms] : families)
      std::printf("DFLASH_PROFILE depth=%u list=%s family=%s ms=%.6f\n", depth, which ? "projection" : "draft",
                  family.c_str(), median(ms));
    for (size_t i = 0; i < launches.size(); ++i)
      std::printf("DFLASH_LAUNCH depth=%u list=%s index=%zu ms=%.6f label=%s\n", depth,
                  which ? "projection" : "draft", i, median(launches[i]), list.labels[i].c_str());
    std::printf("DFLASH_PROFILE_TOTAL depth=%u list=%s device_ms=%.6f wall_ms=%.6f (instrumented)\n", depth,
                which ? "projection" : "draft", median(device), median(walls));
  }
  runtime::ProfileEvents vp(e.context());
  opt.profile = &vp;
  opt.profile_m = 8;
  opt.profile_keys = depth + 1;
  opt.profile_projection = &project.list;
  e.recapture_verify(opt);
  std::vector<double> gap, projection_host, outside, projection_outside;
  for (unsigned r = 0; r <= rounds; ++r) {
    restore(true);
    d.buffers().ctl()->k = 7;
    for (auto* prof : {&dp, &vp, &pp})
      for (auto& event : prof->events) event.reset();
    const double wall = timed([&]() {
      e.verify(7, &draft.list);
      const auto& out = *d.buffers().readback();
      const auto accept = server::accept_greedy(e.verify_ids(), out.ids, 7);
      e.commit(accept.accepted, accept.next_token, true);
    });
    double device = 0, projection_ms = 0;
    for (auto* prof : {&dp, &vp, &pp})
      for (auto& event : prof->events) {
        const double ms = event.duration_us() / 1000.0;
        device += ms;
        if (prof == &pp) projection_ms += ms;
      }
    if (r) {
      gap.push_back(wall - device);
      projection_host.push_back(projection_ms + wall - device);
      const double dp_ms = dp.events.back().elapsed_us(dp.events.front()) / 1000.0;
      const double vp_ms = vp.events.back().elapsed_us(vp.events.front()) / 1000.0;
      const double pp_ms = pp.events.back().elapsed_us(pp.events.front()) / 1000.0;
      outside.push_back(wall - dp_ms - vp_ms - pp_ms);
      projection_outside.push_back(pp_ms + outside.back());
    }
  }
  std::printf("DFLASH_GAP depth=%u host_upper_ms=%.6f projection_plus_host_upper_ms=%.6f "
              "(instrumented signals include host-scope flush overhead)\n", depth, median(gap), median(projection_host));
  std::printf("DFLASH_HOST depth=%u outside_lists_ms=%.6f projection_plus_host_ms=%.6f "
              "(wall minus device-list intervals; instrumented, intervals include dispatch and signal flushes)\n",
              depth, median(outside), median(projection_outside));
  e.recapture_verify({});   // the engine's lists must release vp's events before it dies
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3) throw std::runtime_error("usage: probe_dflash_iter TARGET DRAFTER [--format int8|bf16|int4] [--candidate-head] [--depth N] [--calls N] [--rounds N] | --engine-taps SOURCES OUT [--anchors N]");
    uint32_t depth = 4096;
    unsigned calls = 16, rounds = 3, anchors = 16;
    std::string sources_path, out;
    bool candidate_head = false;
    auto format = loader::DraftFormat::Int8;
    for (int i = 3; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--candidate-head") { candidate_head = true; continue; }
      if (i + 1 >= argc) throw std::runtime_error(arg + ": missing value");
      if (arg == "--depth") depth = std::stoul(argv[++i]);
      else if (arg == "--calls") calls = std::stoul(argv[++i]);
      else if (arg == "--rounds") rounds = std::stoul(argv[++i]);
      else if (arg == "--format") {
        const std::string f = argv[++i];
        if (f == "int4") format = loader::DraftFormat::Int4G64;
        else if (!loader::parse_draft_format(f, format)) throw std::runtime_error("--format int8|bf16|int4");
      }
      else if (arg == "--anchors") anchors = std::stoul(argv[++i]);
      else if (arg == "--engine-taps" && i + 2 < argc) { sources_path = argv[++i]; out = argv[++i]; }
      else throw std::runtime_error("unknown option: " + arg);
    }
    if (!calls || !rounds || !anchors) throw std::runtime_error("calls, rounds and anchors must be positive");
    Json sources;
    uint32_t max_len = std::max(16384u, depth + 1024);
    if (!sources_path.empty()) {
      std::ifstream in(sources_path);
      in >> sources;
      for (const auto& s : sources)
        max_len = std::max(max_len, uint32_t(s.at("prompt_ids").size() + s.at("cont_ids").size() + 8));
      max_len = (max_len + 7) & ~7u;   // the prefill's row pitch needs a multiple of 8
    }
    l0::Context ctx(0);
    auto model = loader::load(ctx, argv[1], max_len, false, loader::LmHeadForm::Int8);
    auto weights = loader::load_drafter(ctx, argv[2], model, format, 0, candidate_head);
    const auto rows = model.linears.at({loader::kTopLevel, model::LinearId::LmHead}).shape.N;
    df::DFlashDrafter drafter(ctx, std::move(weights), rows, max_len, 7);
    const runtime::KvCache kv = runtime::default_kv_cache();
    runtime::Engine e(ctx, std::move(model), max_len, false, kv, &drafter);
    std::printf("DFLASH_KV cache=%s\n", kv == runtime::KvCache::Int8 ? "int8" : "bf16");
    drafter.attach(e, false);
    if (sources_path.empty()) costs(e, drafter, depth, calls, rounds);
    else engine_taps(e, drafter, sources, out, anchors);
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "probe_dflash_iter: %s\n", ex.what());
    return 2;
  }
}
