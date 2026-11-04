// spec_judge_test - tests/server/spec_judge.h's grading without and with an f32 judge (spec 28 N4 / N6; the
// operator's 2026-11-04 rule: a divergence whose context has no f32 reference AWAITS it - the test SKIPs (77), it
// does not FAIL; a judged divergence and every other check still fail). Host only: the judges are written here.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <unistd.h>

#include "check.h"
#include "spec_judge.h"

namespace {

constexpr uint32_t V = 16;

// A --lean judge file: logits [rows][V] f32 (safetensors: u64 header length, the JSON header, the data).
void write_judge(const std::string& path, const std::vector<std::vector<float>>& rows) {
  const size_t n = rows.size() * V * 4;
  std::string h = "{\"logits\":{\"dtype\":\"F32\",\"shape\":[" + std::to_string(rows.size()) + "," +
                  std::to_string(V) + "],\"data_offsets\":[0," + std::to_string(n) + "]}}";
  while (h.size() % 8) h += ' ';
  std::ofstream f(path, std::ios::binary);
  const uint64_t hl = h.size();
  f.write(reinterpret_cast<const char*>(&hl), 8);
  f.write(h.data(), std::streamsize(h.size()));
  for (const auto& r : rows) f.write(reinterpret_cast<const char*>(r.data()), V * 4);
}
// One judge row per forced id (and the last position's): the f32 top at `top`, far above the rest.
std::vector<std::vector<float>> rows_for(const std::vector<uint32_t>& tops) {
  std::vector<std::vector<float>> rows;
  for (uint32_t t : tops) {
    std::vector<float> r(V, 0.f);
    r[t] = 10.f;
    rows.push_back(r);
  }
  return rows;
}
greedy_tie::RefStep step(uint32_t top1, uint32_t second, double margin) {
  greedy_tie::RefStep s;
  s.top1 = top1;
  s.second = second;
  s.margin_ulps = margin;
  return s;
}

}  // namespace

int main() {
  const std::vector<uint32_t> prompt = {1, 2}, plain = {3, 4, 5, 6}, spec = {3, 9, 5, 6};
  // the plain rows: the speculative id 9 at position 1 is far beyond the 1-ulp tie rule
  const std::vector<greedy_tie::RefStep> steps = {step(3, 0, 50), step(4, 9, 40), step(5, 0, 50), step(6, 0, 50)};
  std::vector<uint32_t> ctx = prompt;
  ctx.push_back(plain[0]);
  const std::vector<uint32_t> fp(plain.begin() + 1, plain.end()), fs(spec.begin() + 1, spec.end());

  // 1. no judge dir: the divergence awaits its reference - not failed, counted, exit 77
  unsetenv("B70_FORCED_JUDGE");
  unsetenv("B70_DUMP_FORCED");
  {
    const golden::ForcedJudge fj;
    const spec_judge::StreamVerdict sv = spec_judge::streams(fj, "t_stream", prompt, plain, steps, spec, V, V);
    std::printf("no judge: %s\n", sv.line().c_str());
    CHECK(sv.tie.v == greedy_tie::Verdict::Diverged);
    CHECK(!sv.judged && sv.awaiting && sv.pass);
    CHECK(sv.line().find("AWAITING") != std::string::npos);
    // a near-tie needs no judge: the tie rule passes it, nothing awaits
    const std::vector<greedy_tie::RefStep> tie = {step(3, 0, 50), step(4, 9, 0.5), step(5, 0, 50), step(6, 0, 50)};
    const spec_judge::StreamVerdict tv = spec_judge::streams(fj, "t_tie", prompt, plain, tie, spec, V, V);
    CHECK(tv.tie.v == greedy_tie::Verdict::Tie && tv.pass && !tv.awaiting);
    // identical streams: nothing to judge
    CHECK(spec_judge::streams(fj, "t_same", prompt, plain, steps, plain, V, V).pass);
    // N4 rows: below the 0.99999 bar without a judge awaits; above it passes
    const spec_judge::RowJudge rj = spec_judge::open_rows(fj, "t_rows", prompt, plain, V);
    CHECK(!rj);
    std::vector<float> a(V, 1.f), b(V, 1.f);
    const spec_judge::RowVerdict low = spec_judge::row(a.data(), b.data(), rj, 1, 0.5, V);
    CHECK(low.pass && low.awaiting && !low.judged);
    CHECK(spec_judge::row_line(low).find("AWAITING") != std::string::npos);
    const spec_judge::RowVerdict high = spec_judge::row(a.data(), b.data(), rj, 1, 0.999999, V);
    CHECK(high.pass && !high.awaiting);
  }
  CHECK_EQ(spec_judge::awaiting().divergences, uint64_t(2));
  CHECK_EQ(spec_judge::awaiting().contexts.size(), size_t(2));
  CHECK_EQ(spec_judge::exit_code(true, "spec_judge_test (awaiting)"), 77);   // nothing else failed: SKIP
  CHECK_EQ(spec_judge::exit_code(false, "spec_judge_test (awaiting + a failure)"), 1);   // a failure still fails

  // 2. with judges: graded for real, nothing more awaits
  char tmpl[] = "/tmp/spec_judge_test_XXXXXX";
  CHECK(mkdtemp(tmpl) != nullptr);
  const std::string dir = tmpl;
  setenv("B70_FORCED_JUDGE", dir.c_str(), 1);
  const std::string kp = golden::ForcedJudge::key("t_judged", ctx, fp), ks = golden::ForcedJudge::key("t_judged", ctx, fs);
  write_judge(dir + "/" + kp + ".golden.safetensors", rows_for({4, 5, 6, 0}));   // accepts every plain id
  {
    // the speculative stream's judge rejects its 9 (f32 top 4, 10 below it): more misses than plain - FAIL
    write_judge(dir + "/" + ks + ".golden.safetensors", rows_for({4, 5, 6, 0}));
    const golden::ForcedJudge fj;
    const spec_judge::StreamVerdict sv = spec_judge::streams(fj, "t_judged", prompt, plain, steps, spec, V, V);
    std::printf("judged, rejected: %s\n", sv.line().c_str());
    CHECK(sv.judged && !sv.awaiting && !sv.pass);
    CHECK_EQ(sv.miss_spec, 1u);
    CHECK_EQ(sv.miss_plain, 0u);
    // the same divergence where f32 prefers the speculative id: PASS
    write_judge(dir + "/" + ks + ".golden.safetensors", rows_for({9, 5, 6, 0}));
    const spec_judge::StreamVerdict ok = spec_judge::streams(fj, "t_judged", prompt, plain, steps, spec, V, V);
    std::printf("judged, accepted: %s\n", ok.line().c_str());
    CHECK(ok.judged && ok.pass && !ok.awaiting);
    // N4 with a judge row: graded by the distance to f32, not awaiting
    const std::vector<uint32_t> rf = {3, 4, 5};
    write_judge(dir + "/" + golden::ForcedJudge::key("t_rows", prompt, rf) + ".golden.safetensors",
                rows_for({3, 4, 5, 6}));
    const spec_judge::RowJudge rj = spec_judge::open_rows(fj, "t_rows", prompt, rf, V);
    CHECK(static_cast<bool>(rj));
    std::vector<float> far(V, 0.f), near(V, 0.f);
    far[0] = 10.f;    // nowhere near the judge's row 1 (top 4)
    near[4] = 10.f;   // the judge's row itself
    const spec_judge::RowVerdict rv = spec_judge::row(far.data(), near.data(), rj, 1, 0.5, V);
    CHECK(rv.judged && !rv.awaiting && !rv.pass);
  }
  CHECK_EQ(spec_judge::awaiting().divergences, uint64_t(2));   // unchanged by the judged ones
  std::puts("spec_judge_test OK");
  return 0;
}
