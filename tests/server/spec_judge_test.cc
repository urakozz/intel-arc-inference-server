// spec_judge_test - tests/server/spec_judge.h's grading without and with an f32 judge (spec 28 N4 / N6; the
// operator's 2026-11-04 rule: a divergence whose context has no f32 reference AWAITS it - the test SKIPs (77), it
// does not FAIL; a judged divergence and every other check still fail). Host only: the judges are written here.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <utility>
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
// A bf16 top-k reference (qwen_bf16_top.py's format): ids int32 [rows][C], logits f32 [rows][C].
void write_bf16top(const std::string& path, const std::vector<std::vector<std::pair<int32_t, float>>>& rows) {
  const size_t C = rows[0].size(), n = rows.size() * C * 4;
  std::string h = "{\"ids\":{\"dtype\":\"I32\",\"shape\":[" + std::to_string(rows.size()) + "," + std::to_string(C) +
                  "],\"data_offsets\":[0," + std::to_string(n) + "]},\"logits\":{\"dtype\":\"F32\",\"shape\":[" +
                  std::to_string(rows.size()) + "," + std::to_string(C) + "],\"data_offsets\":[" + std::to_string(n) +
                  "," + std::to_string(2 * n) + "]}}";
  while (h.size() % 8) h += ' ';
  std::ofstream f(path, std::ios::binary);
  const uint64_t hl = h.size();
  f.write(reinterpret_cast<const char*>(&hl), 8);
  f.write(h.data(), std::streamsize(h.size()));
  for (const auto& r : rows)
    for (const auto& [id, v] : r) f.write(reinterpret_cast<const char*>(&id), 4);
  for (const auto& r : rows)
    for (const auto& [id, v] : r) f.write(reinterpret_cast<const char*>(&v), 4);
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

  // 3. the bf16-unstable rule (operator 2026-11-04). The speculative pick 9 sits 12 ulp below the f32 top 4 at the
  //    speculative judge's row 0 (ulp of 10.0 = 1/16): beyond the near-miss bar, more misses than plain.
  {
    const double u = 1.0 / 16;
    std::vector<std::vector<float>> jr = rows_for({4, 5, 6, 0});
    jr[0][9] = float(10.0 - 12 * u);
    const std::string kb = dir + "/" + ks + ".bf16top.safetensors";
    write_judge(dir + "/" + ks + ".golden.safetensors", jr);
    setenv("B70_FORCED_BF16", dir.c_str(), 1);
    const golden::ForcedJudge fj;
    std::vector<std::vector<uint32_t>> asked;
    auto plain_gap = [&](double g) {   // a plain row with the pair (4, 9) g ulp apart
      return spec_judge::PlainRow([&asked, g, u](const std::vector<uint32_t>& gen) {
        asked.push_back(gen);
        std::vector<float> r(V, 0.f);
        r[4] = 10.f;
        r[9] = float(10.0 - g * u);
        return r;
      });
    };
    auto bf16 = [&](double g) {   // the bf16 reference: the pair g ulp apart at row 0 (other rows: their f32 tops)
      std::vector<std::vector<std::pair<int32_t, float>>> rows;
      rows.push_back({{4, 10.f}, {9, float(10.0 - g * u)}});
      for (int32_t t : {5, 6, 0}) rows.push_back({{t, 10.f}, {t, 10.f}});
      write_bf16top(kb, rows);
    };
    auto run = [&](const char* what, const spec_judge::PlainRow& pr) {
      const spec_judge::StreamVerdict sv = spec_judge::streams(fj, "t_judged", prompt, plain, steps, spec, V, V, pr);
      std::printf("%s: %s\n", what, sv.line().c_str());
      CHECK(sv.judged);
      CHECK_EQ(sv.beyond.size(), size_t(1));
      return sv;
    };
    // (a) no bf16 reference: the strict bar
    spec_judge::StreamVerdict sv = run("no bf16 reference", plain_gap(-3));
    CHECK(!sv.pass && !sv.beyond[0].have_bf16);
    CHECK(sv.line().find("no bf16 reference") != std::string::npos);
    CHECK(asked.empty());   // no plain replay for a row the rule does not reach
    // (b) bf16 within the bar of f32 (gap 10 vs 12): stable - the strict bar
    bf16(10);
    sv = run("bf16 stable", plain_gap(-3));
    CHECK(!sv.pass && sv.beyond[0].have_bf16 && !sv.beyond[0].unstable);
    CHECK(asked.empty());
    // (c) bf16-unstable (gap -5 vs 12), plain only 11 from f32 (gap 1): allowance 11 < 12 - FAIL
    bf16(-5);
    sv = run("unstable, plain close to f32", plain_gap(1));
    CHECK(!sv.pass && sv.beyond[0].unstable && sv.beyond[0].have_plain);
    CHECK(std::fabs(sv.beyond[0].allow - 11.0) < 1e-3);
    CHECK_EQ(asked.size(), size_t(1));
    CHECK(asked[0] == std::vector<uint32_t>({3}));   // prompt + plain[0, d) + spec[d, d + 0): the divergence row
    // (d) bf16-unstable, plain 15 from f32 (gap -3): allowance 15 >= 12 - PASS, printed loudly
    sv = run("unstable, plain far from f32", plain_gap(-3));
    CHECK(sv.pass && sv.beyond[0].pass);
    CHECK(std::fabs(sv.beyond[0].g32 - 12.0) < 1e-3 && std::fabs(sv.beyond[0].g16 + 5.0) < 1e-3 &&
          std::fabs(sv.beyond[0].gp + 3.0) < 1e-3 && std::fabs(sv.beyond[0].allow - 15.0) < 1e-3);
    CHECK(sv.line().find("bf16-unstable rule") != std::string::npos);
    // (e) bf16-unstable but no plain row: the strict bar
    sv = run("unstable, no plain row", spec_judge::PlainRow{});
    CHECK(!sv.pass && sv.beyond[0].unstable && !sv.beyond[0].have_plain);
    // (g) the reviewed rows (operator 2026-11-05): case (c)'s miss (row 0, f32 top 4, speculative 9) fails the rules.
    //     An entry with another id, or at another row, does not pass it; the exact entry does, printed as REVIEWED.
    static const std::string kks = ks;   // the entries keep the key's char*
    const size_t base = spec_judge::reviewed().rows.size();
    spec_judge::add_reviewed({kks.c_str(), 0, 4, 8, 12.0, "test", "another id at the same row"});
    spec_judge::add_reviewed({kks.c_str(), 1, 4, 9, 12.0, "test", "another row"});
    sv = run("reviewed: other id / other row only", plain_gap(1));
    CHECK(!sv.pass && sv.beyond[0].reviewed < 0);
    CHECK(sv.line().find("REVIEWED (operator") == std::string::npos);
    spec_judge::add_reviewed({kks.c_str(), 0, 4, 9, 12.0, "test", "the exact entry"});
    sv = run("reviewed: exact entry", plain_gap(1));
    CHECK(sv.pass && sv.beyond[0].pass && sv.beyond[0].reviewed == int(base + 2));
    CHECK(sv.line().find("PASS as REVIEWED") != std::string::npos);
    CHECK(!sv.beyond[0].unstable || sv.beyond[0].g32 > sv.beyond[0].allow);   // the rules alone still fail it
    CHECK_EQ(spec_judge::reviewed().hits[base + 2], uint64_t(1));
    // the two entries no miss matched are stale (their tag ran); the exact one is not; kReviewedRows' entries are not
    // (their tests did not run here). Stale entries are printed, never failed.
    const std::vector<size_t> stale = spec_judge::stale_reviewed();
    CHECK(stale == std::vector<size_t>({base, base + 1}));
    spec_judge::report_reviewed("spec_judge_test (reviewed rows)");
    // a different context with the same row and ids (another tag): no entry matches it
    CHECK_EQ(spec_judge::match_reviewed("t_other_" + kks.substr(kks.size() - 16), 0, 4, 9), -1);
    // (h) a variant of the test (operator 2026-11-07: mtp_gpu_32k_test / _128k_test): the same tag, row and ids under
    //     another hash (its stream differs after the row) match by the f32 gap, within kReviewedGapUlp only
    const std::string variant = spec_judge::key_tag(kks) + "_fedcba9876543210";
    bool by_gap = false;
    CHECK_EQ(spec_judge::match_reviewed(variant, 0, 4, 9), -1);                       // no gap given: exact only
    CHECK_EQ(spec_judge::match_reviewed(variant, 0, 4, 9, 12.004, &by_gap), int(base + 2));
    CHECK(by_gap);
    CHECK_EQ(spec_judge::match_reviewed(kks, 0, 4, 9, 12.004, &by_gap), int(base + 2));   // the exact key: not by gap
    CHECK(!by_gap);
    CHECK_EQ(spec_judge::match_reviewed(variant, 0, 4, 9, 12.02, &by_gap), -1);        // another gap: another row
    CHECK_EQ(spec_judge::match_reviewed(variant, 1, 4, 9, 12.0, &by_gap), int(base + 1));   // row 1: its own entry
    CHECK_EQ(spec_judge::match_reviewed(variant, 2, 4, 9, 12.0, &by_gap), -1);         // a row with no entry
    CHECK_EQ(spec_judge::match_reviewed(variant, 0, 4, 7, 12.0, &by_gap), -1);         // a pick with no entry
    CHECK_EQ(spec_judge::match_reviewed("t_other_fedcba9876543210", 0, 4, 9, 12.0, &by_gap), -1);   // another tag
    // (f) the allowance never drops below the bar: a near-miss (6 ulp) needs no rule at all
    jr[0][9] = float(10.0 - 6 * u);
    write_judge(dir + "/" + ks + ".golden.safetensors", jr);
    sv = spec_judge::streams(fj, "t_judged", prompt, plain, steps, spec, V, V, plain_gap(6));
    CHECK(sv.pass && sv.beyond.empty());
    unsetenv("B70_FORCED_BF16");
  }
  std::puts("spec_judge_test OK");
  return 0;
}
