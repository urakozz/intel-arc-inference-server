// spec_judge_test - tests/server/spec_judge.h's grading without and with an f32 judge (spec 28 N4 / N6; the
// the 2026-11-04 rule: a divergence whose context has no f32 reference AWAITS it - the test SKIPs (77), it
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

  // 3. the bf16-unstable rule. The speculative pick 9 sits 12 ulp below the f32 top 4 at the
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
    // one-sided (item 30 (4)): only a move TOWARD the speculative pick counts.
    // (e2) plain more confident in the f32 top than f32 (gap 25): |plain - f32| = 13 would have allowed the flip;
    //      f32 - plain = -13 allows the bar alone - FAIL
    sv = run("unstable, plain more confident than f32", plain_gap(25));
    CHECK(!sv.pass && sv.beyond[0].unstable && sv.beyond[0].have_plain);
    CHECK(std::fabs(sv.beyond[0].allow - 8.0) < 1e-3);
    // (e3) the bf16 reference more confident in the f32 top (gap 25): |bf16 - f32| = 13 would have called the row
    //      unstable; it moved away from the speculative pick - stable, the strict bar, no plain replay
    const size_t asked_before = asked.size();
    bf16(25);
    sv = run("bf16 away from the speculative pick", plain_gap(-3));
    CHECK(!sv.pass && sv.beyond[0].have_bf16 && !sv.beyond[0].unstable);
    CHECK_EQ(asked.size(), asked_before);
    bf16(-5);
    // (g) the reviewed rows: case (c)'s miss (row 0, f32 top 4, speculative 9) fails the rules.
    //     An entry with another id, or at another row, does not pass it; the exact entry does, printed as REVIEWED.
    static const std::string kks = ks;   // the entries keep the key's char*
    const size_t base = spec_judge::reviewed().rows.size();
    spec_judge::add_reviewed({kks.c_str(), 0, 4, 8, 12.0, "test", "another id at the same row"});
    spec_judge::add_reviewed({kks.c_str(), 1, 4, 9, 12.0, "test", "another row"});
    sv = run("reviewed: other id / other row only", plain_gap(1));
    CHECK(!sv.pass && sv.beyond[0].reviewed < 0);
    CHECK(sv.line().find("REVIEWED (2") == std::string::npos);
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
    // (h) a variant of the test (mtp_gpu_32k_test / _128k_test): the same tag, row and ids under
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

  // 4. the end of turn (item 30 (3)): each stream graded up to and including its first end-of-turn
  //    id, read from the snapshot's generation_config.json; the rows after it are printed, not graded.
  {
    std::ofstream(dir + "/config.json") << "{}\n";
    std::ofstream(dir + "/generation_config.json") << "{\"eos_token_id\": [7, 15]}\n";
    golden::load_end_of_turn(dir);
    CHECK(golden::end_of_turn_ids() == std::vector<uint32_t>({7, 15}));
    const std::vector<uint32_t> s = {3, 7, 5, 15};
    CHECK_EQ(golden::turn_end(s.data(), s.size()), size_t(2));   // the stop row 1 is graded, rows 2 and 3 are not
    CHECK_EQ(golden::turn_end(s.data(), size_t(1)), size_t(1));   // no stop: every row
    const golden::ForcedJudge fj;
    const std::vector<greedy_tie::RefStep> far5 = {step(3, 0, 50), step(4, 9, 40), step(7, 0, 50), step(5, 0, 50),
                                                   step(6, 0, 50)};
    // judges for plain {3, 4, 7, 5, 6} and a speculative stream from its first difference at 1; rows: the f32 tops
    auto judged = [&](const char* tag, const std::vector<uint32_t>& sp, const std::vector<uint32_t>& plain_tops,
                      const std::vector<uint32_t>& spec_tops) {
      const std::vector<uint32_t> pl = {3, 4, 7, 5, 6};
      const std::vector<uint32_t> c = {1, 2, 3}, p(pl.begin() + 1, pl.end()), q(sp.begin() + 1, sp.end());
      write_judge(dir + "/" + golden::ForcedJudge::key(tag, c, p) + ".golden.safetensors", rows_for(plain_tops));
      write_judge(dir + "/" + golden::ForcedJudge::key(tag, c, q) + ".golden.safetensors", rows_for(spec_tops));
      const spec_judge::StreamVerdict sv = spec_judge::streams(fj, tag, prompt, pl, far5, sp, V, V);
      std::printf("%s: %s\n", tag, sv.line().c_str());
      CHECK(sv.judged);
      return sv;
    };
    // (a) both end their turn at stream row 2 (suffix row 1); the speculative rows after it are rejected 10 below the
    //     f32 top - past the end of turn, not graded: PASS, each printed
    spec_judge::StreamVerdict sv = judged("t_turn_a", {3, 9, 7, 8, 8}, {4, 7, 5, 6, 0}, {9, 7, 1, 1, 0});
    CHECK(sv.pass && sv.miss_spec == 0 && sv.miss_plain == 0);
    CHECK(sv.graded_spec == 2 && sv.graded_plain == 2 && sv.rows == 4);
    CHECK_EQ(sv.past.size(), size_t(2));
    CHECK(sv.line().find("past end of turn (row 2): not graded") != std::string::npos);
    // (b) the stop row itself is graded: the judge rejects the speculative end-of-turn id - a miss, FAIL
    sv = judged("t_turn_b", {3, 9, 7, 8, 8}, {4, 7, 5, 6, 0}, {9, 5, 8, 8, 0});
    CHECK(!sv.pass && sv.miss_spec == 1 && sv.beyond.size() == 1 && sv.beyond[0].row == 1);
    // (c) the two streams end at different offsets: plain at stream row 2, the speculative stream at 4. Plain's
    //     rejected rows after its end do not count; the speculative stream's rows count up to its own end
    sv = judged("t_turn_c", {3, 9, 8, 8, 7}, {4, 7, 1, 1, 0}, {9, 8, 8, 7, 0});
    CHECK(sv.pass && sv.miss_plain == 0 && sv.graded_plain == 2 && sv.graded_spec == 4);
    CHECK_EQ(sv.past.size(), size_t(2));   // plain's rows 2 and 3
    sv = judged("t_turn_c2", {3, 9, 8, 8, 7}, {4, 7, 1, 1, 0}, {9, 8, 1, 7, 0});
    CHECK(!sv.pass && sv.miss_spec == 1 && sv.beyond[0].row == 2);   // its row 2 lies before its own end: graded
    // (d) a first difference after the shared end of turn: nothing graded, nothing dumped or awaited
    const uint64_t before = spec_judge::awaiting().divergences;
    unsetenv("B70_FORCED_JUDGE");
    const golden::ForcedJudge none;
    const std::vector<uint32_t> pl = {3, 7, 4, 5}, sp = {3, 7, 9, 9};
    const std::vector<greedy_tie::RefStep> far2 = {step(3, 0, 50), step(7, 0, 50), step(4, 9, 40), step(5, 0, 50)};
    sv = spec_judge::streams(none, "t_turn_d", prompt, pl, far2, sp, V, V);
    std::printf("t_turn_d: %s\n", sv.line().c_str());
    CHECK(sv.pass && !sv.judged && !sv.awaiting && sv.tie.at == 2);
    CHECK(sv.line().find("past end of turn (row 2): not graded") != std::string::npos);
    CHECK_EQ(spec_judge::awaiting().divergences, before);
    golden::end_of_turn_ids().clear();
  }

  // 5. the gates' accept lists (golden_common.h AcceptList; item 30 (4)): `prompt:step` accepts
  //    any id at that row, `prompt:step=id` that id only.
  {
    char a0[] = "gate", a1[] = "--accept=cjk:15=99737,prose:18", a2[] = "x";
    char* av[] = {a0, a1, a2};
    int ac = 3;
    const golden::AcceptList acc = golden::take_accept(ac, av);
    CHECK_EQ(ac, 2);
    CHECK(acc.has("cjk", 15, 99737) && !acc.has("cjk", 15, 99736) && !acc.has("cjk", 16, 99737));
    CHECK(acc.has("prose", 18, 0) && acc.has("prose", 18, 99737) && !acc.has("code", 18, 0));
  }
  std::puts("spec_judge_test OK");
  return 0;
}
