#include <cmath>
#include <limits>

#include "check.h"
#include "runtime/qwen4exp_f32_judge.h"

int main() {
  const auto accepts = [](uint64_t pf_bad, uint64_t dec_bad, uint32_t pf_tokens, uint32_t dec_tokens) {
    return q4judge::RelativeCounts{pf_bad, dec_bad, pf_tokens, dec_tokens}.passes();
  };
  CHECK(accepts(1279, 1089, 32, 32));   // measured L18
  CHECK(accepts(8885, 8413, 29, 28));   // measured L37
  CHECK(accepts(6, 5, 0, 0));          // exactly 1.2x
  CHECK(!accepts(7, 5, 0, 0));
  CHECK(!accepts(5, 4, 0, 0));         // no rounding up an integer budget
  CHECK(accepts(0, 0, 32, 32));
  CHECK(!accepts(1, 0, 32, 32));
  CHECK(!accepts(1279, 1089, 31, 32)); // token loss fails even with acceptable routing
  CHECK(!accepts(1307, 1089, 32, 32)); // routing regression fails even with equal tokens
  const uint64_t max_count = std::numeric_limits<uint64_t>::max();
  CHECK(accepts(max_count, max_count, 32, 32));
  CHECK(!accepts(max_count, 0, 32, 32));
  const float reference[] = {0, -1, 2, NAN, INFINITY};
  CHECK(q4judge::finite_reference(reference, 3));  // padded logits outside vocab_used are not judged
  CHECK(!q4judge::finite_reference(reference, 4));
  CHECK(!q4judge::finite_reference(reference + 4, 1));
  using q4judge::within_twice;
  CHECK(within_twice(0.0, 0.0));
  CHECK(within_twice(0.2, 0.1));
  CHECK(within_twice(0.19, 0.1));
  CHECK(!within_twice(0.201, 0.1));
  CHECK(!within_twice(0.001, 0.0));
  CHECK(!within_twice(std::numeric_limits<double>::quiet_NaN(), 0.1));
  CHECK(!within_twice(0.1, std::numeric_limits<double>::infinity()));
  CHECK(!within_twice(-0.1, 0.1));
  q4judge::Cosine both_zero, one_zero, same, opposite, invalid;
  one_zero.add(1, 0);
  same.add(2, 4);
  opposite.add(2, -4);
  invalid.add(NAN, 1);
  CHECK_EQ(both_zero.distance(), 0.0);
  CHECK_EQ(one_zero.distance(), 1.0);
  CHECK_EQ(same.distance(), 0.0);
  CHECK_EQ(opposite.distance(), 2.0);
  CHECK(!within_twice(invalid.distance(), 0.1));
  const uint32_t a[] = {4, 2, 9}, b[] = {2, 9, 4}, c[] = {4, 3, 9};
  CHECK_EQ(q4judge::set_distance(a, b, 3), 0.0);
  CHECK(std::fabs(q4judge::set_distance(a, c, 3) - 1.0 / 3) < 1e-12);
  const uint32_t duplicate[] = {4, 4, 2}, outside[] = {2, 3, 10}, huge[] = {0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF};
  CHECK(q4judge::valid_set(a, 3, 10));
  CHECK(!q4judge::valid_set(duplicate, 3, 10));
  CHECK(!q4judge::valid_set(outside, 3, 10));
  CHECK(!q4judge::valid_set(huge, 3, 512));
}
