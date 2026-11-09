#include <cmath>
#include <limits>

#include "check.h"
#include "runtime/qwen4exp_f32_judge.h"

int main() {
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
