// Host-only rules of Bolt's f32 judge: set counts are symmetric; near-ties are printed, never subtracted.
#include "bolt_judge.h"
#include "check.h"

int main() {
  const uint32_t e[] = {1, 2, 3};
  const int32_t same[] = {3, 1, 2}, diff[] = {1, 2, 4};
  bolt_judge::Counts c;
  c.add(e, diff, same, 3, 0.00390625f, 0.8f);
  CHECK_EQ(c.engine, uint64_t(0));
  CHECK_EQ(c.golden, uint64_t(1));
  CHECK_EQ(c.near, uint64_t(1));
  CHECK(c.ok());
  c.add(e, diff, diff, 3, 0.004f, 0.8f);
  CHECK_EQ(c.engine, uint64_t(1));
  CHECK_EQ(c.golden, uint64_t(1));
  CHECK_EQ(c.near, uint64_t(1));
  CHECK(c.ok());
  c.add(e, diff, diff, 3, 0.0f, 0.8f);
  CHECK(!c.ok());   // near-tie differences still count
  CHECK_EQ(c.pairs, uint64_t(3));
  CHECK(bolt_judge::near_cut(0.0f, -0.8f));
  CHECK(!bolt_judge::near_cut(INFINITY, 0.8f));
  CHECK(bolt_judge::state_ok(0.998, 0.997));
  CHECK(bolt_judge::state_ok(0.997, 0.997));
  CHECK(!bolt_judge::state_ok(0.996, 0.997));
  CHECK(!bolt_judge::state_ok(NAN, 0.997));
  bolt_judge::Cosines cos;
  cos.add(1, 1, 1);
  cos.add(0, 1, 0);
  CHECK(cos.engine() > cos.baseline() && cos.ok());
  bolt_judge::Cosines zeros;
  CHECK(zeros.ok());
  std::puts("bolt_judge_test OK");
}
