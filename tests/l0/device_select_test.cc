// Device selection, doc 04 "Device selection": `--device N` wins, then
// `ONEAPI_DEVICE_SELECTOR`, then device 0. Level Zero itself never reads that
// variable - only the SYCL runtime does - so the raw-L0 path has to parse it,
// and a parser that guesses wrong binds the wrong card silently. This test is
// the whole contract. It needs no GPU: `device_index_from_env()` is a pure
// parse that touches no driver, so it fails loudly on any machine, not just
// on the box.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include "check.h"
#include "l0/context.h"

namespace {
constexpr const char* kVar = "ONEAPI_DEVICE_SELECTOR";

// Parse with the variable set to `v`, or unset when `v == nullptr`.
uint32_t parse(const char* v) {
  if (v == nullptr)
    ::unsetenv(kVar);
  else
    ::setenv(kVar, v, 1);
  return l0::Context::device_index_from_env();
}

// The exception message for `v`, or "" if the parse unexpectedly succeeded.
std::string rejection(const char* v) {
  ::setenv(kVar, v, 1);
  try {
    l0::Context::device_index_from_env();
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return "";
}

bool has(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}
}  // namespace

int main() {
  // Accepted: unset and `level_zero:*` both mean device 0 (there is no
  // multi-device execution until P/D disaggregation exists); `level_zero:N`
  // means GPU N, leading zeros and all.
  CHECK_EQ(parse(nullptr), 0u);
  CHECK_EQ(parse(""), 0u);  // exported-but-empty reads as unset
  CHECK_EQ(parse("level_zero:*"), 0u);
  CHECK_EQ(parse("level_zero:0"), 0u);
  CHECK_EQ(parse("level_zero:1"), 1u);
  CHECK_EQ(parse("level_zero:01"), 1u);
  CHECK_EQ(parse("level_zero:7"), 7u);
  // The boundary at the sentinel. `l0::Context::kFromEnv` is 0xFFFFFFFF and
  // means "ask the environment", so it is the one value this parser must never
  // *return* - returning it would send Context straight back here and recurse
  // in spirit, and `--device 4294967295` is rejected by the CLI for the same
  // reason. One below it is an ordinary (absurd) index and is accepted: the
  // rejection is the sentinel itself, not "a big number".
  CHECK_EQ(parse("level_zero:4294967294"), 4294967294u);
  CHECK_EQ(parse("level_zero:4294967294"), l0::Context::kFromEnv - 1u);

  // Rejected: another backend, a multi-backend list, a device list, a
  // malformed index, an index that cannot be a uint32. Binding card 0 for any
  // of these would be a silent lie about which GPU is running the model, so
  // every one throws, and the message names the offending value and both
  // accepted forms so the operator can fix it without reading this file.
  const char* bad[] = {
      "opencl:0",              // another backend
      "cuda:*",                //
      "level_zero:0;opencl:*", // a multi-backend list
      "level_zero:0,1",        // a device list
      "level_zero:x",          // garbage index
      "level_zero:",           // empty index
      "level_zero",            // no index at all
      "level_zero:-1",         // signed
      "level_zero:1.0",        //
      "*",                     // backend omitted
      "level_zero:99999999999",// does not fit a uint32
      "level_zero:4294967295", // fits a uint32, but IS kFromEnv (the sentinel)
      "level_zero:4294967296", // one past a uint32
      " level_zero:1",         // leading space is not a selector
  };
  for (const char* v : bad) {
    const std::string msg = rejection(v);
    CHECK(!msg.empty());
    CHECK(has(msg, v));
    CHECK(has(msg, "level_zero:N"));
    CHECK(has(msg, "level_zero:*"));
  }

  ::unsetenv(kVar);
  CHECK_EQ(parse(nullptr), 0u);
  std::puts("device_select_test OK");
  return 0;
}
