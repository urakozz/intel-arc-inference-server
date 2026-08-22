#pragma once
#include <level_zero/ze_api.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace l0 {
const char* result_name(ze_result_t r);
struct Error : std::runtime_error {
  ze_result_t result;
  Error(const char* call, ze_result_t r)
      : std::runtime_error(std::string(call) + " -> " + result_name(r)), result(r) {}
};
}  // namespace l0

// Every Level Zero call goes through this. DEVICE_LOST means the driver is
// wedged (doc 06 has the history); nothing sensible can follow, so abort.
#define ZE_CHECK(call)                                                        \
  do {                                                                        \
    ze_result_t _r = (call);                                                  \
    if (_r != ZE_RESULT_SUCCESS) {                                            \
      if (_r == ZE_RESULT_ERROR_DEVICE_LOST) {                                \
        std::fprintf(stderr, "FATAL: ZE_RESULT_ERROR_DEVICE_LOST in %s\n", #call); \
        std::abort();                                                         \
      }                                                                       \
      throw ::l0::Error(#call, _r);                                           \
    }                                                                         \
  } while (0)
