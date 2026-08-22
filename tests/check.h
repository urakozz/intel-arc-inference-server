#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CHECK(cond)                                                             \
  do {                                                                          \
    if (!(cond)) {                                                              \
      std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

#define CHECK_EQ(a, b)                                                          \
  do {                                                                          \
    auto _a = (a); auto _b = (b);                                               \
    if (!(_a == _b)) {                                                          \
      std::fprintf(stderr, "%s:%d: CHECK_EQ failed: %s != %s\n", __FILE__,      \
                   __LINE__, #a, #b);                                           \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                                   \
  do {                                                                          \
    double _a = (a); double _b = (b); double _t = (tol);                        \
    if (!(std::fabs(_a - _b) <= _t)) {                                          \
      std::fprintf(stderr, "%s:%d: CHECK_NEAR failed: %s=%g %s=%g tol=%g\n",    \
                   __FILE__, __LINE__, #a, _a, #b, _b, _t);                     \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)
