// Fix round A (2026-11-01): every test binary's stdout is line-buffered. On the box
// stdout is a pipe or a file, so it was fully buffered, and a test that died (a
// DEVICE_LOST abort, a signal) took its last phase lines with it - the crash logs
// ended before the phase that crashed. tests/CMakeLists.txt adds this file to every
// executable it declares, so no test's main has to remember, and every launcher
// (ctest, the box harness, a hand run under tee) gets the lines as they are printed.
// stderr is unbuffered already.
#include <cstdio>

namespace {
struct LineBufferedStdout {
  LineBufferedStdout() { std::setvbuf(stdout, nullptr, _IOLBF, 0); }
};
const LineBufferedStdout line_buffered_stdout;   // before main, before any output
}  // namespace
