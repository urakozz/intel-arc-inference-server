#pragma once
// The loader reports itself with std::printf - to stdout, which is b70-decode's machine-readable channel and
// b70-serve's request log. Rather than change the loaders (the report is wanted, just not there), point fd 1 at fd 2
// for the duration of a load so the banner joins the other diagnostics on stderr.
#include <unistd.h>

#include <cstdio>

namespace cli {

class StdoutToStderr {
 public:
  StdoutToStderr() {
    std::fflush(stdout);
    saved_ = ::dup(1);
    if (saved_ >= 0 && ::dup2(2, 1) < 0) {
      ::close(saved_);
      saved_ = -1;
    }
  }
  ~StdoutToStderr() {
    std::fflush(stdout);
    if (saved_ >= 0) {
      ::dup2(saved_, 1);
      ::close(saved_);
    }
  }
  StdoutToStderr(const StdoutToStderr&) = delete;
  StdoutToStderr& operator=(const StdoutToStderr&) = delete;

 private:
  int saved_ = -1;
};

}  // namespace cli
