#pragma once
// A contract violation (pre / post / contract_assert, enforced: the top CMakeLists.txt) ends the process. `dies` runs
// `f` in a forked child and is true when the child ended by SIGABRT - the handler (src/common/contract_handler.cc)
// has printed the violated predicate on stderr. The child writes no core file and touches no device: call it with
// what fails before any device work (a precondition), never with a step in flight.
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <functional>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

inline bool dies(const std::function<void()>& f) {
  std::fflush(nullptr);
  const pid_t pid = ::fork();
  if (pid < 0) return false;
  if (pid == 0) {
    const rlimit none{0, 0};
    ::setrlimit(RLIMIT_CORE, &none);
    f();
    ::_exit(0);   // no violation
  }
  int status = 0;
  while (::waitpid(pid, &status, 0) < 0)
    if (errno != EINTR) return false;
  return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}
