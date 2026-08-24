#include "loader/snapshot.h"
#include <sys/stat.h>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

namespace loader {
namespace {
bool is_dir(const std::string& p) {
  struct stat st{};
  return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}
bool is_file(const std::string& p) {
  struct stat st{};
  return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}
}  // namespace

std::string resolve_snapshot(const std::string& arg) {
  if (is_dir(arg)) {
    std::string dir = arg;
    if (dir.back() != '/') dir += '/';
    if (!is_file(dir + "config.json"))
      throw std::runtime_error("snapshot directory has no config.json: " + dir);
    return dir;
  }
  auto slash = arg.find('/');
  if (slash == std::string::npos || arg.find('/', slash + 1) != std::string::npos)
    throw std::runtime_error("not a directory and not an <org>/<name> repo id: " + arg);
  const char* hf_home = std::getenv("HF_HOME");
  std::string base = hf_home ? std::string(hf_home)
                             : std::string(std::getenv("HOME") ? std::getenv("HOME") : "") +
                                   "/.cache/huggingface";
  std::string model_dir =
      base + "/hub/models--" + arg.substr(0, slash) + "--" + arg.substr(slash + 1);
  if (!is_dir(model_dir))
    throw std::runtime_error("model not in local HF cache (run `hf download " + arg +
                             "`): " + model_dir);
  std::ifstream ref(model_dir + "/refs/main");
  std::string rev;
  if (!ref || !std::getline(ref, rev) || rev.empty())
    throw std::runtime_error("cannot read revision: " + model_dir + "/refs/main");
  std::string snap = model_dir + "/snapshots/" + rev + "/";
  if (!is_file(snap + "config.json"))
    throw std::runtime_error("snapshot incomplete (no config.json): " + snap);
  return snap;
}
}  // namespace loader
