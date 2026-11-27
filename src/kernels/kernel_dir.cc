#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>

#include "kernels/kernels.h"

namespace kernels {

std::string resolve_kernel_dir(const std::string& compiled, const std::string& exe_path) {
  namespace fs = std::filesystem;
  std::error_code ec;
  if (fs::is_directory(compiled, ec)) return compiled;
  std::string beside;
  if (!exe_path.empty()) {
    const fs::path rel = fs::path(exe_path).parent_path() / ".." / "kernels";
    beside = rel.lexically_normal().string();
    const fs::path abs = fs::weakly_canonical(rel, ec);
    if (!ec && fs::is_directory(abs, ec)) return abs.string();
  }
  throw std::runtime_error("no compiled kernel directory found: looked at '" + compiled + "' and '" +
                           (beside.empty() ? std::string("<the executable's directory>/../kernels") : beside) + "'");
}

const std::string& kernel_dir() {
  static const std::string dir = [] {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path exe = fs::read_symlink("/proc/self/exe", ec);
    return resolve_kernel_dir(compiled_kernel_dir(), ec ? std::string() : exe.string());
  }();
  return dir;
}

std::string path(const std::string& variant) {
  return kernel_dir() + "/" + variant + ".bin";
}

}  // namespace kernels
