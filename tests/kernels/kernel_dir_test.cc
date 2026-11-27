// The kernel directory lookup (src/kernels/kernel_dir.cc): the compiled-in path wins when it exists; an
// extracted release tarball's <executable's directory>/../kernels is the fallback; with neither, one plain
// sentence names both places. Host only - it drives the pure resolve_kernel_dir with its own directories.
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "check.h"
#include "kernels/kernels.h"

namespace {
namespace fs = std::filesystem;

void check_lookup() {
  const fs::path root = fs::temp_directory_path() / "b70_kernel_dir_test";
  fs::remove_all(root);
  fs::create_directories(root / "bin");
  fs::create_directories(root / "kernels");
  fs::create_directories(root / "build" / "kernels");
  fs::create_directories(root / "empty" / "bin");
  const std::string exe = (root / "bin" / "b70-serve").string();
  const std::string compiled = (root / "build" / "kernels").string();
  const std::string missing = (root / "elsewhere" / "kernels").string();
  // The compiled-in directory exists: it wins, even beside an executable with its own kernels/.
  CHECK_EQ(kernels::resolve_kernel_dir(compiled, exe), compiled);
  // The compiled-in directory is missing: the executable's ../kernels.
  CHECK_EQ(kernels::resolve_kernel_dir(missing, exe), fs::weakly_canonical(root / "kernels").string());
  // Neither: one sentence naming both places.
  const std::string lonely = (root / "empty" / "bin" / "b70-decode").string();
  bool threw = false;
  try {
    (void)kernels::resolve_kernel_dir(missing, lonely);
  } catch (const std::runtime_error& e) {
    threw = true;
    const std::string msg = e.what();
    CHECK(msg.find(missing) != std::string::npos);
    CHECK(msg.find((root / "empty" / "kernels").string()) != std::string::npos);
  }
  CHECK(threw);
  fs::remove_all(root);
  std::printf("kernel_dir: the compiled-in path wins, <executable>/../kernels falls back, neither names both\n");
}
}  // namespace

int main() {
  check_lookup();
  return 0;
}
