#pragma once
// The flags b70-decode and b70-serve renamed on 2026-10-06 (vLLM's convention: `--pp` /
// `--pipeline-parallel-size` is pipeline parallel), so an old command line fails naming the
// new spelling instead of a bare "unknown option". No aliases: the old names are refused.
//
//   --pipeline 1|2   -> --pp N / --pipeline-parallel-size N
//   --pp N (bench)   -> --prefill-length N   (`--pp` itself now refuses any value but 1 or 2)
//   --pp-chunk C     -> --prefill-chunk C
//   --pp-backend B   -> --prefill-backend B
#include <string>

namespace cli {

// "unknown option '<arg>'", plus the new spelling when `arg` is a renamed flag.
inline std::string unknown_option(const std::string& arg) {
  std::string m = "unknown option '" + arg + "'";
  if (arg == "--pp-chunk") m += " (renamed: --prefill-chunk C)";
  if (arg == "--pp-backend") m += " (renamed: --prefill-backend B)";
  if (arg == "--pipeline") m += " (renamed: --pp N, or --pipeline-parallel-size N)";
  return m;
}

}  // namespace cli
