#pragma once
// The model name b70-serve answers to (/v1/models, the responses' `model` field) - derived from the model the
// operator named, as vLLM serves its --model argument (the zero-config cleanup, operator 2026-11-06, removed
// --served-name):
//
//   an HF repo id ("org/name")                          as given
//   a snapshot in an HF cache (.../models--org--name/snapshots/<rev>)   "org/name"
//   any other directory                                 its last component
//
// Requests may name any model: the server does not refuse a mismatch (one model per process).
#include <string>
#include <vector>

namespace cli {

inline std::string served_name(const std::string& path) {
  std::string p = path;
  while (p.size() > 1 && p.back() == '/') p.pop_back();
  std::vector<std::string> parts;
  for (size_t b = 0; b <= p.size();) {
    const size_t e = p.find('/', b);
    const size_t end = e == std::string::npos ? p.size() : e;
    if (end > b) parts.push_back(p.substr(b, end - b));
    if (e == std::string::npos) break;
    b = e + 1;
  }
  if (parts.empty()) return "b70";
  // An HF cache snapshot: a "models--org--name" component.
  for (const std::string& c : parts)
    if (c.rfind("models--", 0) == 0) {
      const std::string rest = c.substr(8);
      const size_t sep = rest.find("--");
      if (sep != std::string::npos && sep > 0 && sep + 2 < rest.size())
        return rest.substr(0, sep) + "/" + rest.substr(sep + 2);
    }
  // A repo id: two components, relative, not a path that names a directory explicitly.
  const bool relative = p[0] != '/' && p[0] != '.' && p[0] != '~';
  if (relative && parts.size() == 2) return parts[0] + "/" + parts[1];
  return parts.back();
}

}  // namespace cli
