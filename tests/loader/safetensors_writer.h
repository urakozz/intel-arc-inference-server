#pragma once
// The loader tests' synthetic checkpoints: one safetensors file (header and every tensor 8-byte aligned, an optional
// __metadata__ of string -> string) and the one-shard checkpoint directory the loaders read - that file plus a
// model.safetensors.index.json naming it for every tensor.
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace safetensors_writer {

namespace fs = std::filesystem;

struct Tensor {
  std::string dtype;
  std::vector<uint64_t> shape;
  std::vector<uint8_t> bytes;
};
using Tensors = std::map<std::string, Tensor>;

inline void write_safetensors(const fs::path& file, const Tensors& ts,
                              const std::map<std::string, std::string>& meta = {}) {
  std::string hdr = "{";
  bool first = true;
  if (!meta.empty()) {
    hdr += "\"__metadata__\":{";
    bool f2 = true;
    for (const auto& [k, v] : meta) {
      hdr += (f2 ? "" : ",") + ("\"" + k + "\":\"" + v + "\"");
      f2 = false;
    }
    hdr += "}";
    first = false;
  }
  uint64_t off = 0;
  for (const auto& [name, t] : ts) {
    hdr += (first ? "" : ",");
    first = false;
    hdr += "\"" + name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[";
    for (size_t i = 0; i < t.shape.size(); ++i) hdr += (i ? "," : "") + std::to_string(t.shape[i]);
    hdr += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + t.bytes.size()) + "]}";
    off += t.bytes.size();
    off = (off + 7) / 8 * 8;
  }
  hdr += "}";
  while ((8 + hdr.size()) % 8 != 0) hdr += ' ';
  std::ofstream f(file, std::ios::binary);
  const uint64_t hlen = hdr.size();
  f.write(reinterpret_cast<const char*>(&hlen), 8);
  f.write(hdr.data(), std::streamsize(hdr.size()));
  uint64_t pos = 0;
  for (const auto& [name, t] : ts) {
    (void)name;
    f.write(reinterpret_cast<const char*>(t.bytes.data()), std::streamsize(t.bytes.size()));
    pos += t.bytes.size();
    const uint64_t pad = (pos + 7) / 8 * 8 - pos;
    const char zeros[8] = {};
    f.write(zeros, std::streamsize(pad));
    pos += pad;
  }
}

// dir/<shard> holding `ts`, and dir/model.safetensors.index.json mapping every tensor to it.
inline void write_checkpoint(const fs::path& dir, const Tensors& ts,
                             const std::string& shard = "model-00001-of-00001.safetensors",
                             const std::map<std::string, std::string>& meta = {}) {
  fs::create_directories(dir);
  write_safetensors(dir / shard, ts, meta);
  std::ofstream ix(dir / "model.safetensors.index.json");
  ix << "{\"metadata\":{},\"weight_map\":{";
  bool first = true;
  for (const auto& [name, t] : ts) {
    (void)t;
    ix << (first ? "" : ",") << "\"" << name << "\":\"" << shard << "\"";
    first = false;
  }
  ix << "}}";
}

}  // namespace safetensors_writer
