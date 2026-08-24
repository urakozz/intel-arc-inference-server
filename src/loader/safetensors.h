#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace loader {

struct TensorInfo {
  std::string dtype;             // "BF16" | "F16" | "F32" | "I32" (as spelled by safetensors)
  std::vector<uint64_t> shape;
  uint64_t begin = 0, end = 0;   // offsets into the file's data section
  uint32_t file = 0;             // index into file_names()
};

// Read-only mmap of one file.
class MappedFile {
 public:
  explicit MappedFile(const std::string& path);
  ~MappedFile();
  MappedFile(MappedFile&& o) noexcept;
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  const uint8_t* data() const { return data_; }
  size_t size() const { return size_; }

 private:
  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
};

// The checkpoint's sharded tensor set, manifest = the index (dedup by name).
class SafetensorsSet {
 public:
  explicit SafetensorsSet(const std::string& snapshot_dir);
  const std::map<std::string, TensorInfo>& tensors() const { return tensors_; }
  const uint8_t* data(const TensorInfo& t) const;
  size_t bytes(const TensorInfo& t) const { return t.end - t.begin; }
  const std::vector<std::string>& file_names() const { return names_; }

  // Parses one safetensors header (8-byte LE length + JSON). Exposed for tests.
  static std::vector<std::pair<std::string, TensorInfo>> parse_header(const uint8_t* p, size_t n);

 private:
  std::vector<MappedFile> files_;
  std::vector<size_t> data_start_;   // per file: 8 + header_len
  std::vector<std::string> names_;
  std::map<std::string, TensorInfo> tensors_;
};

}  // namespace loader
