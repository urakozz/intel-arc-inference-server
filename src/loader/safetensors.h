#pragma once
#include <cstdint>
#include <functional>
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

// Read-only mmap of one file. Its page cache is dropped when it is destroyed: a load reads more of the model than the
// RAM left beside the host-memory allocations holds, and the cached shards kept free memory at the low watermark, so the
// kernel's reclaim made the GPU driver swap host-memory allocations out under the first request (they came back at
// the swap file's ~40 MB/s, a 4,511-id tp prefill took 77 s).
class MappedFile {
 public:
  explicit MappedFile(const std::string& path);
  ~MappedFile();
  MappedFile(MappedFile&& o) noexcept;
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  const uint8_t* data() const { return data_; }
  size_t size() const { return size_; }
  int fd() const { return fd_; }

 private:
  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
  int fd_ = -1;
};

// The checkpoint's sharded tensor set, manifest = the index (dedup by name).
// Bytes [begin, end) of one file of a SafetensorsSet (file offsets), for read_ahead.
struct FileRange {
  uint32_t file = 0;
  uint64_t begin = 0, end = 0;
};

class SafetensorsSet {
 public:
  explicit SafetensorsSet(const std::string& snapshot_dir);
  // Spec 19b: `single_file_ok` also opens a directory with no index and one `model.safetensors` (every published
  // DFlash drafter ships that way): its header is the manifest. Without it - every target loader - a missing index
  // is the error it always was.
  SafetensorsSet(const std::string& snapshot_dir, bool single_file_ok);
  // A file every tensor of which the index maps to names `skip` holds for is never opened (it may be missing): its
  // tensors are left out of tensors(). Tensors `skip` holds for in other files stay in.
  SafetensorsSet(const std::string& snapshot_dir, const std::function<bool(const std::string&)>& skip);
  const std::map<std::string, TensorInfo>& tensors() const { return tensors_; }
  const uint8_t* data(const TensorInfo& t) const;
  size_t bytes(const TensorInfo& t) const { return t.end - t.begin; }
  const std::vector<std::string>& file_names() const { return names_; }
  // The bytes `ts` occupy, sorted by file and offset, touching ranges merged.
  std::vector<FileRange> ranges(const std::vector<const TensorInfo*>& ts) const;
  // Reads `r` into the page cache - `readers` threads, 8 MiB preads - and returns the bytes read. A loader that walks
  // the checkpoint tensor by tensor then faults on cached pages instead of waiting on the disk a page fault at a time
  // (MADV_SEQUENTIAL's readahead alone read a Flash-Next checkpoint at ~0.3 GB/s from a disk that reads 1.1 GB/s).
  size_t read_ahead(const std::vector<FileRange>& r, unsigned readers) const;

  // Parses one safetensors header (8-byte LE length + JSON). Exposed for tests.
  static std::vector<std::pair<std::string, TensorInfo>> parse_header(const uint8_t* p, size_t n);

 private:
  SafetensorsSet(const std::string& snapshot_dir, bool single_file_ok,
                 const std::function<bool(const std::string&)>& skip);
  std::vector<MappedFile> files_;
  std::vector<size_t> data_start_;   // per file: 8 + header_len
  std::vector<std::string> names_;
  std::map<std::string, TensorInfo> tensors_;
};

}  // namespace loader
