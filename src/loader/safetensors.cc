#include "loader/safetensors.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "common/json.h"

namespace loader {

MappedFile::MappedFile(const std::string& path) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open: " + path);
  struct stat st{};
  if (::fstat(fd, &st) != 0) { ::close(fd); throw std::runtime_error("fstat failed: " + path); }
  size_ = size_t(st.st_size);
  void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (p == MAP_FAILED) throw std::runtime_error("mmap failed: " + path);
  ::madvise(p, size_, MADV_SEQUENTIAL);
  data_ = static_cast<const uint8_t*>(p);
}
MappedFile::~MappedFile() {
  if (data_) ::munmap(const_cast<uint8_t*>(data_), size_);
}
MappedFile::MappedFile(MappedFile&& o) noexcept : data_(o.data_), size_(o.size_) {
  o.data_ = nullptr;
  o.size_ = 0;
}

std::vector<std::pair<std::string, TensorInfo>> SafetensorsSet::parse_header(const uint8_t* p,
                                                                             size_t n) {
  if (n < 8) throw std::runtime_error("safetensors: file too small");
  uint64_t hlen = 0;
  std::memcpy(&hlen, p, 8);
  // n >= 8 here, so subtract instead of adding: `8 + hlen` would wrap for
  // hlen near UINT64_MAX and let a crafted/truncated file through.
  if (hlen > n - 8) throw std::runtime_error("safetensors: header length exceeds file");
  common::json::Value h =
      common::json::parse(std::string_view(reinterpret_cast<const char*>(p) + 8, hlen));
  std::vector<std::pair<std::string, TensorInfo>> out;
  for (const auto& [name, e] : h.obj()) {
    if (name == "__metadata__") continue;
    TensorInfo t;
    t.dtype = e.at("dtype").str();
    for (const auto& d : e.at("shape").arr()) t.shape.push_back(uint64_t(d.num()));
    const auto& off = e.at("data_offsets").arr();
    // Indexed by [0]/[1] two lines down: a header is free to write any array,
    // so the pair is checked before it is read (the file is untrusted input).
    if (off.size() != 2)
      throw std::runtime_error("safetensors: tensor '" + name + "' has " +
                               std::to_string(off.size()) +
                               " data_offsets, expected exactly 2 [begin, end]");
    t.begin = uint64_t(off[0].num());
    t.end = uint64_t(off[1].num());
    out.emplace_back(name, std::move(t));
  }
  return out;
}

SafetensorsSet::SafetensorsSet(const std::string& snapshot_dir) {
  std::ifstream ixf(snapshot_dir + "model.safetensors.index.json");
  if (!ixf)
    throw std::runtime_error("no model.safetensors.index.json in " + snapshot_dir +
                             " (single-file checkpoints are out of scope)");
  std::stringstream ss;
  ss << ixf.rdbuf();
  common::json::Value ix = common::json::parse(ss.str());
  const auto& wm = ix.at("weight_map").obj();

  std::map<std::string, uint32_t> file_id;
  std::map<std::string, std::map<std::string, TensorInfo>> per_file;  // file -> header map
  for (const auto& [tensor, filev] : wm) {
    const std::string& fname = filev.str();
    if (file_id.find(fname) == file_id.end()) {
      file_id[fname] = uint32_t(files_.size());
      files_.emplace_back(snapshot_dir + fname);
      names_.push_back(fname);
      auto entries = parse_header(files_.back().data(), files_.back().size());
      uint64_t hlen = 0;
      std::memcpy(&hlen, files_.back().data(), 8);
      data_start_.push_back(size_t(8 + hlen));
      auto& m = per_file[fname];
      for (auto& [n, t] : entries) m.emplace(n, std::move(t));
    }
    auto& header = per_file[fname];
    auto it = header.find(tensor);
    if (it == header.end())
      throw std::runtime_error("index names tensor '" + tensor + "' in " + fname +
                               " but the file's header has no such tensor");
    TensorInfo t = it->second;
    t.file = file_id[fname];
    const size_t avail = files_[t.file].size() - data_start_[t.file];
    if (t.begin > t.end || t.end > avail)
      throw std::runtime_error("tensor '" + tensor + "' in " + fname +
                               " has data_offsets [" + std::to_string(t.begin) + ", " +
                               std::to_string(t.end) + "] outside the file's " +
                               std::to_string(avail) + "-byte data section");
    tensors_.emplace(tensor, std::move(t));   // index is the manifest: one entry per name
  }
}

const uint8_t* SafetensorsSet::data(const TensorInfo& t) const {
  return files_[t.file].data() + data_start_[t.file] + t.begin;
}

}  // namespace loader
