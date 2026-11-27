#include "loader/safetensors.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <memory>
#include <string>
#include <cstring>
#include <fstream>
#include <thread>
#include <set>
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
  if (p == MAP_FAILED) { ::close(fd); throw std::runtime_error("mmap failed: " + path); }
  fd_ = fd;
  ::madvise(p, size_, MADV_SEQUENTIAL);
  data_ = static_cast<const uint8_t*>(p);
}
MappedFile::~MappedFile() {
  if (data_) ::munmap(const_cast<uint8_t*>(data_), size_);
  if (fd_ >= 0) {
    ::posix_fadvise(fd_, 0, 0, POSIX_FADV_DONTNEED);   // unmapped first: the advice drops only unmapped pages
    ::close(fd_);
  }
}
MappedFile::MappedFile(MappedFile&& o) noexcept : data_(o.data_), size_(o.size_), fd_(o.fd_) {
  o.data_ = nullptr;
  o.size_ = 0;
  o.fd_ = -1;
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

SafetensorsSet::SafetensorsSet(const std::string& snapshot_dir) : SafetensorsSet(snapshot_dir, false) {}

SafetensorsSet::SafetensorsSet(const std::string& snapshot_dir, bool single_file_ok)
    : SafetensorsSet(snapshot_dir, single_file_ok, nullptr) {}

SafetensorsSet::SafetensorsSet(const std::string& snapshot_dir, const std::function<bool(const std::string&)>& skip)
    : SafetensorsSet(snapshot_dir, false, skip) {}

SafetensorsSet::SafetensorsSet(const std::string& snapshot_dir, bool single_file_ok,
                               const std::function<bool(const std::string&)>& skip) {
  std::ifstream ixf(snapshot_dir + "model.safetensors.index.json");
  if (!ixf && single_file_ok && std::ifstream(snapshot_dir + "model.safetensors")) {
    // One file, its header the manifest (spec 19b): the same bounds check as an indexed shard's.
    const std::string fname = "model.safetensors";
    files_.emplace_back(snapshot_dir + fname);
    names_.push_back(fname);
    if (files_.back().size() < 8)
      throw std::runtime_error(snapshot_dir + fname + " is shorter than its 8-byte header length");
    uint64_t hlen = 0;
    std::memcpy(&hlen, files_.back().data(), 8);
    data_start_.push_back(size_t(8 + hlen));
    const size_t avail = files_.back().size() - data_start_.back();
    for (auto& [n, t] : parse_header(files_.back().data(), files_.back().size())) {
      t.file = 0;
      if (t.begin > t.end || t.end > avail)
        throw std::runtime_error("tensor '" + n + "' in " + fname + " has data_offsets [" +
                                 std::to_string(t.begin) + ", " + std::to_string(t.end) + "] outside the file's " +
                                 std::to_string(avail) + "-byte data section");
      tensors_.emplace(n, std::move(t));
    }
    return;
  }
  if (!ixf)
    throw std::runtime_error("no model.safetensors.index.json in " + snapshot_dir +
                             " (single-file checkpoints are out of scope)");
  std::stringstream ss;
  ss << ixf.rdbuf();
  common::json::Value ix = common::json::parse(ss.str());
  const auto& wm = ix.at("weight_map").obj();
  // Files holding only skipped tensors: never opened.
  std::set<std::string> unread;
  if (skip) {
    std::set<std::string> read;
    for (const auto& [tensor, filev] : wm) (skip(tensor) ? unread : read).insert(filev.str());
    for (const std::string& f : read) unread.erase(f);
  }

  std::map<std::string, uint32_t> file_id;
  std::map<std::string, std::map<std::string, TensorInfo>> per_file;  // file -> header map
  for (const auto& [tensor, filev] : wm) {
    const std::string& fname = filev.str();
    if (unread.contains(fname)) continue;
    if (!file_id.contains(fname)) {
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

std::vector<FileRange> SafetensorsSet::ranges(const std::vector<const TensorInfo*>& ts) const {
  std::vector<FileRange> r;
  for (const TensorInfo* t : ts) r.push_back({t->file, data_start_[t->file] + t->begin, data_start_[t->file] + t->end});
  std::sort(r.begin(), r.end(), [](const FileRange& a, const FileRange& b) {
    return a.file != b.file ? a.file < b.file : a.begin < b.begin;
  });
  std::vector<FileRange> merged;
  for (const FileRange& x : r) {
    if (!merged.empty() && merged.back().file == x.file && x.begin <= merged.back().end)
      merged.back().end = std::max(merged.back().end, x.end);
    else
      merged.push_back(x);
  }
  return merged;
}

size_t SafetensorsSet::read_ahead(const std::vector<FileRange>& r, unsigned readers) const {
  constexpr uint64_t kBlock = uint64_t(8) << 20;
  std::vector<FileRange> blocks;   // the ranges cut into kBlock pieces, in order: the readers take the next one
  for (const FileRange& x : r)
    for (uint64_t b = x.begin; b < x.end; b += kBlock) blocks.push_back({x.file, b, std::min(x.end, b + kBlock)});
  std::atomic<size_t> next{0}, bytes{0};
  std::vector<std::thread> pool;
  for (unsigned i = 0; i < std::max(1u, readers); ++i)
    pool.emplace_back([&] {
      std::vector<uint8_t> buf(kBlock);
      for (size_t k; (k = next++) < blocks.size();) {
        const FileRange& b = blocks[k];
        for (uint64_t at = b.begin; at < b.end;) {
          const ssize_t n = ::pread(files_[b.file].fd(), buf.data(), size_t(b.end - at), off_t(at));
          if (n <= 0) break;   // the repack's own read of the mapping reports a short file
          at += uint64_t(n);
          bytes += size_t(n);
        }
      }
    });
  for (std::thread& t : pool) t.join();
  return bytes;
}

namespace {
constexpr uint64_t kPage = 4096;
// A range's whole pages [lo, hi) (lo >= hi: none).
std::pair<uint64_t, uint64_t> whole_pages(const FileRange& x) {
  return {(x.begin + kPage - 1) / kPage * kPage, x.end / kPage * kPage};
}
}  // namespace

size_t SafetensorsSet::read_mapped(const std::vector<FileRange>& r, unsigned readers) const {
  constexpr uint64_t kBlock = uint64_t(8) << 20;
  std::map<uint32_t, int> direct;   // file -> its O_DIRECT descriptor (-1: refused)
  for (const FileRange& x : r)
    if (!direct.contains(x.file)) {
      const std::string self = "/proc/self/fd/" + std::to_string(files_[x.file].fd());
      direct[x.file] = ::open(self.c_str(), O_RDONLY | O_DIRECT);
    }
  struct Block {
    FileRange b;
    bool direct;
  };
  std::vector<Block> blocks;
  std::vector<std::pair<uint8_t*, size_t>> overlays;
  const auto buffered = [&](uint32_t f, uint64_t begin, uint64_t end) {
    for (uint64_t b = begin; b < end; b += kBlock) blocks.push_back({{f, b, std::min(end, b + kBlock)}, false});
  };
  for (const FileRange& x : r) {
    const auto [lo, hi] = whole_pages(x);
    if (direct[x.file] < 0 || lo >= hi) {
      buffered(x.file, x.begin, x.end);
      continue;
    }
    buffered(x.file, x.begin, lo);
    buffered(x.file, hi, x.end);
    uint8_t* at = const_cast<uint8_t*>(files_[x.file].data()) + lo;
    if (::mmap(at, hi - lo, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == MAP_FAILED)
      throw std::runtime_error(names_[x.file] + ": cannot map memory over bytes [" + std::to_string(lo) + ", " +
                               std::to_string(hi) + ") to read them into");
    overlays.emplace_back(at, hi - lo);
    for (uint64_t b = lo; b < hi; b += kBlock) blocks.push_back({{x.file, b, std::min(hi, b + kBlock)}, true});
  }
  std::atomic<size_t> next{0}, bytes{0};
  std::atomic<bool> short_read{false};
  std::vector<std::thread> pool;
  for (unsigned i = 0; i < std::max(1u, readers); ++i)
    pool.emplace_back([&] {
      std::vector<uint8_t> buf(kBlock);
      for (size_t k; (k = next++) < blocks.size() && !short_read;) {
        const auto& [b, d] = blocks[k];
        uint8_t* dst = d ? const_cast<uint8_t*>(files_[b.file].data()) + b.begin : buf.data();
        const int fd = d ? direct[b.file] : files_[b.file].fd();
        for (uint64_t at = b.begin; at < b.end;) {
          const ssize_t n = ::pread(fd, dst + (at - b.begin), size_t(b.end - at), off_t(at));
          if (n <= 0) {
            short_read = short_read || d;   // a buffered piece: the repack's own read reports a short file
            break;
          }
          at += uint64_t(n);
          bytes += size_t(n);
        }
      }
    });
  for (std::thread& t : pool) t.join();
  for (const auto& [f, fd] : direct)
    if (fd >= 0) ::close(fd);
  for (const auto& [at, n] : overlays) ::mprotect(at, n, PROT_READ);
  if (short_read) throw std::runtime_error("a checkpoint file is shorter than its header says (an O_DIRECT read came back short)");
  return bytes;
}

void SafetensorsSet::release_mapped(const std::vector<FileRange>& r) const {
  for (const FileRange& x : r) {
    const auto [lo, hi] = whole_pages(x);
    if (lo >= hi) continue;
    uint8_t* at = const_cast<uint8_t*>(files_[x.file].data()) + lo;
    if (::mmap(at, hi - lo, PROT_READ, MAP_PRIVATE | MAP_FIXED, files_[x.file].fd(), off_t(lo)) == MAP_FAILED)
      throw std::runtime_error(names_[x.file] + ": cannot map the file back over bytes [" + std::to_string(lo) + ", " +
                               std::to_string(hi) + ")");
  }
}

void SafetensorsSet::read_bytes(const TensorInfo& t, uint64_t off, size_t n, void* dst) const {
  const uint64_t at = data_start_[t.file] + t.begin + off;
  ::posix_fadvise(files_[t.file].fd(), 0, 0, POSIX_FADV_RANDOM);
  if (off + n > t.end - t.begin || ::pread(files_[t.file].fd(), dst, n, off_t(at)) != ssize_t(n))
    throw std::runtime_error(names_[t.file] + ": a read of " + std::to_string(n) + " B at " + std::to_string(at) +
                             " came back short");
}

void SafetensorsSet::read_into(const TensorInfo& t, void* dst, unsigned readers) const {
  constexpr uint64_t kBlock = uint64_t(8) << 20, kAlign = 4096;
  const uint64_t begin = data_start_[t.file] + t.begin, end = data_start_[t.file] + t.end;
  if (begin == end) return;
  const std::string self = "/proc/self/fd/" + std::to_string(files_[t.file].fd());
  int fd = ::open(self.c_str(), O_RDONLY | O_DIRECT);
  const bool direct = fd >= 0;
  if (!direct) fd = files_[t.file].fd();
  const uint64_t first = begin / kAlign * kAlign, blocks = (end - first + kBlock - 1) / kBlock;
  std::atomic<uint64_t> next{0};
  std::atomic<bool> short_read{false};
  std::vector<std::thread> pool;
  for (unsigned i = 0; i < std::max(1u, readers); ++i)
    pool.emplace_back([&] {
      const std::unique_ptr<uint8_t, decltype(&std::free)> buf(static_cast<uint8_t*>(std::aligned_alloc(kAlign, kBlock)),
                                                               &std::free);
      for (uint64_t k; (k = next++) < blocks && !short_read;) {
        const uint64_t at = first + k * kBlock, want = std::min(kBlock, end - at);
        const uint64_t len = (want + kAlign - 1) / kAlign * kAlign;   // O_DIRECT reads whole sectors; EOF cuts it
        uint64_t got = 0;
        while (got < want) {
          const ssize_t n = ::pread(fd, buf.get() + got, size_t(len - got), off_t(at + got));
          if (n <= 0) break;
          got += uint64_t(n);
        }
        if (got < want) {
          short_read = true;
          break;
        }
        const uint64_t lo = std::max(at, begin);
        std::memcpy(static_cast<uint8_t*>(dst) + (lo - begin), buf.get() + (lo - at), size_t(at + want - lo));
      }
    });
  for (std::thread& th : pool) th.join();
  if (direct) ::close(fd);
  if (short_read)
    throw std::runtime_error(names_[t.file] + ": a read of bytes [" + std::to_string(begin) + ", " +
                             std::to_string(end) + ") came back short - the file is shorter than its header says");
}

}  // namespace loader
