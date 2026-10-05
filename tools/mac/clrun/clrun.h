// tools/mac/clrun/clrun.h - build one of this repo's .cl kernels with given defines on
// the Mac's OpenCL device (Apple's OpenCL 1.2 framework) and run it on host buffers.
//
// **Indicative only.** Apple's OpenCL compiler is not ocloc, the Mac's GPU (Intel UHD 630
// on the Intel Macs, or an Apple GPU) is not a B70, and the subgroup operations are
// EMULATED by tools/mac/opencl/intel_shim.h (B70_CL_EMULATE): a "subgroup" is 16
// consecutive work-items and a block read is the plain load it is specified to equal.
// A kernel that agrees with its host reference here has right indexing, right guards and
// right arithmetic order on SOME device; one that disagrees has a bug worth reading
// before the box. Rounding (fp contraction), timing, register pressure, DPAS, 2D block
// I/O and cross-lane operations are not tested here at all - the B70 is the real test.
//
// Typical use (tools/mac/clrun/gemv_i8w_run.cc is a complete driver):
//
//   clrun::Device dev;                                   // B70_MAC_CL_DEVICE picks one
//   clrun::Program prog(dev, "src/kernels/gemv_i8w.cl", {"M=1", "K=512", "N=256"});
//   clrun::Buffer w(dev, host_w), out(dev, n * sizeof(float));
//   prog.run("gemv_i8w", {N}, {64}, w, scales, x, out);
//   std::vector<float> got = out.read<float>();
#pragma once

#define CL_SILENCE_DEPRECATION
#include <OpenCL/opencl.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace clrun {

struct Error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// One device, its context and an in-order queue. The device is the first whose name
// contains $B70_MAC_CL_DEVICE (case-sensitive), else the first GPU whose name contains
// "Intel" (the closest relative of the B70 a Mac can have), else the first GPU.
class Device {
 public:
  Device();
  ~Device();
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  const std::string& name() const { return name_; }
  cl_device_id id() const { return dev_; }
  cl_context context() const { return ctx_; }
  cl_command_queue queue() const { return q_; }

 private:
  cl_device_id dev_ = nullptr;
  cl_context ctx_ = nullptr;
  cl_command_queue q_ = nullptr;
  std::string name_;
};

class Buffer {
 public:
  Buffer(Device& dev, std::size_t bytes);  // zero-filled
  template <class T>
  Buffer(Device& dev, const std::vector<T>& host) : Buffer(dev, host.size() * sizeof(T)) {
    write(host.data(), host.size() * sizeof(T));
  }
  ~Buffer();
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  void write(const void* src, std::size_t bytes);
  void read(void* dst, std::size_t bytes) const;
  template <class T>
  std::vector<T> read() const {
    std::vector<T> v(bytes_ / sizeof(T));
    read(v.data(), v.size() * sizeof(T));
    return v;
  }
  cl_mem mem() const { return mem_; }
  std::size_t bytes() const { return bytes_; }

 private:
  Device& dev_;
  cl_mem mem_ = nullptr;
  std::size_t bytes_ = 0;
};

// A kernel argument: a Buffer, or any trivially copyable scalar/vector by value.
struct Arg {
  std::size_t size;
  const void* ptr;
  cl_mem mem = nullptr;
  Arg(const Buffer& b) : size(sizeof(cl_mem)), ptr(nullptr), mem(b.mem()) {}  // NOLINT
  template <class T>
  Arg(const T& v) : size(sizeof(T)), ptr(&v) {}  // NOLINT
};

class Program {
 public:
  // `cl_path` is relative to the repo root (or absolute); `defines` are NAME=VALUE (or
  // NAME) without -D, exactly as src/kernels/CMakeLists.txt passes them to ocloc. The
  // source is compiled after intel_shim.h in its B70_CL_EMULATE mode.
  Program(Device& dev, const std::string& cl_path, const std::vector<std::string>& defines);
  ~Program();
  Program(const Program&) = delete;
  Program& operator=(const Program&) = delete;

  // Enqueue `kernel` over an N-D range (global/local sizes per dimension, 1 to 3) and
  // wait for it. Throws with the CL error name on failure.
  void run(const char* kernel, const std::vector<std::size_t>& global,
           const std::vector<std::size_t>& local, const std::vector<Arg>& args);
  template <class... A>
  void run(const char* kernel, const std::vector<std::size_t>& global,
           const std::vector<std::size_t>& local, const A&... args) {
    run(kernel, global, local, std::vector<Arg>{Arg(args)...});
  }

  const std::string& build_log() const { return log_; }

 private:
  Device& dev_;
  cl_program prog_ = nullptr;
  std::string log_;
};

// The repo root: $B70_ROOT, else the B70_CLRUN_ROOT the drivers were compiled with, else ".".
std::string repo_root();

// bf16 <-> fp32, round-to-nearest-even, as the kernels' references use.
std::uint16_t f32_to_bf16(float f);
float bf16_to_f32(std::uint16_t h);

}  // namespace clrun
