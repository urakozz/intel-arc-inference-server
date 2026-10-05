// tools/mac/clrun/clrun.cc - see clrun.h (indicative Mac OpenCL runs).
#include "clrun.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace clrun {
namespace {

const char* cl_err(cl_int e) {
  switch (e) {
    case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
    case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
    case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
    case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
    default: return "CL error";
  }
}

void check(cl_int e, const char* what) {
  if (e != CL_SUCCESS) {
    throw Error(std::string(what) + ": " + cl_err(e) + " (" + std::to_string(e) + ")");
  }
}

std::string slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw Error("cannot read " + path);
  std::ostringstream s;
  s << f.rdbuf();
  return s.str();
}

std::string device_name(cl_device_id d) {
  char buf[256] = {};
  clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof buf - 1, buf, nullptr);
  return buf;
}

}  // namespace

std::string repo_root() {
  if (const char* r = std::getenv("B70_ROOT")) return r;
#ifdef B70_CLRUN_ROOT
  return B70_CLRUN_ROOT;  // tools/mac_check.sh compiles the drivers with the absolute root
#else
  return ".";
#endif
}

Device::Device() {
  cl_platform_id plat = nullptr;
  check(clGetPlatformIDs(1, &plat, nullptr), "clGetPlatformIDs");
  cl_device_id devs[16];
  cl_uint n = 0;
  check(clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 16, devs, &n), "clGetDeviceIDs");
  const char* want = std::getenv("B70_MAC_CL_DEVICE");
  cl_device_id pick = nullptr, intel_gpu = nullptr, any_gpu = nullptr;
  for (cl_uint i = 0; i < n; ++i) {
    const std::string nm = device_name(devs[i]);
    cl_device_type t = 0;
    clGetDeviceInfo(devs[i], CL_DEVICE_TYPE, sizeof t, &t, nullptr);
    if (want && nm.find(want) != std::string::npos && !pick) pick = devs[i];
    if ((t & CL_DEVICE_TYPE_GPU) && !any_gpu) any_gpu = devs[i];
    if ((t & CL_DEVICE_TYPE_GPU) && nm.find("Intel") != std::string::npos && !intel_gpu) {
      intel_gpu = devs[i];
    }
  }
  if (want && !pick) throw Error(std::string("no OpenCL device matches B70_MAC_CL_DEVICE=") + want);
  dev_ = pick ? pick : intel_gpu ? intel_gpu : any_gpu;
  if (!dev_) throw Error("no OpenCL GPU on this machine");
  name_ = device_name(dev_);
  cl_int e = CL_SUCCESS;
  ctx_ = clCreateContext(nullptr, 1, &dev_, nullptr, nullptr, &e);
  check(e, "clCreateContext");
  q_ = clCreateCommandQueue(ctx_, dev_, 0, &e);
  check(e, "clCreateCommandQueue");
}

Device::~Device() {
  if (q_) clReleaseCommandQueue(q_);
  if (ctx_) clReleaseContext(ctx_);
}

Buffer::Buffer(Device& dev, std::size_t bytes) : dev_(dev), bytes_(bytes) {
  cl_int e = CL_SUCCESS;
  mem_ = clCreateBuffer(dev.context(), CL_MEM_READ_WRITE, bytes ? bytes : 4, nullptr, &e);
  check(e, "clCreateBuffer");
  std::vector<unsigned char> zero(bytes ? bytes : 4, 0);
  write(zero.data(), zero.size());
}

Buffer::~Buffer() {
  if (mem_) clReleaseMemObject(mem_);
}

void Buffer::write(const void* src, std::size_t bytes) {
  check(clEnqueueWriteBuffer(dev_.queue(), mem_, CL_TRUE, 0, bytes, src, 0, nullptr, nullptr),
        "clEnqueueWriteBuffer");
}

void Buffer::read(void* dst, std::size_t bytes) const {
  check(clEnqueueReadBuffer(dev_.queue(), mem_, CL_TRUE, 0, bytes, dst, 0, nullptr, nullptr),
        "clEnqueueReadBuffer");
}

Program::Program(Device& dev, const std::string& cl_path, const std::vector<std::string>& defines)
    : dev_(dev) {
  const std::string root = repo_root();
  const std::string path = cl_path.size() && cl_path[0] == '/' ? cl_path : root + "/" + cl_path;
  // The defines and the shim go in as source text, not as -D/-include options. Apple's
  // compiler builds against a precompiled header, and on its CPU and AMD devices a -D
  // named like an identifier in that header (`M`, `K` - every kernel here has them) is
  // rejected ("definition of macro 'K' conflicts with an identifier used in the
  // precompiled header"); a #define in the source is not.
  std::string head = "#define B70_CL_EMULATE 1\n";
  for (const std::string& d : defines) {
    const std::size_t eq = d.find('=');
    head += "#define " + (eq == std::string::npos ? d : d.substr(0, eq) + " " + d.substr(eq + 1)) + "\n";
  }
  const std::string src = head + slurp(root + "/tools/mac/opencl/intel_shim.h") + "\n#line 1\n" +
                          slurp(path);
  const char* s = src.c_str();
  const std::size_t len = src.size();
  cl_int e = CL_SUCCESS;
  prog_ = clCreateProgramWithSource(dev.context(), 1, &s, &len, &e);
  check(e, "clCreateProgramWithSource");
  std::string opts;  // for the error message only
  for (const std::string& d : defines) opts += " -D" + d;
  const cl_device_id id = dev.id();
  const cl_int be = clBuildProgram(prog_, 1, &id, "", nullptr, nullptr);
  std::size_t log_len = 0;
  clGetProgramBuildInfo(prog_, id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_len);
  log_.assign(log_len, '\0');
  if (log_len) clGetProgramBuildInfo(prog_, id, CL_PROGRAM_BUILD_LOG, log_len, &log_[0], nullptr);
  while (!log_.empty() && (log_.back() == '\0' || log_.back() == '\n')) log_.pop_back();
  if (be != CL_SUCCESS) {
    throw Error("build of " + cl_path + " [" + opts + " ] failed on " + dev.name() + ":\n" + log_);
  }
}

Program::~Program() {
  if (prog_) clReleaseProgram(prog_);
}

void Program::run(const char* kernel, const std::vector<std::size_t>& global,
                  const std::vector<std::size_t>& local, const std::vector<Arg>& args) {
  if (global.empty() || global.size() > 3 || global.size() != local.size()) {
    throw Error("run: global and local need the same 1..3 dimensions");
  }
  cl_int e = CL_SUCCESS;
  cl_kernel k = clCreateKernel(prog_, kernel, &e);
  check(e, kernel);
  for (cl_uint i = 0; i < args.size(); ++i) {
    const Arg& a = args[i];
    e = a.mem ? clSetKernelArg(k, i, sizeof(cl_mem), &a.mem) : clSetKernelArg(k, i, a.size, a.ptr);
    if (e != CL_SUCCESS) {
      clReleaseKernel(k);
      check(e, (std::string(kernel) + " arg " + std::to_string(i)).c_str());
    }
  }
  e = clEnqueueNDRangeKernel(dev_.queue(), k, static_cast<cl_uint>(global.size()), nullptr,
                             global.data(), local.data(), 0, nullptr, nullptr);
  if (e == CL_SUCCESS) e = clFinish(dev_.queue());
  clReleaseKernel(k);
  check(e, kernel);
}

std::uint16_t f32_to_bf16(float f) {
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<std::uint16_t>((u >> 16) | 0x40);  // NaN
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<std::uint16_t>(u >> 16);
}

float bf16_to_f32(std::uint16_t h) {
  const std::uint32_t u = static_cast<std::uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

}  // namespace clrun
