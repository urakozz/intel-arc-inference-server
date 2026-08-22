#include "l0/module.h"
#include <fstream>
#include <iterator>
#include <vector>
#include "l0/error.h"
#include "l0/kernel.h"

namespace l0 {
Module::Module(Context& ctx, const std::string& bin_path) {
  std::ifstream f(bin_path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open kernel binary: " + bin_path);
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

  ze_module_desc_t d{};
  d.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  d.format = ZE_MODULE_FORMAT_NATIVE;
  d.inputSize = bytes.size();
  d.pInputModule = bytes.data();
  ze_module_build_log_handle_t log = nullptr;
  ze_result_t r = zeModuleCreate(ctx.handle(), ctx.device(), &d, &m_, &log);
  if (r != ZE_RESULT_SUCCESS) {
    std::string text;
    if (log) {
      size_t n = 0;
      zeModuleBuildLogGetString(log, &n, nullptr);
      text.resize(n);
      zeModuleBuildLogGetString(log, &n, text.data());
      zeModuleBuildLogDestroy(log);
    }
    throw std::runtime_error("zeModuleCreate(" + bin_path + ") -> " + result_name(r) + "\n" + text);
  }
  if (log) zeModuleBuildLogDestroy(log);
}
Module::~Module() {
  if (m_) zeModuleDestroy(m_);
}
Kernel Module::kernel(const char* name) { return Kernel(*this, name); }
}  // namespace l0
