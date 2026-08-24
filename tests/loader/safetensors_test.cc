#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "check.h"
#include "loader/safetensors.h"
#include "loader/snapshot.h"

// Writes one safetensors file: 8-byte LE header length, header JSON, data.
static void write_st(const std::string& path, const std::string& header,
                     const std::vector<uint8_t>& data) {
  std::ofstream f(path, std::ios::binary);
  uint64_t n = header.size();
  f.write(reinterpret_cast<const char*>(&n), 8);
  f.write(header.data(), std::streamsize(header.size()));
  f.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
}

int main() {
  const std::string dir = "/tmp/b70_st_test";
  std::remove((dir + "/a.safetensors").c_str());
  std::remove((dir + "/b.safetensors").c_str());
  std::remove((dir + "/model.safetensors.index.json").c_str());
  std::remove((dir + "/config.json").c_str());
  // GCC's warn_unused_result on system() is not silenced by a (void) cast,
  // so the result is checked instead (-Werror is live).
  CHECK(system(("mkdir -p " + dir).c_str()) == 0);

  // t0: 4 bf16 values in a.safetensors; t1: 8 bytes I32 in b.safetensors;
  // "dup" appears in BOTH files with different bytes - the index points at b.
  std::vector<uint8_t> da = {1, 0, 2, 0, 3, 0, 4, 0, /*dup in a:*/ 0xAA, 0xAA};
  std::vector<uint8_t> db = {5, 0, 0, 0, 6, 0, 0, 0, /*dup in b:*/ 0xBB, 0xBB};
  write_st(dir + "/a.safetensors",
           R"({"t0":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},)"
           R"("dup":{"dtype":"BF16","shape":[1],"data_offsets":[8,10]}})", da);
  write_st(dir + "/b.safetensors",
           R"({"t1":{"dtype":"I32","shape":[2],"data_offsets":[0,8]},)"
           R"("dup":{"dtype":"BF16","shape":[1],"data_offsets":[8,10]}})", db);
  std::ofstream(dir + "/model.safetensors.index.json")
      << R"({"metadata":{},"weight_map":{"t0":"a.safetensors","t1":"b.safetensors","dup":"b.safetensors"}})";
  std::ofstream(dir + "/config.json") << "{}";

  // resolve_snapshot: a directory with config.json resolves to itself.
  std::string snap = loader::resolve_snapshot(dir);
  CHECK(snap.back() == '/');

  loader::SafetensorsSet set(snap);
  CHECK_EQ(set.tensors().size(), size_t(3));
  const auto& t0 = set.tensors().at("t0");
  CHECK_EQ(t0.dtype, std::string("BF16"));
  CHECK_EQ(t0.shape[0], uint64_t(2));
  CHECK_EQ(set.bytes(t0), size_t(8));
  CHECK_EQ(set.data(t0)[0], uint8_t(1));
  const auto& t1 = set.tensors().at("t1");
  CHECK_EQ(reinterpret_cast<const uint32_t*>(set.data(t1))[1], uint32_t(6));
  // Dedup by index: "dup" must come from b.safetensors.
  const auto& dup = set.tensors().at("dup");
  CHECK_EQ(set.data(dup)[0], uint8_t(0xBB));

  // Missing tensor in named file -> throw naming both.
  std::ofstream(dir + "/model.safetensors.index.json")
      << R"({"weight_map":{"ghost":"a.safetensors"}})";
  bool threw = false;
  try { loader::SafetensorsSet bad(snap); } catch (const std::runtime_error& e) {
    threw = std::string(e.what()).find("ghost") != std::string::npos;
  }
  CHECK(threw);

  // Repo-id resolution failure names the path it tried.
  threw = false;
  try { loader::resolve_snapshot("no-such-org/no-such-model"); }
  catch (const std::runtime_error& e) {
    threw = std::string(e.what()).find("models--no-such-org--no-such-model") != std::string::npos;
  }
  CHECK(threw);
  std::puts("safetensors_test OK");
  return 0;
}
