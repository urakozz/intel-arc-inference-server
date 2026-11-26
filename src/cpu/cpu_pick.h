#pragma once
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace cpu {

// The box shares its CPU with other jobs, and a spinning barrier waits for its slowest thread. Sample /proc/stat for
// `sample_ms` and order the cpus: both hyperthreads of a core idle (one cpu of each such core) first, then the other idle
// cpus, then busy ones; the hyperthread siblings last. physical_cores() is the count of the first group of entries.
inline std::vector<int> pick_cpus(unsigned sample_ms = 200) {
  auto read = [] {
    std::vector<std::pair<uint64_t, uint64_t>> v;   // total, idle
    std::ifstream f("/proc/stat");
    std::string line;
    while (std::getline(f, line))
      if (line.compare(0, 3, "cpu") == 0 && line[3] != ' ') {
        std::istringstream is(line.substr(3));
        int id;
        uint64_t x, tot = 0, idle = 0;
        is >> id;
        for (int i = 0; is >> x; ++i) { tot += x; if (i == 3) idle = x; }
        if (size_t(id) >= v.size()) v.resize(id + 1);
        v[id] = {tot, idle};
      }
    return v;
  };
  const auto a = read();
  std::this_thread::sleep_for(std::chrono::milliseconds(sample_ms));
  const auto b = read();
  std::vector<bool> idle(a.size());
  for (size_t i = 0; i < a.size(); ++i) idle[i] = double(b[i].second - a[i].second) > 0.9 * double(b[i].first - a[i].first);
  std::vector<int> first, second, third;
  for (size_t i = 0; i < a.size(); ++i) {
    std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(i) + "/topology/thread_siblings_list");
    std::string sib;
    f >> sib;   // "i,j" (or "i-j")
    int lo = std::atoi(sib.c_str());
    int hi = sib.size() > 1 ? std::atoi(sib.c_str() + sib.find_first_of(",-") + 1) : lo;
    if (int(i) != lo) continue;   // one entry per core, by its lower cpu
    if (idle[lo] && idle[hi]) first.push_back(lo);
    else if (idle[lo]) second.push_back(lo);
    else third.push_back(lo);
  }
  std::vector<int> all = first;
  all.insert(all.end(), second.begin(), second.end());
  all.insert(all.end(), third.begin(), third.end());
  for (size_t i = 0; i < a.size(); ++i) {   // the hyperthread siblings after every core's first cpu
    std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(i) + "/topology/thread_siblings_list");
    std::string sib;
    f >> sib;
    if (int(i) != std::atoi(sib.c_str())) all.push_back(int(i));
  }
  return all;
}

// One cpu per physical core (the lower cpu of each thread_siblings_list), from sysfs.
inline unsigned physical_cores() {
  unsigned n = 0;
  for (unsigned i = 0;; ++i) {
    std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(i) + "/topology/thread_siblings_list");
    if (!f) break;
    std::string sib;
    f >> sib;
    if (int(i) == std::atoi(sib.c_str())) ++n;
  }
  return n ? n : std::thread::hardware_concurrency();
}

}  // namespace cpu
