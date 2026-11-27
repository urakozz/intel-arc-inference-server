#pragma once
// b70-decode --dump-taps: the file an offline drafter acceptance measurement reads (tools/oracle/dump_taps.py's
// layout, tools/oracle/dflash_accept.py): ids, tap.L<id> [rows, hidden] bf16 a drafter tap, greedy (the id after each
// position; -1 on prompt rows but the last), top_ids / top_logits (the head's top 64, ties to the lower id) and lse
// (log-sum-exp over the live vocabulary); metadata name, n_prompt, n, tap_from, layers, vocab_used, snapshot, kv. Rows
// are positions tap_from .. n - 1: the prompt's last kKeep and every generated one. Inkling and Qwen3.8-Flash-Next
// write it.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cli {

struct TapDump {
  static constexpr uint32_t kTop = 64, kKeep = 2048;
  uint32_t n_prompt = 0, tap_from = 0, rows = 0, hidden = 0, vocab_used = 0;
  std::vector<uint32_t> layers;              // the drafter's tap ids, in its order
  std::vector<int32_t> ids, greedy, top_ids;
  std::vector<float> top_logits, lse;
  std::vector<std::vector<uint16_t>> taps;   // per tap: bf16 [rows][hidden]
  std::string snapshot, kv;

  TapDump(uint32_t np, uint32_t n, uint32_t hidden_, uint32_t vocab_used_, std::vector<uint32_t> layers_)
      : n_prompt(np), tap_from(np > kKeep ? np - kKeep : 0), rows(np + n - tap_from), hidden(hidden_),
        vocab_used(vocab_used_), layers(std::move(layers_)), greedy(rows, -1), top_ids(size_t(rows) * kTop, -1),
        top_logits(size_t(rows) * kTop, 0.f), lse(rows, 0.f),
        taps(layers.size(), std::vector<uint16_t>(size_t(rows) * hidden)) {}

  // Row `row`'s top 64 and log-sum-exp from `logits` (at least vocab_used values).
  void head(uint32_t row, const float* logits) {
    std::vector<uint32_t> order(vocab_used);
    for (uint32_t i = 0; i < vocab_used; ++i) order[i] = i;
    std::partial_sort(order.begin(), order.begin() + kTop, order.end(), [&](uint32_t a, uint32_t b) {
      return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
    });
    const double top = logits[order[0]];
    double sum = 0.;
    for (uint32_t i = 0; i < vocab_used; ++i) sum += std::exp(double(logits[i]) - top);
    lse[row] = float(top + std::log(sum));
    for (uint32_t j = 0; j < kTop; ++j) {
      top_ids[size_t(row) * kTop + j] = int32_t(order[j]);
      top_logits[size_t(row) * kTop + j] = logits[order[j]];
    }
  }

  // safetensors: the header JSON padded to 8 bytes, then the tensors back to back in the header's order. The name is
  // the file's (<corpus>__<label>.taps.safetensors -> <corpus>/<label>).
  void write(const std::string& path) const {
    std::string name = std::filesystem::path(path).filename().string();
    if (const auto dot = name.find(".taps.safetensors"); dot != std::string::npos) name.resize(dot);
    if (const auto sep = name.find("__"); sep != std::string::npos) name.replace(sep, 2, "/");
    struct Tensor {
      std::string name, dtype;
      std::vector<uint64_t> shape;
      const void* data;
      size_t bytes;
    };
    std::vector<Tensor> tensors = {
        {"ids", "I32", {ids.size()}, ids.data(), ids.size() * 4},
        {"greedy", "I32", {rows}, greedy.data(), greedy.size() * 4},
        {"top_ids", "I32", {rows, kTop}, top_ids.data(), top_ids.size() * 4},
        {"top_logits", "F32", {rows, kTop}, top_logits.data(), top_logits.size() * 4},
        {"lse", "F32", {rows}, lse.data(), lse.size() * 4}};
    std::string list;
    for (size_t t = 0; t < layers.size(); ++t) {
      tensors.push_back({"tap.L" + std::to_string(layers[t]), "BF16", {rows, hidden}, taps[t].data(), taps[t].size() * 2});
      list += (t ? "," : "") + std::to_string(layers[t]);
    }
    std::string header = "{\"__metadata__\":{\"name\":\"" + name + "\",\"n_prompt\":\"" + std::to_string(n_prompt) +
                         "\",\"n\":\"" + std::to_string(ids.size()) + "\",\"tap_from\":\"" + std::to_string(tap_from) +
                         "\",\"layers\":\"" + list + "\",\"vocab_used\":\"" + std::to_string(vocab_used) +
                         "\",\"snapshot\":\"" + snapshot + "\",\"kv\":\"" + kv + "\"}";
    uint64_t offset = 0;
    for (const auto& t : tensors) {
      header += ",\"" + t.name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[";
      for (size_t i = 0; i < t.shape.size(); ++i) header += (i ? "," : "") + std::to_string(t.shape[i]);
      header += "],\"data_offsets\":[" + std::to_string(offset) + "," + std::to_string(offset + t.bytes) + "]}";
      offset += t.bytes;
    }
    header += "}";
    while (header.size() % 8) header += ' ';
    const std::string tmp = path + ".tmp";
    {
      std::ofstream out(tmp, std::ios::binary);
      const uint64_t length = header.size();
      out.write(reinterpret_cast<const char*>(&length), 8);
      out.write(header.data(), std::streamsize(header.size()));
      for (const auto& t : tensors) out.write(static_cast<const char*>(t.data), std::streamsize(t.bytes));
      if (!out) throw std::runtime_error("Cannot write " + path);
    }
    std::filesystem::rename(tmp, path);
  }
};

}  // namespace cli
