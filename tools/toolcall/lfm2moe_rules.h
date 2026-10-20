#pragma once
// Spec 24e (plan 24e Task 5, decision 7's evidence): how often each lenient reading of the pythonic tool-call parser
// (server/toolcall_lfm2.h) fires on generated text - the body of tools/toolcall/lfm2moe_rules.cc, a header so that
// tests/server/lfm2moe_rules_test.cc runs it.
//
// Each file is a generation (tools/toolcall/engine_generate.sh's <name>.<tag>.txt, or oracle_generate.py's <name>.bf16.txt):
// the decode of the ids with special tokens kept, which opens inside reasoning (the prompt ends in <think>). b70-decode
// generates its --n ids whatever the model says, so each text is cut at the first <|im_end|> (the EOS the server stops on)
// before it is read - the text b70-serve would have parsed. One line a file (calls, reasoning bytes, content bytes, the
// counters, and a note when a block was left as content) and the totals; the result is the number of files whose
// tool-call block was NOT read as calls (an invalid block left in the content), so a run with 0 parse failures is 0.
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "server/toolcall.h"
#include "server/toolcall_lfm2.h"

namespace lfm2moe_rules {

inline size_t run(const std::vector<std::string>& files, std::FILE* out) {
  server::PythonicRules total;
  size_t with_calls = 0, call_total = 0, failures = 0;
  std::fprintf(out, "%-44s %5s %9s %9s  %s\n", "file", "calls", "reasoning", "content",
               "rules (quote name bare json pdict dropped stray)");
  for (const std::string& file : files) {
    std::ifstream in(file, std::ios::binary);
    if (!in) throw std::runtime_error("lfm2moe_rules: cannot read " + file);
    std::stringstream ss;
    ss << in.rdbuf();
    std::string text = ss.str();
    const size_t eos = text.find("<|im_end|>");
    if (eos != std::string::npos) text.resize(eos);
    server::Lfm2OutputStream stream(nlohmann::json::array());
    std::vector<server::Delta> deltas = stream.push(text);
    for (server::Delta& d : stream.finish()) deltas.push_back(std::move(d));
    size_t calls = 0, reasoning = 0, content = 0;
    std::string content_text;
    for (const server::Delta& d : deltas) {
      if (d.kind == server::Delta::Call) {
        ++calls;
      } else if (d.kind == server::Delta::Reasoning) {
        reasoning += d.text.size();
      } else {
        content += d.text.size();
        content_text += d.text;
      }
    }
    const bool left_as_content = content_text.find("<|tool_call_start|>") != std::string::npos;
    if (left_as_content) ++failures;
    const server::PythonicRules& r = stream.rules();
    total += r;
    if (calls > 0) ++with_calls;
    call_total += calls;
    std::fprintf(out, "%-44s %5zu %9zu %9zu  %u %u %u %u %u %u %u%s\n", file.c_str(), calls, reasoning, content,
                 r.quote_recovered, r.quoted_name, r.bare_call, r.json_literal, r.positional_dict, r.dropped_unfinished,
                 r.stray_end, left_as_content ? "  [a tool-call block was left as content]" : "");
  }
  std::fprintf(out, "\n%zu files, %zu with calls (%zu calls), %zu with a block left as content\n", files.size(), with_calls,
               call_total, failures);
  std::fprintf(out, "rules fired: quote_recovered %u, quoted_name %u, bare_call %u, json_literal %u, positional_dict %u, "
                    "dropped_unfinished %u, stray_end %u\n", total.quote_recovered, total.quoted_name, total.bare_call,
               total.json_literal, total.positional_dict, total.dropped_unfinished, total.stray_end);
  return failures;
}

}  // namespace lfm2moe_rules
