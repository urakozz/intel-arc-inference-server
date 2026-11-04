// Spec 20 (the 2026-11-04 tool-call item): Kolibri-1's long tool sessions through the server, with the REAL chat
// template (tests/tokenizer/kolibri's vendored tokenizer_config.json, chat::Template - template_kolibri_test holds it
// equal to transformers' on 16 message lists) and a scripted engine. A community report (HF Aleph-Alpha/Kolibri-1)
// says the model reasons but stops answering after 10-15 messages with tool memory; this checks OUR side:
//   - a 20-turn agentic session (two user queries, 18 tool-calling turns - two with parallel calls, three writing
//     multi-line file contents with RAW newlines / tabs, the repaired form - and two answers), every request the full
//     history as an OpenAI client sends it (the server's own response echoed, the tool results appended), sampled with
//     the card's 1.0 / 0.97 / 128 and reasoning_effort medium, odd turns streamed: every prompt is exactly the
//     template's render, by induction - the first one spelled out, each next one = the previous prompt + the turn the
//     model generated (its canonical form: a repaired call re-renders with \n escapes) + <|im_end|> + the tool
//     results in one user turn of <tool_response> blocks + the generation prompt; at the second user query every
//     earlier turn loses its <think> block (the template's last_query_index rule) and nothing else; every response
//     parses back to the turn's reasoning, content and calls, finish_reason tool_calls / stop;
//   - the same session from a client that does NOT echo reasoning_content: the turns after the last query render
//     "<think>\n\n</think>\n\n" (the template's rule for a turn without reasoning), and nothing else differs;
//   - the thinking switches: top-level reasoning_effort none / minimal / low / medium / high / xhigh / max (vLLM hands
//     it to the template as this kwarg) and chat_template_kwargs' reasoning_effort / enable_thinking false pick the
//     template's sentence and generation prompt; an explicit kwarg wins; a bad top-level value is a 400;
//   - a reasoning budget: max_tokens ending inside the reasoning is finish_reason length with the reasoning so far,
//     no content and no calls - in the body and in streamed frames.
//   kolibri_session_test <tests/tokenizer/kolibri>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "check.h"
#include "server/mock.h"
#include "server/openai.h"
#include "server/server.h"
#include "tokenizer/chat_template.h"

namespace {

using json = nlohmann::json;
using ojson = nlohmann::ordered_json;

struct RealTemplate : server::TemplateIface {
  explicit RealTemplate(const std::string& dir) : t(dir) {}
  std::string render(const ojson& m, const ojson& tools, bool think) override { return last = t.render(m, tools, think); }
  std::string render_with_kwargs(const ojson& m, const ojson& tools, bool think, const json& kwargs) override {
    return last = t.render(m, tools, think, kwargs);
  }
  chat::Template t;
  std::string last;
};

struct LongEngine : MockEngine {
  uint32_t max_len() override { return 262144; }
};

constexpr uint32_t kImEnd = 127906;

struct Reply {
  std::string finish, reasoning, content;
  json content_field;   // the body's message.content (null with calls and no content); streamed: the joined text
  ojson calls = ojson::array();
};

struct Fixture {
  explicit Fixture(const std::string& dir) : tmpl(dir), server({tok, tmpl, engine}, options()) { CHECK(server.start()); }
  ~Fixture() { server.stop(); }
  static server::Options options() {
    server::Options o;
    o.host = "127.0.0.1";
    o.port = 0;
    o.served_model = "kolibri";
    o.eos_ids = {kImEnd, 127901};
    o.chat_format = server::ChatFormat::for_model_type("kolibri1");
    return o;
  }
  // One UTF-8 character a token, then EOS (kolibri_server_test's script).
  void script(const std::string& text, bool eos = true) {
    engine.script.clear();
    engine.at = 0;
    for (size_t at = 0; at < text.size();) {
      size_t n = 1;
      while (at + n < text.size() && (static_cast<unsigned char>(text[at + n]) & 0xC0) == 0x80) ++n;
      const std::vector<uint32_t> id = tok.raw(text.substr(at, n), n);
      engine.script.insert(engine.script.end(), id.begin(), id.end());
      at += n;
    }
    if (eos) engine.script.push_back(kImEnd);
  }
  httplib::Client client() {
    httplib::Client c("127.0.0.1", server.bound_port());
    c.set_read_timeout(60, 0);
    return c;
  }
  int post_status(const ojson& body, std::string* text = nullptr) {
    auto c = client();
    const auto r = c.Post("/v1/chat/completions", body.dump(), "application/json");
    CHECK(r);
    if (text != nullptr) *text = r->body;
    return r->status;
  }
  Reply post(const ojson& body) {
    Reply out;
    auto c = client();
    if (!body.value("stream", false)) {
      const auto r = c.Post("/v1/chat/completions", body.dump(), "application/json");
      CHECK(r);
      if (r->status != 200) std::fprintf(stderr, "status %d: %s\n", r->status, r->body.c_str());
      CHECK_EQ(r->status, 200);
      const ojson b = ojson::parse(r->body);
      const ojson& choice = b.at("choices").at(0);
      const ojson& m = choice.at("message");
      out.finish = choice.at("finish_reason");
      out.content_field = json::parse(m.at("content").dump());
      if (m.at("content").is_string()) out.content = m.at("content");
      if (m.contains("reasoning_content")) out.reasoning = m.at("reasoning_content");
      if (m.contains("tool_calls")) out.calls = m.at("tool_calls");
      return out;
    }
    std::string sse;
    const auto r = c.Post("/v1/chat/completions", httplib::Headers(), body.dump(), "application/json",
                          [&sse](const char* d, size_t n) {
                            sse.append(d, n);
                            return true;
                          },
                          nullptr);
    CHECK(r);
    CHECK_EQ(r->status, 200);
    bool done = false;
    for (size_t at = 0, end; (end = sse.find("\n\n", at)) != std::string::npos; at = end + 2) {
      const std::string frame = sse.substr(at, end - at);
      if (frame == "data: [DONE]") {
        done = true;
        continue;
      }
      const ojson b = ojson::parse(frame.substr(6));
      const ojson& ch = b.at("choices").at(0);
      const ojson& d = ch.at("delta");
      if (d.contains("reasoning_content")) out.reasoning += d.at("reasoning_content").get<std::string>();
      if (d.contains("content") && d.at("content").is_string()) out.content += d.at("content").get<std::string>();
      if (d.contains("tool_calls")) {
        ojson call = d.at("tool_calls").at(0);
        CHECK_EQ(call.at("index").get<size_t>(), out.calls.size());
        call.erase("index");
        out.calls.push_back(call);
      }
      if (!ch.at("finish_reason").is_null()) out.finish = ch.at("finish_reason");
    }
    CHECK(done);
    out.content_field = out.content.empty() && !out.calls.empty() ? json(nullptr) : json(out.content);
    return out;
  }

  MockTok tok;
  RealTemplate tmpl;
  LongEngine engine;
  server::Server server;
};

// transformers' tojson (json.dumps: ", " and ": ", UTF-8 kept), in the value's own key order.
std::string py_dump(const ojson& v) {
  std::string s;
  if (v.is_object()) {
    s = "{";
    for (auto it = v.begin(); it != v.end(); ++it) s += (it == v.begin() ? "" : ", ") + ojson(it.key()).dump() + ": " + py_dump(it.value());
    return s + "}";
  }
  if (v.is_array()) {
    s = "[";
    for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + py_dump(v[i]);
    return s + "]";
  }
  return v.dump();
}

size_t count(const std::string& s, const std::string& what) {
  size_t n = 0;
  for (size_t at = s.find(what); at != std::string::npos; at = s.find(what, at + what.size())) ++n;
  return n;
}

void replace_all(std::string& s, const std::string& from, const std::string& to) {
  for (size_t at = s.find(from); at != std::string::npos; at = s.find(from, at + to.size())) s.replace(at, from.size(), to);
}

const ojson kTools = ojson::parse(R"([
  {"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {"type": "object",
    "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
  {"type": "function", "function": {"name": "write_file", "description": "Write a file.", "parameters": {"type": "object",
    "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
  {"type": "function", "function": {"name": "run", "description": "Run a shell command.", "parameters": {"type": "object",
    "properties": {"command": {"type": "string"}}, "required": ["command"]}}}
])");

const std::string kMedium =
    "Reasoning effort is set to medium. Think through the task methodically in the user's language, check key "
    "assumptions, and provide a well-supported answer.";
const std::string kHigh =
    "Reasoning effort is set to high. Think carefully through the task in the user's language, validate key assumptions, "
    "consider plausible alternatives, and prioritize correctness and clarity.";
const std::string kLow =
    "Reasoning effort is set to low. Think briefly through only the essential steps in the user's language, then proceed "
    "directly to the answer.";
const std::string kNone = "Reasoning is disabled. Proceed straight to answering according to the user's instructions.";

struct Turn {
  std::string user;   // a user message before this turn ("" = none)
  std::string reasoning, content;
  std::vector<std::pair<std::string, ojson>> calls;
  std::vector<std::string> results;
  bool raw = false;   // the model writes the string arguments' newlines / tabs as raw bytes
};

ojson args(std::initializer_list<std::pair<const char*, std::string>> kv) {
  ojson a = ojson::object();
  for (const auto& [k, v] : kv) a[k] = v;
  return a;
}

std::vector<Turn> session() {
  using P = std::pair<std::string, ojson>;
  const auto run = [](const std::string& cmd) { return P{"run", args({{"command", cmd}})}; };
  const auto read = [](const std::string& path) { return P{"read_file", args({{"path", path}})}; };
  const auto write = [](const std::string& path, const std::string& c) {
    return P{"write_file", args({{"path", path}, {"content", c}})};
  };
  const std::string fixed = "def add(a, b):\n\treturn a + b\n";
  std::vector<Turn> t = {
      {"Der Test in /w/test_calc.py schlägt fehl. Bitte repariere ihn.", "Ich lese zuerst den Test.", "",
       {read("/w/test_calc.py")}, {"def test_add():\n    assert add(2, 2) == 4"}},
      {"", "Jetzt die Implementierung.", "Ich schaue mir calc.py an.", {read("/w/calc.py")},
       {"def add(a, b):\n    return a - b"}},
      {"", "Falsches Vorzeichen. Erst den Fehler bestätigen.", "", {run("pytest -q")}, {"F\n1 failed in 0.02s"}},
      {"", "Stand des Repos prüfen.", "", {run("git status --short"), run("git log -1 --oneline")}, {"", "abc123 init"}},
      {"", "Die Datei mit dem richtigen Operator neu schreiben.", "", {write("/w/calc.py", fixed)}, {"ok"}, true},
      {"", "Tests erneut.", "", {run("pytest -q")}, {"1 passed in 0.01s"}},
      {"", "README prüfen, ob es die Funktion beschreibt.", "", {read("/w/README.md")}, {"# calc"}},
      {"", "Eine Zeile ergänzen.", "Ich ergänze die README.",
       {write("/w/README.md", "# calc\n\nAddiert „Größe“-Werte.\n")}, {"ok"}, true},
      {"", "Noch einmal alles.", "", {run("pytest -q")}, {"1 passed in 0.01s"}},
      {"", "Fertig.", "Der Fehler war `a - b` statt `a + b`; der Test läuft jetzt.", {}, {}},
      {"Lass bitte auch den Linter laufen.", "Ruff ausführen.", "", {run("ruff check /w")},
       {"calc.py:1:1: E302 expected 2 blank lines\nFound 1 error."}},
      {"", "Die Datei ansehen.", "", {read("/w/calc.py")}, {fixed}},
      {"", "Leerzeilen davor einfügen.", "", {write("/w/calc.py", "\n\ndef add(a, b):\n\treturn a + b\n")}, {"ok"}, true},
      {"", "Ruff erneut.", "", {run("ruff check /w")}, {"All checks passed!"}},
      {"", "Und die Tests.", "", {run("pytest -q")}, {"1 passed in 0.01s"}},
      {"", "Was hat sich geändert?", "", {run("git diff --stat")}, {" README.md | 2 ++\n calc.py   | 4 +++-"}},
      {"", "Beide Dateien gegenlesen.", "", {read("/w/calc.py"), read("/w/README.md")},
       {"\n\ndef add(a, b):\n\treturn a + b\n", "# calc\n\nAddiert „Größe“-Werte."}},
      {"", "Vormerken.", "", {run("git add -A")}, {""}},
      {"", "Committen.", "", {run("git commit -m \"fix add; lint\"")}, {"[main def456] fix add; lint"}},
      {"", "Fertig.", "Ruff ist sauber, die Tests laufen, und alles ist committet.", {}, {}},
  };
  return t;
}

// The turn's text: as the model writes it (raw) or as the template re-renders the echoed turn (canonical).
std::string turn_text(const Turn& t, bool canonical, bool with_reasoning) {
  std::string s = "<think>\n" + (with_reasoning ? t.reasoning : std::string()) + "\n</think>\n\n" + t.content;
  for (size_t i = 0; i < t.calls.size(); ++i) {
    std::string a = py_dump(t.calls[i].second);
    if (t.raw && !canonical) {
      replace_all(a, "\\n", "\n");
      replace_all(a, "\\t", "\t");
    }
    s += (i > 0 || !t.content.empty() ? "\n" : "") + std::string("<tool_call>\n{\"name\": \"") + t.calls[i].first +
         "\", \"arguments\": " + a + "}\n</tool_call>";
  }
  return s;
}

std::string first_prompt(const std::string& sentence, const std::string& user) {
  std::string s = "<|im_start|>system\n# Reasoning effort\n\n" + sentence +
                  "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                  "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
  for (const ojson& tool : kTools) s += "\n" + py_dump(tool);
  s += "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
       "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": "
       "<args-json-object>}\n</tool_call><|im_end|>\n";
  return s + "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
}

void show_diff(const std::string& got, const std::string& want) {
  size_t i = 0;
  while (i < got.size() && i < want.size() && got[i] == want[i]) ++i;
  const size_t from = i > 80 ? i - 80 : 0;
  std::fprintf(stderr, "  first difference at byte %zu of %zu (want %zu)\n  got:  %s\n  want: %s\n", i, got.size(),
               want.size(), json(got.substr(from, 240)).dump().c_str(), json(want.substr(from, 240)).dump().c_str());
}

// Runs the session; returns the last request's prompt.
std::string run_session(const std::string& dir, bool echo_reasoning, bool top_level_effort) {
  Fixture f(dir);
  const std::vector<Turn> turns = session();
  ojson messages = ojson::array();
  std::string prev, expected;
  size_t raw_calls = 0, parallel = 0;
  for (size_t k = 0; k < turns.size(); ++k) {
    const Turn& t = turns[k];
    // The expected prompt, from the previous one.
    if (k == 0) {
      expected = first_prompt(kMedium, t.user);
    } else {
      const Turn& p = turns[k - 1];
      expected = prev + turn_text(p, true, echo_reasoning) + "<|im_end|>\n";
      if (!p.results.empty()) {
        expected += "<|im_start|>user";
        for (const std::string& r : p.results) expected += "\n<tool_response>\n" + r + "\n</tool_response>";
        expected += "<|im_end|>\n";
      }
      if (!t.user.empty()) {
        // A new query: every earlier turn's think block goes (they are all before the last query now).
        for (size_t j = 0; j < k; ++j)
          replace_all(expected, "<think>\n" + (echo_reasoning ? turns[j].reasoning : std::string()) + "\n</think>\n\n", "");
        expected += "<|im_start|>user\n" + t.user + "<|im_end|>\n";
      }
      expected += "<|im_start|>assistant\n";
    }
    if (!t.user.empty()) messages.push_back({{"role", "user"}, {"content", t.user}});
    ojson body = {{"model", "kolibri"}, {"messages", messages}, {"tools", kTools}, {"stream", k % 2 == 1},
                  {"temperature", 1.0},  {"top_p", 0.97},        {"top_k", 128}};
    if (top_level_effort) body["reasoning_effort"] = "medium";
    else body["chat_template_kwargs"] = {{"reasoning_effort", "medium"}};
    f.script(turn_text(t, false, true));
    const Reply r = f.post(body);
    const std::string& prompt = f.tmpl.last;
    if (prompt != expected) {
      std::fprintf(stderr, "turn %zu (echo reasoning %d): the prompt is not the template's render\n", k, echo_reasoning);
      show_diff(prompt, expected);
      CHECK(false);
    }
    // The response is the turn.
    CHECK_EQ(r.finish, std::string(t.calls.empty() ? "stop" : "tool_calls"));
    CHECK_EQ(r.reasoning, t.reasoning);
    CHECK_EQ(r.content, t.content);
    CHECK_EQ(r.calls.size(), t.calls.size());
    for (size_t i = 0; i < t.calls.size(); ++i) {
      CHECK_EQ(r.calls[i].at("function").at("name").get<std::string>(), t.calls[i].first);
      CHECK_EQ(ojson::parse(r.calls[i].at("function").at("arguments").get<std::string>()), t.calls[i].second);
    }
    raw_calls += t.raw ? t.calls.size() : 0;
    parallel += t.calls.size() > 1 ? 1 : 0;
    // The client's history: the response echoed (reasoning_content too, or not), then the results.
    ojson echoed = {{"role", "assistant"}, {"content", ojson::parse(r.content_field.dump())}};
    if (echo_reasoning) echoed["reasoning_content"] = r.reasoning;
    if (!r.calls.empty()) {
      echoed["tool_calls"] = ojson::array();
      for (const ojson& c : r.calls)
        echoed["tool_calls"].push_back({{"id", c.at("id")}, {"type", "function"}, {"function", c.at("function")}});
    }
    messages.push_back(echoed);
    for (size_t i = 0; i < t.results.size(); ++i)
      messages.push_back({{"role", "tool"}, {"tool_call_id", r.calls[i].at("id")}, {"content", t.results[i]}});
    prev = prompt;
  }
  CHECK_EQ(raw_calls, 3U);
  CHECK_EQ(parallel, 2U);
  // The last prompt: every tool result, and think blocks only on the turns after the second query.
  size_t results = 0;
  for (const Turn& t : turns) results += t.results.size();
  CHECK_EQ(count(prev, "<tool_response>"), results);
  CHECK_EQ(count(prev, "<think>"), 9U);
  CHECK_EQ(count(prev, "<|im_start|>assistant\n"), turns.size());
  std::printf("session (echo reasoning %d, %s): 20 turns, every prompt the template's render; %zu tool results, %zu "
              "raw-newline calls repaired, %zu parallel turns; last prompt %zu bytes\n",
              echo_reasoning, top_level_effort ? "top-level reasoning_effort" : "chat_template_kwargs", results, raw_calls,
              parallel, prev.size());
  return prev;
}

void case_sessions(const std::string& dir) {
  const std::string with = run_session(dir, true, true);
  const std::string without = run_session(dir, false, false);
  // Without reasoning echoed only the think blocks of the turns after the last query differ: empty.
  std::string want = with;
  for (const Turn& t : session()) replace_all(want, "<think>\n" + t.reasoning + "\n</think>", "<think>\n\n</think>");
  CHECK(want == without);
  CHECK_EQ(count(without, "<think>\n\n</think>\n\n"), 9U);
}

ojson simple(const ojson& extra) {
  ojson b = {{"model", "kolibri"}, {"messages", ojson::array({{{"role", "user"}, {"content", "Hallo?"}}})}};
  for (auto it = extra.begin(); it != extra.end(); ++it) b[it.key()] = it.value();
  return b;
}

void case_switches(const std::string& dir) {
  Fixture f(dir);
  struct Row {
    ojson extra;
    std::string sentence;
    bool off;
  };
  const std::vector<Row> rows = {
      {ojson::object(), kHigh, false},
      {{{"reasoning_effort", "none"}}, kNone, true},
      {{{"reasoning_effort", "minimal"}}, kLow, false},
      {{{"reasoning_effort", "low"}}, kLow, false},
      {{{"reasoning_effort", "medium"}}, kMedium, false},
      {{{"reasoning_effort", "high"}}, kHigh, false},
      {{{"reasoning_effort", "xhigh"}}, kHigh, false},
      {{{"reasoning_effort", "max"}}, kHigh, false},
      {{{"chat_template_kwargs", {{"reasoning_effort", "low"}}}}, kLow, false},
      {{{"chat_template_kwargs", {{"enable_thinking", false}}}}, kNone, true},
      {{{"chat_template_kwargs", {{"reasoning_effort", "none"}}}}, kNone, true},
      // An explicit kwarg wins over the top-level field.
      {{{"reasoning_effort", "none"}, {"chat_template_kwargs", {{"reasoning_effort", "high"}}}}, kHigh, false},
  };
  for (const Row& row : rows) {
    f.script(row.off ? "Hallo." : "<think>\nkurz\n</think>\n\nHallo.");
    const Reply r = f.post(simple(row.extra));
    const std::string& p = f.tmpl.last;
    const std::string want = "<|im_start|>system\n# Reasoning effort\n\n" + row.sentence +
                             "<|im_end|>\n<|im_start|>user\nHallo?<|im_end|>\n<|im_start|>assistant\n" +
                             (row.off ? "<think>\n\n</think>\n\n" : "");
    if (p != want) {
      std::fprintf(stderr, "switch %s\n", row.extra.dump().c_str());
      show_diff(p, want);
      CHECK(false);
    }
    CHECK_EQ(r.finish, std::string("stop"));
    CHECK_EQ(r.content, std::string("Hallo."));
    CHECK_EQ(r.reasoning, std::string(row.off ? "" : "kurz"));
  }
  for (const char* bad : {"ultra", "NONE"}) {
    std::string text;
    CHECK_EQ(f.post_status(simple({{"reasoning_effort", bad}}), &text), 400);
    CHECK(text.find("reasoning_effort") != std::string::npos);
  }
  std::string text;
  CHECK_EQ(f.post_status(simple({{"reasoning_effort", 3}}), &text), 400);
  std::printf("switches: top-level reasoning_effort (7 levels), chat_template_kwargs reasoning_effort / enable_thinking "
              "false, an explicit kwarg wins, bad values 400\n");
}

void case_budget(const std::string& dir) {
  Fixture f(dir);
  for (bool stream : {false, true}) {
    f.script("<think>\nIch denke sehr lange nach, über viele Schritte hinweg, ohne Ende.", false);
    ojson b = simple({{"max_tokens", 30}, {"stream", stream}, {"tools", kTools}, {"reasoning_effort", "high"}});
    const Reply r = f.post(b);
    CHECK_EQ(r.finish, std::string("length"));
    CHECK_EQ(r.reasoning, std::string("Ich denke sehr lange n"));   // 30 one-character tokens, "<think>\n" the first 8
    CHECK_EQ(r.content, std::string(""));
    CHECK(r.calls.empty());
  }
  std::printf("budget: max_tokens inside the reasoning ends with finish_reason length, the reasoning so far, no content "
              "or calls (body and stream)\n");
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "tests/tokenizer/kolibri";
  case_sessions(dir);
  case_switches(dir);
  case_budget(dir);
  std::printf("kolibri_session_test OK\n");
  return 0;
}
