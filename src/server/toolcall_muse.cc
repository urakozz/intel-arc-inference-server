#include "server/toolcall_muse.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <utility>

namespace server {
namespace {

using json = nlohmann::json;

const std::string kEom = "<|eom|>";
const std::string kEot = "<|eot|>";
const std::string kStart = "<|start|>";
const std::string kMessage = "<|message|>";
const std::string kAssistant = "assistant";
const std::string kTo = "to=";
const std::string kInvoke = "<atem:invoke";
const std::string kInvokeEnd = "</atem:invoke>";
const std::string kParam = "<atem:parameter";
const std::string kParamEnd = "</atem:parameter>";
const std::string kName = "name=\"";
constexpr size_t kMaxMarker = 11;   // "<|message|>"

// ASCII whitespace as Python's str.strip() sees it (the \x1c-\x1f separators included).
bool is_ws(char c) {
  return c == ' ' || c == '\n' || c == '\t' || c == '\r' || c == '\f' || c == '\v' || (c >= '\x1c' && c <= '\x1f');
}
// [^\S\n]: whitespace but not a newline.
bool is_hws(char c) { return c != '\n' && is_ws(c); }
bool is_rcpt(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '-';
}
bool is_word(char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_'; }

std::string strip(const std::string& s) {
  size_t b = 0, e = s.size();
  while (b < e && is_ws(s[b])) ++b;
  while (e > b && is_ws(s[e - 1])) --e;
  return s.substr(b, e - b);
}

size_t rtrim_end(const std::string& s) {
  size_t e = s.size();
  while (e > 0 && is_ws(s[e - 1])) --e;
  return e;
}

bool at(const std::string& s, size_t pos, const std::string& what) { return s.compare(pos, what.size(), what) == 0; }

// The earliest complete marker at or after `from`: its position and which (0 eom, 1 eot, 2 start, 3 message).
std::pair<size_t, int> find_marker(const std::string& s, size_t from) {
  for (size_t p = s.find("<|", from); p != std::string::npos; p = s.find("<|", p + 1)) {
    if (at(s, p, kEom)) return {p, 0};
    if (at(s, p, kEot)) return {p, 1};
    if (at(s, p, kStart)) return {p, 2};
    if (at(s, p, kMessage)) return {p, 3};
  }
  return {std::string::npos, -1};
}

// The header ending with the <|message|> at `m`: its start (>= lo) and recipient - extended backwards over
// `to=RECIPIENT` (the recipient non-empty), horizontal whitespace, and `<|start|>\s*assistant`: the leftmost start
// of vLLM's _MSG_HEADER_RE that ends at this <|message|>.
std::pair<size_t, std::string> header_start(const std::string& s, size_t m, size_t lo) {
  size_t i = m;
  std::string rcpt;
  size_t j = i;
  while (j > lo && is_rcpt(s[j - 1])) --j;
  if (j < i && j >= lo + kTo.size() && at(s, j - kTo.size(), kTo)) {
    rcpt = s.substr(j, i - j);
    i = j - kTo.size();
  }
  while (i > lo && is_hws(s[i - 1])) --i;
  if (i >= lo + kAssistant.size() && at(s, i - kAssistant.size(), kAssistant)) {
    size_t k = i - kAssistant.size();
    while (k > lo && is_ws(s[k - 1])) --k;
    if (k >= lo + kStart.size() && at(s, k - kStart.size(), kStart)) i = k - kStart.size();
  }
  return {i, rcpt};
}

// Where a still-growing body may be emitted up to: before a tail that is a proper prefix of a marker, a
// `to=RECIPIENT` (or "to" / "t") that may grow into a header, and the horizontal whitespace before either.
size_t safe_end(const std::string& s) {
  const size_t len = s.size();
  size_t k = 0;
  for (size_t n = std::min(len, kMaxMarker - 1); n > 0 && k == 0; --n) {
    for (const std::string* mk : {&kEom, &kEot, &kStart, &kMessage}) {
      if (n < mk->size() && s.compare(len - n, n, *mk, 0, n) == 0) {
        k = n;
        break;
      }
    }
  }
  size_t j = len - k;
  size_t t = j;
  while (t > 0 && is_rcpt(s[t - 1])) --t;
  if (t >= kTo.size() && at(s, t - kTo.size(), kTo)) {
    j = t - kTo.size();
  } else if (k == 0) {
    if (j >= 2 && s[j - 2] == 't' && s[j - 1] == 'o') j -= 2;
    else if (j >= 1 && s[j - 1] == 't') j -= 1;
  }
  while (j > 0 && is_hws(s[j - 1])) --j;
  return j;
}

// ---- schema typing (spec 23 decision 15) ----------------------------------------------------------------------------
void collect_types(const json& schema, std::vector<std::string>& out) {
  if (!schema.is_object()) return;
  if (schema.contains("type")) {
    const json& t = schema.at("type");
    if (t.is_string()) out.push_back(t.get<std::string>());
    if (t.is_array())
      for (const json& x : t)
        if (x.is_string()) out.push_back(x.get<std::string>());
  }
  for (const char* u : {"anyOf", "oneOf"})
    if (schema.contains(u) && schema.at(u).is_array())
      for (const json& choice : schema.at(u)) collect_types(choice, out);
  if (schema.contains("nullable") && schema.at("nullable") == true &&
      std::find(out.begin(), out.end(), "null") == out.end())
    out.push_back("null");
}

std::vector<std::string> schema_types(const std::string& function, const std::string& key, const json& tools) {
  std::vector<std::string> types;
  if (!tools.is_array()) return types;
  for (const json& tool : tools) {
    if (!tool.is_object()) continue;
    const json& fn = tool.contains("function") ? tool.at("function") : tool;
    if (!fn.is_object() || !fn.contains("name") || fn.at("name") != function) continue;
    if (!fn.contains("parameters") || !fn.at("parameters").is_object()) return types;
    const json& params = fn.at("parameters");
    if (!params.contains("properties") || !params.at("properties").is_object()) return types;
    const json& props = params.at("properties");
    if (props.contains(key)) collect_types(props.at(key), types);
    return types;
  }
  return types;
}

// Python's int(s): surrounding whitespace, a sign, digits with single underscores between them.
bool py_int(const std::string& raw, int64_t& v) {
  const std::string s = strip(raw);
  size_t i = 0;
  bool neg = false;
  if (i < s.size() && (s[i] == '+' || s[i] == '-')) neg = s[i++] == '-';
  if (i == s.size()) return false;
  uint64_t acc = 0;
  bool digit_before = false;
  for (; i < s.size(); ++i) {
    const char c = s[i];
    if (c == '_') {
      if (!digit_before || i + 1 == s.size() || s[i + 1] < '0' || s[i + 1] > '9') return false;
      digit_before = false;
      continue;
    }
    if (c < '0' || c > '9') return false;
    if (acc > (uint64_t(std::numeric_limits<int64_t>::max()) - uint64_t(c - '0')) / 10) return false;
    acc = acc * 10 + uint64_t(c - '0');
    digit_before = true;
  }
  v = neg ? -int64_t(acc) : int64_t(acc);
  return true;
}

// transformers' _coerce (RP:42-64) for one type: false when it does not fit.
bool coerce(const std::string& s, const std::string& type, json& out) {
  if (type == "integer") {
    int64_t v = 0;
    if (!py_int(s, v)) return false;
    out = v;
    return true;
  }
  if (type == "number") {
    const std::string t = strip(s);
    if (t.empty() || t.find_first_not_of("0123456789+-.eE") != std::string::npos) return false;
    char* end = nullptr;
    errno = 0;
    const double d = std::strtod(t.c_str(), &end);
    if (end != t.c_str() + t.size() || errno == ERANGE || !std::isfinite(d)) return false;
    if (d == std::floor(d) && t.find('.') == std::string::npos && std::fabs(d) < 9.2e18) out = int64_t(d);
    else out = d;
    return true;
  }
  if (type == "boolean") {
    std::string t = strip(s);
    std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    if (t == "true" || t == "1") out = true;
    else if (t == "false" || t == "0") out = false;
    else return false;
    return true;
  }
  if (type == "null") {
    const std::string t = strip(s);
    if (t != "null" && t != "None") return false;
    out = nullptr;
    return true;
  }
  if (type == "object" || type == "array") {
    json j = json::parse(s, nullptr, false);
    if (j.is_discarded() || (type == "object" ? !j.is_object() : !j.is_array())) return false;
    out = std::move(j);
    return true;
  }
  return false;
}

// ---- ATEM ------------------------------------------------------------------------------------------------------------
// `name="VALUE"` inside the tag that starts at `from` (before its first '>'), preceded by a non-word byte - the regex
// `[^>]*?\bname="([^"]+)"`'s first fitting candidate. `after` is set past the closing quote.
bool tag_name(const std::string& s, size_t from, std::string& name, size_t& after) {
  for (size_t q = from; q < s.size() && s[q] != '>'; ++q) {
    if (!at(s, q, kName) || (q > 0 && is_word(s[q - 1]))) continue;
    const size_t b = q + kName.size();
    const size_t e = s.find('"', b);
    if (e == std::string::npos || e == b) continue;
    name = s.substr(b, e - b);
    after = e + 1;
    return true;
  }
  return false;
}

// One complete `<atem:invoke ...>...</atem:invoke>`: false when it names no function.
bool parse_invoke(const std::string& inv, const json& tools, ToolCall& call) {
  std::string emitted;
  size_t after = 0;
  if (!tag_name(inv, kInvoke.size(), emitted, after)) return false;
  call.name = muse_normalize_name(emitted, tools);
  call.arguments = json::object();
  for (size_t p = inv.find(kParam); p != std::string::npos; p = inv.find(kParam, p + 1)) {
    const size_t q = p + kParam.size();
    if (q < inv.size() && is_word(inv[q])) continue;
    // a name inside the tag, the rest of the tag, then the value up to the first close; a parameter that does not
    // fit is skipped (the regex's search moves on)
    std::string key;
    size_t a = 0;
    if (!tag_name(inv, q, key, a)) continue;
    const size_t gt = inv.find('>', a);
    if (gt == std::string::npos) continue;
    const size_t end = inv.find(kParamEnd, gt + 1);
    if (end == std::string::npos) continue;
    call.arguments[key] = muse_convert_value(inv.substr(gt + 1, end - gt - 1), call.name, key, tools);   // the last wins
    p = end + kParamEnd.size() - 1;
  }
  return true;
}

}  // namespace

std::string muse_normalize_name(const std::string& emitted, const json& tools) {
  if (!tools.is_array() || tools.empty()) return emitted;
  std::vector<std::string> names;
  for (const json& tool : tools) {
    if (!tool.is_object()) continue;
    const json& fn = tool.contains("function") ? tool.at("function") : tool;
    if (fn.is_object() && fn.contains("name") && fn.at("name").is_string()) names.push_back(fn.at("name"));
  }
  if (names.empty() || std::find(names.begin(), names.end(), emitted) != names.end()) return emitted;
  const size_t dot = emitted.find('.');
  if (dot != std::string::npos) {
    const std::string head = emitted.substr(0, dot), tail = emitted.substr(dot + 1);
    if (head == tail && std::find(names.begin(), names.end(), head) != names.end()) return head;
  }
  return emitted;
}

json muse_convert_value(const std::string& raw, const std::string& function, const std::string& key,
                        const json& tools) {
  const std::vector<std::string> types = schema_types(function, key, tools);
  if (types.empty()) {
    json j = json::parse(raw, nullptr, false);
    return j.is_discarded() ? json(raw) : j;
  }
  const bool has_null = std::find(types.begin(), types.end(), "null") != types.end();
  const bool has_string = std::find(types.begin(), types.end(), "string") != types.end();
  const bool only_string =
      has_string && std::all_of(types.begin(), types.end(), [](const std::string& t) { return t == "string" || t == "null"; });
  if (has_null && strip(raw) == "null") return nullptr;
  if (only_string) return raw;
  json j = json::parse(raw, nullptr, false);
  if (!j.is_discarded() && !j.is_string()) return j;
  const std::string s = j.is_discarded() ? strip(raw) : j.get<std::string>();
  for (const std::string& t : types) {
    json out;
    if (coerce(s, t, out)) return out;
  }
  return has_string ? json(raw) : json(s);
}

MuseOutputStream::MuseOutputStream(json tools) : tools_(std::move(tools)), seed_(new_call_seed()) {}

void MuseOutputStream::open_body(const std::string& recipient) {
  mode_ = Mode::Body;
  channel_ = recipient == "self" ? Channel::Reasoning
             : (recipient.empty() || recipient == "user") ? Channel::Content
                                                          : Channel::Tool;
  pending_close_ = 0;
}

// A reasoning / content body's text, in order: leading whitespace skipped, trailing whitespace held (the strip), the
// channel's bodies joined with "\n".
void MuseOutputStream::body_text(std::vector<Delta>& out, const std::string& input) {
  if (channel_ == Channel::Tool) return;
  Block& b = channel_ == Channel::Reasoning ? reasoning_ : content_;
  std::string text = input;
  std::string lead;
  if (!b.started) {
    size_t i = 0;
    while (i < text.size() && is_ws(text[i])) ++i;
    text.erase(0, i);
    if (text.empty()) return;
    b.started = true;
    if (b.any) lead = "\n";
  }
  const size_t e = rtrim_end(text);
  if (e == 0) {
    b.held += text;
    return;
  }
  Delta d;
  d.kind = channel_ == Channel::Reasoning ? Delta::Reasoning : Delta::Content;
  d.text = lead + b.held + text.substr(0, e);
  out.push_back(std::move(d));
  b.held = text.substr(e);
  b.any = true;
}

void MuseOutputStream::end_body() {
  Block& b = channel_ == Channel::Reasoning ? reasoning_ : content_;
  b.held.clear();
  b.started = false;
  pending_close_ = 0;
}

// The complete invokes of a tool body in buf_[0, end) become calls. `body_done`: the body ends at `end` (an invoke
// still open there is dropped - decision 16); otherwise what is handled is erased from buf_, an invoke still open
// kept at buf_'s front.
void MuseOutputStream::tool_calls(std::vector<Delta>& out, size_t end, bool body_done) {
  size_t pos = 0, keep = std::string::npos;
  for (;;) {
    const size_t o = buf_.find(kInvoke, pos);
    if (o == std::string::npos || o >= end) break;
    const size_t q = o + kInvoke.size();
    if (q >= end) {
      if (!body_done) keep = o;
      break;
    }
    if (is_word(buf_[q])) {
      pos = o + 1;
      continue;
    }
    const size_t from = (o == 0 && pending_close_ > q) ? pending_close_ : q;
    const size_t c = buf_.find(kInvokeEnd, from);
    if (c == std::string::npos || c + kInvokeEnd.size() > end) {
      if (!body_done) keep = o;
      break;
    }
    ToolCall call;
    if (parse_invoke(buf_.substr(o, c + kInvokeEnd.size() - o), tools_, call)) {
      call.id = make_call_id(seed_, calls_);
      Delta d;
      d.kind = Delta::Call;
      d.call = std::move(call);
      d.index = calls_++;
      out.push_back(std::move(d));
    }
    pos = c + kInvokeEnd.size();
    pending_close_ = 0;
  }
  if (body_done) return;
  size_t cut = keep;
  if (cut == std::string::npos) cut = std::max(pos, end >= kInvoke.size() ? end - (kInvoke.size() - 1) : size_t(0));
  buf_.erase(0, cut);
  scanned_ = scanned_ > cut ? scanned_ - cut : 0;
  // an invoke open at the front: its close cannot start before here next time
  if (keep != std::string::npos) {
    const size_t q = kInvoke.size();
    pending_close_ = buf_.size() >= kInvokeEnd.size() ? std::max(q, buf_.size() - (kInvokeEnd.size() - 1)) : q;
  }
}

void MuseOutputStream::run(std::vector<Delta>& out, bool final) {
  for (;;) {
    if (mode_ == Mode::Outside) {
      const size_t m = buf_.find(kMessage, scanned_);
      if (m == std::string::npos) {
        scanned_ = buf_.size() >= kMaxMarker ? buf_.size() - (kMaxMarker - 1) : 0;
        if (!final) return;
        if (!framed_) {   // no header at all: the whole output is one content body
          channel_ = Channel::Content;
          body_text(out, buf_);
          end_body();
        }
        buf_.clear();
        scanned_ = 0;
        return;
      }
      const std::pair<size_t, std::string> h = header_start(buf_, m, 0);
      framed_ = true;
      buf_.erase(0, m + kMessage.size());
      scanned_ = 0;
      open_body(h.second);
      continue;
    }
    // A body.
    const std::pair<size_t, int> mk = find_marker(buf_, scanned_);
    if (mk.first != std::string::npos) {
      const size_t end = mk.second == 3 ? header_start(buf_, mk.first, 0).first : mk.first;
      if (channel_ == Channel::Tool) tool_calls(out, end, true);
      else body_text(out, buf_.substr(0, end));
      end_body();
      buf_.erase(0, mk.second <= 1 ? mk.first + kEom.size() : end);
      scanned_ = 0;
      mode_ = Mode::Outside;
      continue;
    }
    scanned_ = buf_.size() >= kMaxMarker ? buf_.size() - (kMaxMarker - 1) : 0;
    if (final) {   // the generation ends inside this body
      if (channel_ == Channel::Tool) tool_calls(out, buf_.size(), true);
      else body_text(out, buf_);
      end_body();
      buf_.clear();
      scanned_ = 0;
      mode_ = Mode::Outside;
      return;
    }
    const size_t safe = safe_end(buf_);
    if (channel_ == Channel::Tool) {
      tool_calls(out, safe, false);
    } else {
      body_text(out, buf_.substr(0, safe));
      buf_.erase(0, safe);
      scanned_ = scanned_ > safe ? scanned_ - safe : 0;
    }
    return;
  }
}

std::vector<Delta> MuseOutputStream::push(const std::string& piece) {
  std::vector<Delta> out;
  buf_ += piece;
  run(out, false);
  return out;
}

std::vector<Delta> MuseOutputStream::finish() {
  std::vector<Delta> out;
  run(out, true);
  return out;
}

MuseParsed parse_muse_output(const std::string& text, const json& tools) {
  MuseOutputStream stream(tools);
  std::vector<Delta> deltas = stream.push(text);
  std::vector<Delta> tail = stream.finish();
  deltas.insert(deltas.end(), std::make_move_iterator(tail.begin()), std::make_move_iterator(tail.end()));
  MuseParsed out;
  for (Delta& d : deltas) {
    if (d.kind == Delta::Reasoning) out.reasoning += d.text;
    else if (d.kind == Delta::Content) out.content += d.text;
    else out.tool_calls.push_back(std::move(d.call));
  }
  return out;
}

}  // namespace server
