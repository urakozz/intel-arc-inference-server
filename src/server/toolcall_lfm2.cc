#include "server/toolcall_lfm2.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <utility>

namespace server {
namespace {

using json = nlohmann::json;

const std::string kThinkEnd = "</think>";
const std::string kStart = "<|tool_call_start|>";
const std::string kEnd = "<|tool_call_end|>";

constexpr size_t kMaxBlock = size_t(1) << 20;   // a block is read up to its first 1 MiB
constexpr int kMaxDepth = 64;                    // containers inside an argument
constexpr size_t kEagerBytes = 4096;             // below this a streamed block is re-read at every piece

bool is_ws(int c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }

bool nonws(std::string_view s, size_t from) {
  for (size_t i = from; i < s.size(); ++i)
    if (!is_ws(static_cast<unsigned char>(s[i]))) return true;
  return false;
}

// The text from `at` to the end of `s` is a proper prefix of `tag`.
size_t tag_prefix_suffix(const std::string& s, const std::string& tag) {
  const size_t max = std::min(s.size(), tag.size() - 1);
  for (size_t n = max; n > 0; --n)
    if (s.compare(s.size() - n, n, tag, 0, n) == 0) return n;
  return 0;
}

// --- the literal parser --------------------------------------------------------------------------------------------
//
// Control flow by exception: NeedMore - the answer depends on text that has not arrived (only when the text is not final);
// Invalid - the text is not a call. Every decision is made from characters that are present, or throws NeedMore, so what
// a parse of a prefix decides is what the parse of the whole text decides.
struct NeedMore {};
struct Invalid {};

enum class Ctx { Kwarg, Elem, DictKey, DictVal };

class Reader {
 public:
  Reader(std::string_view s, bool final, PythonicRules& rules) : s_(s), final_(final), r_(rules) {}

  // The byte at i, -1 at the end of the final text; NeedMore at the end of text that may continue.
  int at(size_t i) const {
    if (i < s_.size()) return static_cast<unsigned char>(s_[i]);
    if (!final_) throw NeedMore{};
    return -1;
  }
  size_t ws(size_t i) const {
    for (;; ++i) {
      const int c = at(i);
      if (c < 0 || !is_ws(c)) return i;
    }
  }

  // name := quoted-name | name-char+ ; returns the name and the position after it.
  size_t name(size_t i, std::string& out) {
    const int q = at(i);
    out.clear();
    if (q == '\'' || q == '"') {
      size_t j = i + 1;
      for (;; ++j) {
        const int c = at(j);
        if (c < 0 || is_ws(c)) throw Invalid{};
        if (c == q) break;
        out += static_cast<char>(c);
      }
      if (out.empty()) throw Invalid{};
      ++r_.quoted_name;
      return j + 1;
    }
    size_t j = i;
    while (name_char(at(j))) out += s_[j++];
    if (out.empty()) throw Invalid{};
    return j;
  }

  // call := name '(' [arg (',' arg)* [',']] ')' ; fills name and arguments, returns the position after ')'.
  size_t call(size_t i, std::string& name_out, json& args) {
    i = name(ws(i), name_out);
    i = ws(i);
    if (at(i) != '(') throw Invalid{};
    args = json::object();
    bool any = false;
    i = ws(i + 1);
    if (at(i) == ')') return i + 1;
    for (;;) {
      i = ws(i);
      if (at(i) == '{') {   // a lone positional dict is the arguments object
        if (any) throw Invalid{};
        json v;
        i = value(i, Ctx::Elem, v, /*top=*/false);
        i = ws(i);
        if (at(i) == ',') i = ws(i + 1);
        if (at(i) != ')') throw Invalid{};
        ++r_.positional_dict;
        args = std::move(v);
        return i + 1;
      }
      std::string key;
      i = ws(name(i, key));
      if (at(i) != '=' || at(i + 1) == '=') throw Invalid{};
      i = ws(i + 1);
      json v;
      i = value(i, Ctx::Kwarg, v, /*top=*/true);
      args[key] = std::move(v);
      any = true;
      i = ws(i);
      const int c = at(i);
      if (c == ',') {
        i = ws(i + 1);
        if (at(i) == ')') return i + 1;
        continue;
      }
      if (c == ')') return i + 1;
      throw Invalid{};
    }
  }

 private:
  static bool name_char(int c) {
    return c >= 0 && !is_ws(c) && std::string_view("()[]{},='\"").find(static_cast<char>(c)) == std::string_view::npos;
  }

  // value := string | number | keyword | list | tuple | dict
  size_t value(size_t i, Ctx ctx, json& out, bool top) {
    const int c = at(i);
    if (c == '\'' || c == '"') {
      std::string str;
      i = string(i, ctx, str);
      out = std::move(str);
      return i;
    }
    if (c == '[') return list(i, ']', out);
    if (c == '(') return list(i, ')', out);
    if (c == '{') return dict(i, out);
    if ((c >= '0' && c <= '9') || c == '+' || c == '-' || c == '.') return number(i, out);
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_') return keyword(i, out, top);
    throw Invalid{};
  }

  size_t keyword(size_t i, json& out, bool top) {
    std::string word;
    for (;; ++i) {
      const int c = at(i);
      if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_')) break;
      word += static_cast<char>(c);
    }
    if (word == "True") out = true;
    else if (word == "False") out = false;
    else if (word == "None") out = nullptr;
    else if (word == "true" || word == "false" || word == "null") {
      out = word == "true" ? json(true) : word == "false" ? json(false) : json(nullptr);
      if (top) ++r_.json_literal;   // the template writes JSON literals only inside lists and dicts
    } else {
      throw Invalid{};
    }
    return i;
  }

  size_t number(size_t i, json& out) {
    const size_t start = i;
    auto digits = [&](size_t j) {
      size_t n = 0;
      while (at(j) >= '0' && at(j) <= '9') ++j, ++n;
      return std::make_pair(j, n);
    };
    if (at(i) == '+' || at(i) == '-') ++i;
    auto [j, n_int] = digits(i);
    i = j;
    size_t n_frac = 0;
    bool is_float = false;
    if (at(i) == '.') {
      is_float = true;
      auto [k, n] = digits(i + 1);
      i = k;
      n_frac = n;
    }
    if (n_int + n_frac == 0) throw Invalid{};
    if (at(i) == 'e' || at(i) == 'E') {
      is_float = true;
      size_t k = i + 1;
      if (at(k) == '+' || at(k) == '-') ++k;
      auto [m, n] = digits(k);
      if (n == 0) throw Invalid{};
      i = m;
    }
    const int next = at(i);   // a number is followed by a delimiter, not by more of a word
    if ((next >= '0' && next <= '9') || (next >= 'a' && next <= 'z') || (next >= 'A' && next <= 'Z') || next == '_' ||
        next == '.')
      throw Invalid{};
    const std::string token(s_.substr(start, i - start));
    if (!is_float) {
      errno = 0;
      if (token[0] == '-') {
        const long long v = std::strtoll(token.c_str(), nullptr, 10);
        if (errno == 0) { out = static_cast<int64_t>(v); return i; }
      } else {
        const unsigned long long v = std::strtoull(token.c_str(), nullptr, 10);
        if (errno == 0) {
          if (v <= static_cast<unsigned long long>(INT64_MAX)) out = static_cast<int64_t>(v);
          else out = static_cast<uint64_t>(v);
          return i;
        }
      }
    }
    const double d = std::strtod(token.c_str(), nullptr);
    if (!std::isfinite(d)) throw Invalid{};
    out = d;
    return i;
  }

  // A string; `ctx` says what may follow its closing quote and which escapes are read: a keyword argument's value
  // (the template's writer) decodes \\ , its own quote, \n and \r; inside a container JSON's set and its own quote.
  // Every other backslash is kept. A quote that is not followed by what the grammar allows after the string is part of it.
  size_t string(size_t i, Ctx ctx, std::string& out) {
    const int q = at(i);
    const bool triple = at(i + 1) == q && at(i + 2) == q;
    size_t j = i + (triple ? 3 : 1);
    out.clear();
    for (;;) {
      const int c = at(j);
      if (c < 0) throw Invalid{};   // unterminated
      if (c == '\\') {
        const int n = at(j + 1);
        if (n < 0) { out += '\\'; ++j; continue; }
        if (ctx == Ctx::Kwarg) {
          if (n == '\\') { out += '\\'; j += 2; continue; }
          if (n == q) { out += static_cast<char>(q); j += 2; continue; }
          if (n == 'n') { out += '\n'; j += 2; continue; }
          if (n == 'r') { out += '\r'; j += 2; continue; }
        } else {
          size_t used = 2;
          if (json_escape(j, n, q, out, used)) { j += used; continue; }
        }
        out += '\\';
        ++j;
        continue;
      }
      if (c == q && (!triple || (at(j + 1) == q && at(j + 2) == q))) {
        const size_t len = triple ? 3 : 1;
        if (closes(j + len, ctx)) return j + len;
        ++r_.quote_recovered;   // part of the string
        out += static_cast<char>(q);
        ++j;
        continue;
      }
      out += static_cast<char>(c);
      ++j;
    }
  }

  // The escapes of a string inside a list or dict: JSON's, and the string's own quote. j is the backslash, n the byte after.
  bool json_escape(size_t j, int n, int q, std::string& out, size_t& used) const {
    switch (n) {
      case '"': out += '"'; return true;
      case '\\': out += '\\'; return true;
      case '/': out += '/'; return true;
      case 'b': out += '\b'; return true;
      case 'f': out += '\f'; return true;
      case 'n': out += '\n'; return true;
      case 'r': out += '\r'; return true;
      case 't': out += '\t'; return true;
      case 'u': {
        uint32_t cp = 0;
        if (!hex4(j + 2, cp)) return false;
        used = 6;
        if (cp >= 0xD800 && cp <= 0xDBFF) {   // a high surrogate wants its low one
          uint32_t lo = 0;
          if (at(j + 6) == '\\' && at(j + 7) == 'u' && hex4(j + 8, lo) && lo >= 0xDC00 && lo <= 0xDFFF) {
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            used = 12;
          } else {
            cp = 0xFFFD;
          }
        } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
          cp = 0xFFFD;
        }
        utf8(cp, out);
        return true;
      }
      default:
        if (n == q) { out += static_cast<char>(q); return true; }
        return false;
    }
  }

  bool hex4(size_t i, uint32_t& cp) const {
    cp = 0;
    for (size_t k = 0; k < 4; ++k) {
      const int c = at(i + k);
      uint32_t d;
      if (c >= '0' && c <= '9') d = static_cast<uint32_t>(c - '0');
      else if (c >= 'a' && c <= 'f') d = static_cast<uint32_t>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') d = static_cast<uint32_t>(c - 'A' + 10);
      else return false;
      cp = cp * 16 + d;
    }
    return true;
  }

  static void utf8(uint32_t cp, std::string& out) {
    if (cp < 0x80) {
      out += static_cast<char>(cp);
    } else if (cp < 0x800) {
      out += static_cast<char>(0xC0 | (cp >> 6));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      out += static_cast<char>(0xE0 | (cp >> 12));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
      out += static_cast<char>(0xF0 | (cp >> 18));
      out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    }
  }

  // Does a string whose candidate closing quote ended at k (the position after it) really end there? What the grammar
  // allows after it: a keyword argument's value is followed by ')' (and then the end of the block, ',' or ']') or by ','
  // and the next argument ('name=', a quoted name, or ')'); a list element by ',' ']' ')'; a dict key by ':'; a dict
  // value by ',' '}'. The end of the block ends any string.
  bool closes(size_t k, Ctx ctx) const {
    k = ws(k);
    const int c = at(k);
    if (c < 0) return true;
    switch (ctx) {
      case Ctx::Kwarg:
        if (c == ')') {
          const int d = at(ws(k + 1));
          return d < 0 || d == ',' || d == ']';
        }
        if (c == ',') {
          const size_t m = ws(k + 1);
          const int d = at(m);
          return d < 0 || d == ')' || key_follows(m);
        }
        return false;
      case Ctx::Elem: return c == ',' || c == ']' || c == ')';
      case Ctx::DictKey: return c == ':';
      case Ctx::DictVal: return c == ',' || c == '}';
    }
    return false;
  }

  // At k: `name=` or `'name'=` (not `==`).
  bool key_follows(size_t k) const {
    const int q = at(k);
    const bool quoted = q == '\'' || q == '"';
    if (quoted) ++k;
    const size_t n0 = k;
    while (name_char(at(k))) ++k;
    if (k == n0) return false;
    if (quoted) {
      if (at(k) != q) return false;
      ++k;
    }
    k = ws(k);
    return at(k) == '=' && at(k + 1) != '=';
  }

  size_t list(size_t i, char close, json& out) {
    if (++depth_ > kMaxDepth) throw Invalid{};
    out = json::array();
    size_t count = 0;
    bool comma = false;
    i = ws(i + 1);
    for (;;) {
      if (at(i) == close) { ++i; break; }
      json v;
      i = ws(value(i, Ctx::Elem, v, false));
      out.push_back(std::move(v));
      ++count;
      const int c = at(i);
      if (c == ',') { comma = true; i = ws(i + 1); continue; }
      if (c == close) { ++i; break; }
      throw Invalid{};
    }
    --depth_;
    if (close == ')' && count == 1 && !comma) out = out[0];   // (1) is 1
    return i;
  }

  size_t dict(size_t i, json& out) {
    if (++depth_ > kMaxDepth) throw Invalid{};
    out = json::object();
    i = ws(i + 1);
    for (;;) {
      if (at(i) == '}') { ++i; break; }
      json k;
      const int kc = at(i);
      if (kc == '[' || kc == '(' || kc == '{') throw Invalid{};
      i = ws(value(i, Ctx::DictKey, k, false));
      std::string key = k.is_string() ? k.get<std::string>() : k.dump();
      if (at(i) != ':') throw Invalid{};
      i = ws(i + 1);
      json v;
      i = ws(value(i, Ctx::DictVal, v, false));
      out[key] = std::move(v);
      const int c = at(i);
      if (c == ',') { i = ws(i + 1); continue; }
      if (c == '}') { ++i; break; }
      throw Invalid{};
    }
    --depth_;
    return i;
  }

  std::string_view s_;
  bool final_;
  PythonicRules& r_;
  int depth_ = 0;
};

// --- a block: '[' call (',' call)* ']' --------------------------------------------------------------------------------

struct BlockState {
  size_t pos = 0;          // everything before it is consumed
  bool opened = false, bracket = false, bare = false, expect_sep = false;
  bool done = false, bad = false;   // ']' read; a call that is not one
  uint32_t calls = 0;
};

enum class Step { Call, More, Done, Bad };

// The next call of the block text `s`. On Call fills `call` and advances `st`; More: wait for text (`st` unchanged);
// Done: no more calls (the block's end or its ']'); Bad: what follows is no call. Rules count only on Call.
Step next_call(std::string_view s, bool final, BlockState& st, ToolCall& call, PythonicRules& rules) {
  if (st.done) return Step::Done;
  if (st.bad) return Step::Bad;
  PythonicRules local;
  Reader rd(s, final, local);
  BlockState t = st;
  try {
    size_t i = t.pos;
    int c;
    if (!t.opened) {
      i = rd.ws(i);
      c = rd.at(i);
      if (c < 0) { st = t; st.pos = i; st.done = true; return Step::Done; }
      if (c == '[') { t.bracket = true; ++i; } else { t.bare = true; }
      t.opened = true;
    } else if (t.expect_sep) {
      i = rd.ws(i);
      c = rd.at(i);
      if (c < 0) { st = t; st.pos = i; st.done = true; return Step::Done; }
      if (c == ']' && t.bracket) { st = t; st.pos = i + 1; st.done = true; return Step::Done; }
      if (c != ',') { st = t; st.bad = true; return Step::Bad; }
      ++i;
      t.expect_sep = false;
    }
    i = rd.ws(i);
    c = rd.at(i);
    if (c < 0) { st = t; st.pos = i; st.done = true; return Step::Done; }
    if (c == ']' && t.bracket) { st = t; st.pos = i + 1; st.done = true; return Step::Done; }
    std::string name;
    json args;
    const size_t end = rd.call(i, name, args);
    if (t.bare && t.calls == 0) ++local.bare_call;
    ++t.calls;
    t.pos = end;
    t.expect_sep = true;
    st = t;
    call.name = std::move(name);
    call.arguments = std::move(args);
    rules += local;
    return Step::Call;
  } catch (const NeedMore&) {
    return Step::More;
  } catch (const Invalid&) {
    st = t;
    st.bad = true;
    return Step::Bad;
  }
}

// A block's text and how far its calls have been read. The same reader serves the whole-text parse and the stream.
struct BlockText {
  std::string text;
  BlockState st;
  Step last = Step::More;

  // Reads every call that can be read now. `final`: no more text will arrive (the end tag, or the end of generation).
  void advance(bool final, std::vector<ToolCall>& calls, PythonicRules& rules) {
    const bool cut = text.size() > kMaxBlock;
    const std::string_view view(text.data(), std::min(text.size(), kMaxBlock));
    for (;;) {
      ToolCall c;
      last = next_call(view, final && !cut, st, c, rules);
      if (last != Step::Call) return;
      calls.push_back(std::move(c));
    }
  }

  // Is there text after the calls that is neither a separator nor the closing ']'?
  bool has_tail() const {
    const std::string_view view(text.data(), std::min(text.size(), kMaxBlock));
    if (last == Step::Done) return nonws(view, st.pos) || text.size() > kMaxBlock;
    return true;   // Bad, or More past the 1 MiB limit
  }
};

}  // namespace

PythonicRules& PythonicRules::operator+=(const PythonicRules& o) {
  quote_recovered += o.quote_recovered;
  quoted_name += o.quoted_name;
  bare_call += o.bare_call;
  json_literal += o.json_literal;
  positional_dict += o.positional_dict;
  dropped_unfinished += o.dropped_unfinished;
  stray_end += o.stray_end;
  return *this;
}

std::optional<std::vector<ToolCall>> parse_pythonic_calls(std::string_view block, bool complete, uint64_t& seed,
                                                          PythonicRules& rules) {
  BlockText b;
  b.text.assign(block.data(), block.size());
  std::vector<ToolCall> calls;
  PythonicRules local;
  b.advance(/*final=*/true, calls, local);
  if (calls.empty()) {
    if (complete) return std::nullopt;
    if (nonws(block, 0)) ++local.dropped_unfinished;
  } else if (b.has_tail()) {
    ++local.dropped_unfinished;
  }
  for (size_t i = 0; i < calls.size(); ++i) calls[i].id = make_call_id(seed, static_cast<uint32_t>(i));
  rules += local;
  return calls;
}

// --- the stream ------------------------------------------------------------------------------------------------------

struct Lfm2OutputStream::BlockReader : BlockText {
  size_t next_attempt = 0;   // above kEagerBytes the block is re-read only when it has grown by an eighth
};

Lfm2OutputStream::Lfm2OutputStream(json tools, bool reasoning_first)
    : state_(reasoning_first ? State::Reasoning : State::Body), seed_(new_call_seed()) {
  (void)tools;   // the literals type the arguments, not the tools' schemas
}

Lfm2OutputStream::~Lfm2OutputStream() = default;

json Lfm2OutputStream::stats() const {
  return {{"rules",
           {{"quote_recovered", rules_.quote_recovered},
            {"quoted_name", rules_.quoted_name},
            {"bare_call", rules_.bare_call},
            {"json_literal", rules_.json_literal},
            {"positional_dict", rules_.positional_dict},
            {"dropped_unfinished", rules_.dropped_unfinished},
            {"stray_end", rules_.stray_end}}}};
}

void Lfm2OutputStream::emit(std::vector<Delta>& out, Delta::Kind kind, std::string text) {
  if (text.empty()) return;
  Delta d;
  d.kind = kind;
  d.text = std::move(text);
  out.push_back(std::move(d));
}

void Lfm2OutputStream::read_calls(std::vector<Delta>& out, bool final) {
  std::vector<ToolCall> calls;
  block_->advance(final, calls, rules_);
  for (ToolCall& c : calls) {
    c.id = make_call_id(seed_, calls_);
    Delta d;
    d.kind = Delta::Call;
    d.call = std::move(c);
    d.index = calls_++;
    out.push_back(std::move(d));
  }
}

// The block is over: `complete` - its end tag arrived; else generation ended inside it.
void Lfm2OutputStream::close_block(std::vector<Delta>& out, bool complete) {
  read_calls(out, /*final=*/true);
  BlockText& b = *block_;
  if (b.st.calls == 0) {
    if (complete) {   // no valid call: the block is content, verbatim with its tags
      emit(out, Delta::Content, kStart + b.text + kEnd);
    } else if (nonws(b.text, 0)) {
      ++rules_.dropped_unfinished;   // a cut block's text is never content
    }
  } else if (b.has_tail()) {
    ++rules_.dropped_unfinished;
  }
  block_.reset();
}

void Lfm2OutputStream::run(std::vector<Delta>& out) {
  for (;;) {
    switch (state_) {
      case State::Reasoning: {
        const size_t i = buf_.find(kThinkEnd);
        if (i != std::string::npos) {
          emit(out, Delta::Reasoning, buf_.substr(0, i));
          buf_.erase(0, i + kThinkEnd.size());
          state_ = State::Body;
          continue;
        }
        const size_t keep = tag_prefix_suffix(buf_, kThinkEnd);
        emit(out, Delta::Reasoning, buf_.substr(0, buf_.size() - keep));
        buf_.erase(0, buf_.size() - keep);
        return;
      }
      case State::Body: {
        const size_t s = buf_.find(kStart), e = buf_.find(kEnd);
        if (e != std::string::npos && (s == std::string::npos || e < s)) {   // a stray end tag
          emit(out, Delta::Content, buf_.substr(0, e));
          buf_.erase(0, e + kEnd.size());
          ++rules_.stray_end;
          continue;
        }
        if (s != std::string::npos) {
          emit(out, Delta::Content, buf_.substr(0, s));
          buf_.erase(0, s + kStart.size());
          block_ = std::make_unique<BlockReader>();
          state_ = State::Call;
          continue;
        }
        const size_t keep = std::max(tag_prefix_suffix(buf_, kStart), tag_prefix_suffix(buf_, kEnd));
        emit(out, Delta::Content, buf_.substr(0, buf_.size() - keep));
        buf_.erase(0, buf_.size() - keep);
        return;
      }
      case State::Call: {
        const size_t e = buf_.find(kEnd);
        if (e != std::string::npos) {
          block_->text.append(buf_, 0, e);
          buf_.erase(0, e + kEnd.size());
          close_block(out, /*complete=*/true);
          state_ = State::Body;
          continue;
        }
        const size_t keep = tag_prefix_suffix(buf_, kEnd);
        const size_t before = block_->text.size();
        block_->text.append(buf_, 0, buf_.size() - keep);
        buf_.erase(0, buf_.size() - keep);
        // Read the calls that have closed: at every piece while the block is small, then each time it has grown by an
        // eighth (re-reading a long call from its start at every piece would be quadratic). Late is harmless: the block's
        // end reads whatever is left, and the deltas do not depend on when a call was read.
        if (block_->text.size() != before &&
            (block_->text.size() < kEagerBytes || block_->text.size() >= block_->next_attempt)) {
          read_calls(out, /*final=*/false);
          if (block_->last == Step::More) {
            const size_t n = block_->text.size();
            block_->next_attempt = n + std::max<size_t>(n / 8, 512);
          }
        }
        return;
      }
    }
  }
}

std::vector<Delta> Lfm2OutputStream::push(const std::string& piece) {
  std::vector<Delta> out;
  buf_ += piece;
  run(out);
  return out;
}

std::vector<Delta> Lfm2OutputStream::finish() {
  std::vector<Delta> out;
  switch (state_) {
    case State::Reasoning:
      emit(out, Delta::Reasoning, std::move(buf_));
      break;
    case State::Body:
      emit(out, Delta::Content, std::move(buf_));
      break;
    case State::Call:
      block_->text += buf_;   // a partial end tag is text of the unfinished tail
      close_block(out, /*complete=*/false);
      break;
  }
  buf_.clear();
  return out;
}

ParsedOutput parse_output_lfm2(const std::string& text, const json& tools) {
  Lfm2OutputStream stream(tools);
  std::vector<Delta> deltas = stream.push(text);
  std::vector<Delta> tail = stream.finish();
  deltas.insert(deltas.end(), std::make_move_iterator(tail.begin()), std::make_move_iterator(tail.end()));
  ParsedOutput out;
  for (Delta& d : deltas) {
    if (d.kind == Delta::Reasoning) out.reasoning += d.text;
    else if (d.kind == Delta::Content) out.content += d.text;
    else out.tool_calls.push_back(std::move(d.call));
  }
  return out;
}

}  // namespace server
