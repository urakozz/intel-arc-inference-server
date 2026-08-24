#pragma once
// Minimal recursive-descent JSON parser for machine-written inputs
// (safetensors headers, HF config/index files). Deliberately in-tree: the
// project owns its dependencies (README goal 3), and the grammar subset
// below covers everything those producers emit.
//   Supported: objects, arrays, strings (escapes incl. \uXXXX with surrogate
//   pairs -> UTF-8), numbers via strtod (offsets < 2^53 are exact), true/
//   false/null. Not supported: comments, trailing commas, NaN/Infinity;
//   duplicate keys resolve to the last occurrence.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace common::json {

class Value {
 public:
  enum class Kind { Null, Bool, Number, String, Array, Object };
  using Array = std::vector<Value>;
  using Object = std::map<std::string, Value>;

  Value() : kind_(Kind::Null) {}
  static Value make_bool(bool b) { Value v; v.kind_ = Kind::Bool; v.bool_ = b; return v; }
  static Value make_num(double d) { Value v; v.kind_ = Kind::Number; v.num_ = d; return v; }
  static Value make_str(std::string s) { Value v; v.kind_ = Kind::String; v.str_ = std::move(s); return v; }
  static Value make_arr(Array a) { Value v; v.kind_ = Kind::Array; v.arr_ = std::move(a); return v; }
  static Value make_obj(Object o) { Value v; v.kind_ = Kind::Object; v.obj_ = std::move(o); return v; }

  bool is_null() const { return kind_ == Kind::Null; }
  bool is_bool() const { return kind_ == Kind::Bool; }
  bool is_number() const { return kind_ == Kind::Number; }
  bool is_string() const { return kind_ == Kind::String; }
  bool is_array() const { return kind_ == Kind::Array; }
  bool is_object() const { return kind_ == Kind::Object; }

  bool boolean() const { require(Kind::Bool); return bool_; }
  double num() const { require(Kind::Number); return num_; }
  const std::string& str() const { require(Kind::String); return str_; }
  const Array& arr() const { require(Kind::Array); return arr_; }
  const Object& obj() const { require(Kind::Object); return obj_; }

  const Value* find(const std::string& key) const {
    require(Kind::Object);
    auto it = obj_.find(key);
    return it == obj_.end() ? nullptr : &it->second;
  }
  const Value& at(const std::string& key) const {
    const Value* v = find(key);
    if (!v) throw std::runtime_error("json: missing key '" + key + "'");
    return *v;
  }

 private:
  static const char* kind_name(Kind k) {
    switch (k) {
      case Kind::Null: return "null";
      case Kind::Bool: return "bool";
      case Kind::Number: return "number";
      case Kind::String: return "string";
      case Kind::Array: return "array";
      case Kind::Object: return "object";
    }
    return "?";
  }
  void require(Kind k) const {
    if (kind_ != k)
      throw std::runtime_error(std::string("json: expected ") + kind_name(k) +
                               ", value is " + kind_name(kind_));
  }
  Kind kind_;
  bool bool_ = false;
  double num_ = 0;
  std::string str_;
  Array arr_;
  Object obj_;
};

namespace detail {

class Parser {
 public:
  explicit Parser(std::string_view t) : t_(t) {}
  Value parse_document() {
    Value v = parse_value();
    skip_ws();
    if (p_ != t_.size()) fail("trailing characters");
    return v;
  }

 private:
  [[noreturn]] void fail(const char* what) const {
    size_t end = p_ + 20 < t_.size() ? p_ + 20 : t_.size();
    throw std::runtime_error("json: " + std::string(what) + " at byte " +
                             std::to_string(p_) + " near '" +
                             std::string(t_.substr(p_, end - p_)) + "'");
  }
  void skip_ws() {
    while (p_ < t_.size() && (t_[p_] == ' ' || t_[p_] == '\t' || t_[p_] == '\n' || t_[p_] == '\r')) ++p_;
  }
  char peek() {
    if (p_ >= t_.size()) fail("unexpected end of input");
    return t_[p_];
  }
  char take() { char c = peek(); ++p_; return c; }
  void expect(char c) {
    if (take() != c) { --p_; fail("unexpected character"); }
  }
  bool consume_lit(std::string_view lit) {
    if (t_.substr(p_, lit.size()) == lit) { p_ += lit.size(); return true; }
    return false;
  }

  Value parse_value() {
    skip_ws();
    char c = peek();
    if (c == '{') return parse_object();
    if (c == '[') return parse_array();
    if (c == '"') return Value::make_str(parse_string());
    if (consume_lit("true")) return Value::make_bool(true);
    if (consume_lit("false")) return Value::make_bool(false);
    if (consume_lit("null")) return Value();
    if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
    fail("expected a value");
  }

  Value parse_object() {
    expect('{');
    Value::Object o;
    skip_ws();
    if (peek() == '}') { ++p_; return Value::make_obj(std::move(o)); }
    for (;;) {
      skip_ws();
      std::string key = parse_string();
      skip_ws();
      expect(':');
      o[std::move(key)] = parse_value();
      skip_ws();
      char c = take();
      if (c == '}') break;
      if (c != ',') { --p_; fail("expected ',' or '}'"); }
    }
    return Value::make_obj(std::move(o));
  }

  Value parse_array() {
    expect('[');
    Value::Array a;
    skip_ws();
    if (peek() == ']') { ++p_; return Value::make_arr(std::move(a)); }
    for (;;) {
      a.push_back(parse_value());
      skip_ws();
      char c = take();
      if (c == ']') break;
      if (c != ',') { --p_; fail("expected ',' or ']'"); }
    }
    return Value::make_arr(std::move(a));
  }

  Value parse_number() {
    size_t start = p_;
    if (peek() == '-') ++p_;
    while (p_ < t_.size() && ((t_[p_] >= '0' && t_[p_] <= '9') || t_[p_] == '.' ||
                              t_[p_] == 'e' || t_[p_] == 'E' || t_[p_] == '+' || t_[p_] == '-'))
      ++p_;
    std::string num(t_.substr(start, p_ - start));
    char* end = nullptr;
    double d = std::strtod(num.c_str(), &end);
    if (end != num.c_str() + num.size()) { p_ = start; fail("malformed number"); }
    return Value::make_num(d);
  }

  void append_utf8(std::string& s, uint32_t cp) {
    if (cp < 0x80) {
      s += char(cp);
    } else if (cp < 0x800) {
      s += char(0xC0 | (cp >> 6));
      s += char(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      s += char(0xE0 | (cp >> 12));
      s += char(0x80 | ((cp >> 6) & 0x3F));
      s += char(0x80 | (cp & 0x3F));
    } else {
      s += char(0xF0 | (cp >> 18));
      s += char(0x80 | ((cp >> 12) & 0x3F));
      s += char(0x80 | ((cp >> 6) & 0x3F));
      s += char(0x80 | (cp & 0x3F));
    }
  }
  uint32_t parse_hex4() {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
      char c = take();
      v <<= 4;
      if (c >= '0' && c <= '9') v |= uint32_t(c - '0');
      else if (c >= 'a' && c <= 'f') v |= uint32_t(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') v |= uint32_t(c - 'A' + 10);
      else { --p_; fail("bad \\u escape"); }
    }
    return v;
  }

  std::string parse_string() {
    expect('"');
    std::string s;
    for (;;) {
      char c = take();
      if (c == '"') break;
      if (c != '\\') { s += c; continue; }
      char e = take();
      switch (e) {
        case '"': s += '"'; break;
        case '\\': s += '\\'; break;
        case '/': s += '/'; break;
        case 'b': s += '\b'; break;
        case 'f': s += '\f'; break;
        case 'n': s += '\n'; break;
        case 'r': s += '\r'; break;
        case 't': s += '\t'; break;
        case 'u': {
          uint32_t cp = parse_hex4();
          if (cp >= 0xD800 && cp <= 0xDBFF) {  // high surrogate
            expect('\\');
            expect('u');
            uint32_t lo = parse_hex4();
            if (lo < 0xDC00 || lo > 0xDFFF) fail("unpaired surrogate");
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          }
          append_utf8(s, cp);
          break;
        }
        default: --p_; fail("bad escape");
      }
    }
    return s;
  }

  std::string_view t_;
  size_t p_ = 0;
};

}  // namespace detail

inline Value parse(std::string_view text) { return detail::Parser(text).parse_document(); }

}  // namespace common::json
