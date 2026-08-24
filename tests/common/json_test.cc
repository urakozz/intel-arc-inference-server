#include <cstdint>
#include "check.h"
#include "common/json.h"

using common::json::Value;
using common::json::parse;

int main() {
  // Object, array, nesting, numbers, bools, null.
  Value v = parse(R"({"a": 1, "b": [true, false, null, 2.5], "c": {"d": "x"}})");
  CHECK(v.is_object());
  CHECK_EQ(v.at("a").num(), 1.0);
  CHECK_EQ(v.at("b").arr().size(), size_t(4));
  CHECK(v.at("b").arr()[0].boolean());
  CHECK(v.at("b").arr()[2].is_null());
  CHECK_NEAR(v.at("b").arr()[3].num(), 2.5, 0.0);
  CHECK_EQ(v.at("c").at("d").str(), std::string("x"));
  CHECK(v.find("missing") == nullptr);

  // String escapes incl. \uXXXX and a surrogate pair (U+1F600 GRINNING FACE).
  // Source stays ASCII: the input spells the two non-ASCII code points with
  // JSON \u escapes, the expectation spells the UTF-8 bytes with \x escapes.
  Value s = parse(R"({"s": "a\"b\\c\nd\u00e9\ud83d\ude00"})");
  CHECK_EQ(s.at("s").str(), std::string("a\"b\\c\nd\xC3\xA9\xF0\x9F\x98\x80"));

  // Large exact integer (safetensors offsets): 2^40 + 5 is exact in double.
  Value n = parse(R"({"off": 1099511627781})");
  CHECK_EQ(uint64_t(n.at("off").num()), uint64_t(1099511627781ULL));

  // The shapes that actually occur: a safetensors header entry.
  Value h = parse(R"({"model.layers.0.mlp.up_proj.qweight":
      {"dtype": "I32", "shape": [640, 17408], "data_offsets": [0, 44564480]}})");
  const Value& e = h.at("model.layers.0.mlp.up_proj.qweight");
  CHECK_EQ(e.at("dtype").str(), std::string("I32"));
  CHECK_EQ(uint64_t(e.at("shape").arr()[1].num()), uint64_t(17408));

  // Errors: position is reported; wrong-kind access throws.
  bool threw = false;
  try { parse("{\"a\": [1, }"); } catch (const std::runtime_error&) { threw = true; }
  CHECK(threw);
  threw = false;
  try { (void)v.at("a").str(); } catch (const std::runtime_error&) { threw = true; }
  CHECK(threw);
  std::puts("json_test OK");
  return 0;
}
