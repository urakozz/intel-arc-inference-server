#pragma once
// A config.json object held to a struct of wanted values by static reflection (C++26 P2996, -freflection).
//
// The struct mirrors the JSON: one member per key, named as the key, in the order the checks run. A member without an
// initializer is a required key; one with an initializer may be absent (the class default the engine is written for,
// so the initializer is also the wanted value unless the caller's `want` says otherwise). `expect` walks the members
// in declaration order and calls `fail` (which throws) at the first key that is absent when required, of the wrong
// type, or not the wanted value, naming the key and both values:
//   uint32_t      a non-negative integer    'k' is 7, the descriptor says 8
//   double        a number                  'k' is 4.000000, the descriptor says 5.000000
//   bool                                    'k' is true, the descriptor needs false
//   std::string                             'k' is 'gelu', the descriptor needs 'silu'
//   a struct      a nested object, held member by member
// `Show` names the value found when it has the wrong type ('k' is 'x', not a non-negative integer); without it the
// refusal reads 'k' is missing or not a non-negative integer.
#include <cmath>
#include <cstdint>
#include <meta>
#include <string>
#include <type_traits>

#include "common/json.h"

namespace common::json {

// A value as a refusal names it: strings in single quotes, numbers as std::to_string writes them.
inline std::string show(const Value& v) {
  if (v.is_null()) return "null";
  if (v.is_bool()) return v.boolean() ? "true" : "false";
  if (v.is_string()) return "'" + v.str() + "'";
  if (v.is_number()) return std::to_string(v.num());
  return v.is_array() ? "[...]" : "{...}";
}

template <bool Show = false, class T, class Fail>
void expect(const Value& o, const T& want, const Fail& fail) {
  template for (constexpr std::meta::info m : std::define_static_array(
                    std::meta::nonstatic_data_members_of(^^T, std::meta::access_context::unchecked()))) {
    using M = typename[:std::meta::type_of(m):];
    constexpr bool may_be_absent = std::meta::has_default_member_initializer(m);
    const std::string name(std::meta::identifier_of(m));
    const std::string k = "'" + name + "'";
    const Value* v = o.find(name);
    const M& w = want.[:m:];
    const std::string found = v ? show(*v) : std::string("absent");
    if (!v && may_be_absent) {
      // the class default: nothing to hold
    } else if constexpr (std::is_same_v<M, uint32_t>) {
      if (!v || !v->is_number() || v->num() < 0 || v->num() != std::floor(v->num()) || v->num() > 4294967295.0)
        fail(Show ? k + " is " + found + ", not a non-negative integer" : k + " is missing or not a non-negative integer");
      else if (uint32_t(v->num()) != w)
        fail(k + " is " + std::to_string(uint32_t(v->num())) + ", the descriptor says " + std::to_string(w));
    } else if constexpr (std::is_same_v<M, double>) {
      if (!v || !v->is_number() || v->num() != w) fail(k + " is " + found + ", the descriptor says " + std::to_string(w));
    } else if constexpr (std::is_same_v<M, bool>) {
      if (!v || !v->is_bool())
        fail(Show ? k + " is " + found + ", not a boolean" : k + " is not a boolean");
      else if (v->boolean() != w)
        fail(k + " is " + (v->boolean() ? "true" : "false") + ", the descriptor needs " + (w ? "true" : "false"));
    } else if constexpr (std::is_same_v<M, std::string>) {
      if (!v || !v->is_string() || v->str() != w)
        fail(k + " is " + (Show ? found : v && v->is_string() ? "'" + v->str() + "'" : std::string("absent")) +
             ", the descriptor needs '" + w + "'");
    } else {
      static_assert(std::is_class_v<M>, "expect: a member is uint32_t, double, bool, std::string or a struct");
      if (!v || !v->is_object())
        fail(k + " is missing or not an object");
      else
        expect<Show>(*v, w, fail);
    }
  }
}

}  // namespace common::json
