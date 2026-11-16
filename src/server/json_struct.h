#pragma once
// The server's JSON bodies as plain structs, by static reflection (C++26 P2996 + P3394 annotations, -freflection) over
// nlohmann::json - common/json_reflect.h's rule for config.json, here for what clients send and receive.
//
// One member per key, named as the key. `read` takes a request object's keys into a struct, member by member in
// declaration order, and calls `fail` (which throws) at the first fault with "<key> <what is wrong>":
//   std::optional<bool>                      must be a boolean
//   std::optional<std::string>               must be a string
//   std::optional<uint32_t>, <uint64_t>      must be a non-negative integer (a JSON integer in range; 1.0 is not one)
//   std::optional<double>                    must be a number
//   std::optional<std::vector<std::string>>  must be a string or array of strings (a string is a list of one);
//                                            <key> entries must be strings
//   std::optional<a struct>                  must be an object, its keys read by the same rules
//   nlohmann::json / ordered_json            any value, as sent (null when absent)
// An absent key leaves the member empty; null is a fault unless the member says `null_is_absent`. `Says` replaces the
// text of every fault of the member; `Holds` adds a rule on the value read, refused with its own text (else the
// member's). `write` makes a response object of a struct: an empty std::optional is left out (null with `nullable`),
// std::nullptr_t is null, a vector an array, a struct an object. The object's keys come out sorted - nlohmann::json's
// order, the bytes every body had before these structs.
#include <cstdint>
#include <limits>
#include <meta>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <nlohmann/json.hpp>

namespace server::json_struct {

struct Says {
  const char* text;
  consteval Says(std::string_view t) : text(std::define_static_string(t)) {}
};
template <class V>
struct Holds {
  bool (*ok)(const V&);
  const char* text;   // "" - the member's text
  consteval Holds(bool (*f)(const V&), std::string_view t = "") : ok(f), text(std::define_static_string(t)) {}
};
struct NullIsAbsent {};
inline constexpr NullIsAbsent null_is_absent;
struct Nullable {};
inline constexpr Nullable nullable;

template <class T>
inline constexpr bool is_optional = false;
template <class T>
inline constexpr bool is_optional<std::optional<T>> = true;
template <class T>
inline constexpr bool is_vector = false;
template <class T>
inline constexpr bool is_vector<std::vector<T>> = true;
template <class T>
inline constexpr bool is_json = std::is_same_v<T, nlohmann::json> || std::is_same_v<T, nlohmann::ordered_json>;

template <class T>
consteval auto members() {
  return std::define_static_array(std::meta::nonstatic_data_members_of(^^T, std::meta::access_context::unchecked()));
}
template <class A>
consteval std::optional<A> annotation(std::meta::info m) {
  for (std::meta::info a : std::meta::annotations_of_with_type(m, ^^A)) return std::meta::extract<A>(a);
  return std::nullopt;
}

template <class T, class J, class Fail>
void read(const J& o, T& out, const Fail& fail) {
  template for (constexpr std::meta::info m : members<T>()) {
    using M = typename[:std::meta::type_of(m):];
    const std::string key(std::meta::identifier_of(m));
    const auto it = o.find(key);
    if constexpr (is_json<M>) {
      if (it != o.end()) out.[:m:] = M(*it);
    } else {
      static_assert(is_optional<M>, "read: a member is a std::optional or a JSON value");
      using V = typename M::value_type;
      constexpr std::optional<Says> says = annotation<Says>(m);
      constexpr std::optional<Holds<V>> holds = annotation<Holds<V>>(m);
      constexpr bool null_absent = annotation<NullIsAbsent>(m).has_value();
      if (it != o.end() && !(null_absent && it->is_null())) {
        const J& v = *it;
        const char* type_text = "must be an object";
        bool typed = false;
        V got{};
        if constexpr (std::is_same_v<V, bool>) {
          type_text = "must be a boolean";
          if ((typed = v.is_boolean())) got = v.template get<bool>();
        } else if constexpr (std::is_same_v<V, std::string>) {
          type_text = "must be a string";
          if ((typed = v.is_string())) got = v.template get<std::string>();
        } else if constexpr (std::is_same_v<V, uint32_t> || std::is_same_v<V, uint64_t>) {
          type_text = "must be a non-negative integer";
          typed = v.is_number_unsigned() ? v.template get<uint64_t>() <= std::numeric_limits<V>::max()
                                         : v.is_number_integer() && v.template get<int64_t>() >= 0 &&
                                               uint64_t(v.template get<int64_t>()) <= std::numeric_limits<V>::max();
          if (typed) got = V(v.template get<uint64_t>());
        } else if constexpr (std::is_same_v<V, double>) {
          type_text = "must be a number";
          if ((typed = v.is_number())) got = v.template get<double>();
        } else if constexpr (std::is_same_v<V, std::vector<std::string>>) {
          type_text = "must be a string or array of strings";
          if (v.is_string()) {
            typed = true;
            got.push_back(v.template get<std::string>());
          } else if ((typed = v.is_array())) {
            for (const J& e : v) {
              if (!e.is_string()) fail(key + " entries must be strings");
              got.push_back(e.template get<std::string>());
            }
          }
        } else {
          static_assert(std::is_class_v<V>, "read: an optional holds bool, std::string, uint32_t, uint64_t, double, "
                                            "std::vector<std::string> or a struct");
          if ((typed = v.is_object())) read(v, got, fail);
        }
        const std::string text = says ? says->text : type_text;
        if (!typed) fail(key + " " + text);
        if constexpr (holds.has_value()) {
          if (!holds->ok(got)) fail(key + " " + (*holds->text ? std::string(holds->text) : text));
        }
        out.[:m:] = std::move(got);
      }
    }
  }
}

template <class T>
nlohmann::json write(const T& v);

template <class V>
nlohmann::json write_value(const V& x) {
  if constexpr (is_vector<V>) {
    nlohmann::json a = nlohmann::json::array();
    for (const auto& e : x) a.push_back(write_value(e));
    return a;
  } else if constexpr (std::is_class_v<V> && !std::is_same_v<V, std::string> && !is_json<V>) {
    return write(x);
  } else {
    return x;
  }
}

template <class T>
nlohmann::json write(const T& v) {
  nlohmann::json o = nlohmann::json::object();
  template for (constexpr std::meta::info m : members<T>()) {
    using M = typename[:std::meta::type_of(m):];
    const std::string key(std::meta::identifier_of(m));
    if constexpr (is_optional<M>) {
      if (v.[:m:]) o[key] = write_value(*v.[:m:]);
      else if (annotation<Nullable>(m).has_value()) o[key] = nullptr;
    } else {
      o[key] = write_value(v.[:m:]);
    }
  }
  return o;
}

}  // namespace server::json_struct
