#pragma once
// A binary's command-line flags as one struct, by static reflection (C++26 P2996 + P3394 annotations, -freflection):
// one member per flag, named as the flag with '_' for '-' (max_num_seqs is --max-num-seqs), its initializer the
// default, and a Help annotation with the value's name and the --help text. `parse` fills the struct from argv in one
// pass; `help` writes the flag lines of --help from the same members. A member's type says what the flag takes:
//   bool          a switch (no value)
//   std::string   the value as given (a value with a syntax of its own - auto|N, a mode - the caller reads after)
//   uint32_t      a non-negative integer: --n expects a non-negative integer, got 'x' / --n is out of range: 1e10
//   Arg<T>        a std::string or uint32_t, and whether the flag was given
// The parser's own refusals: --x needs a value; unknown option '--y' and unexpected extra argument 'z' (after `usage`);
// no positional argument: a snapshot directory or HF repo id is required (after `usage`).
#include <cstdint>
#include <meta>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace cli::flags {

struct Help {
  const char* value;   // the value's name in --help ("N", "auto|L"); "" for a switch
  const char* text;    // --help's text, lines split by '\n'
  const char* shown;   // the default as --help names it when the initializer does not say it ("the most"); "" none
  consteval Help(std::string_view v, std::string_view t, std::string_view s = "")
      : value(std::define_static_string(v)), text(std::define_static_string(t)), shown(std::define_static_string(s)) {}
};
// --help starts a section (its title line) at the member it annotates.
struct Section {
  const char* title;
  consteval Section(std::string_view t) : title(std::define_static_string(t)) {}
};

template <class T>
struct Arg {
  T value{};
  bool given = false;
};

inline uint32_t parse_u32(const std::string& flag, const std::string& v) {
  if (v.empty() || v.find_first_not_of("0123456789") != std::string::npos || v.size() > 10)
    throw std::runtime_error(flag + " expects a non-negative integer, got '" + v + "'");
  const unsigned long long n = std::stoull(v);
  if (n > 0xFFFFFFFFull) throw std::runtime_error(flag + " is out of range: " + v);
  return uint32_t(n);
}

template <class T>
consteval auto members() {
  return std::define_static_array(std::meta::nonstatic_data_members_of(^^T, std::meta::access_context::unchecked()));
}
template <class A>
consteval std::optional<A> annotation(std::meta::info m) {
  for (std::meta::info a : std::meta::annotations_of_with_type(m, ^^A)) return std::meta::extract<A>(a);
  return std::nullopt;
}
inline std::string flag_name(std::string_view member) {
  std::string f = "--" + std::string(member);
  for (char& c : f)
    if (c == '_') c = '-';
  return f;
}

template <class M>
void take(M& m, const std::string& flag, const std::string& v) {
  if constexpr (std::is_same_v<M, std::string>) {
    m = v;
  } else if constexpr (std::is_same_v<M, uint32_t>) {
    m = parse_u32(flag, v);
  } else {
    take(m.value, flag, v);
    m.given = true;
  }
}

// The default as --help shows it: a string as it is, a number unless 0 (no default to name), nothing for a switch.
template <class M>
std::string shown(const M& m) {
  if constexpr (std::is_same_v<M, std::string>) return m;
  else if constexpr (std::is_same_v<M, uint32_t>) return m ? std::to_string(m) : "";
  else if constexpr (std::is_same_v<M, bool>) return "";
  else return shown(m.value);
}

// The flag lines: "  --flag VALUE" padded to `column`, the text (continuation lines at `column`), "(default X)".
template <class T>
std::string help(size_t column) {
  static const T d{};   // static: GCC 16 flags a local one's strings maybe-uninitialized (a false positive)
  std::string out;
  template for (constexpr std::meta::info m : members<T>()) {
    constexpr std::optional<Help> h = annotation<Help>(m);
    static_assert(h.has_value(), "flags: every member has a Help annotation");
    if constexpr (constexpr std::optional<Section> s = annotation<Section>(m); s.has_value()) out += s->title;
    std::string left = "  " + flag_name(std::meta::identifier_of(m)) + (*h->value ? " " + std::string(h->value) : "");
    left += std::string(left.size() + 2 > column ? 2 : column - left.size(), ' ');
    const std::string def = *h->shown ? std::string(h->shown) : shown(d.[:m:]);
    const std::string text = std::string(h->text) + (def.empty() ? "" : " (default " + def + ")");
    size_t at = 0;
    for (bool first = true; at <= text.size(); first = false) {
      const size_t nl = std::min(text.find('\n', at), text.size());
      out += (first ? left : std::string(column, ' ')) + text.substr(at, nl - at) + "\n";
      at = nl + 1;
    }
  }
  return out;
}

// argv into `out` and the one positional argument into `path`; false when -h / --help asked for the usage.
template <class T>
bool parse(int argc, char** argv, T& out, std::string& path, void (*usage)()) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-h" || a == "--help") return false;
    bool known = false;
    template for (constexpr std::meta::info m : members<T>()) {
      using M = typename[:std::meta::type_of(m):];
      if (!known && a == flag_name(std::meta::identifier_of(m))) {
        known = true;
        if constexpr (std::is_same_v<M, bool>) {
          out.[:m:] = true;
        } else {
          if (++i >= argc) throw std::runtime_error(a + " needs a value");
          take(out.[:m:], a, argv[i]);
        }
      }
    }
    if (known) continue;
    if (!a.empty() && a[0] == '-') {
      usage();
      throw std::runtime_error("unknown option '" + a + "'");
    }
    if (!path.empty()) {
      usage();
      throw std::runtime_error("unexpected extra argument '" + a + "'");
    }
    path = a;
  }
  if (path.empty()) {
    usage();
    throw std::runtime_error("a snapshot directory or HF repo id is required");
  }
  return true;
}

}  // namespace cli::flags
