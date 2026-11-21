#pragma once
// Spec 39: an engine's GPU state described once, on its sizes struct, and walked by static reflection (C++26 P2996 +
// P3394 annotations, -freflection) wherever it was listed by hand - the plan's byte groups, reset's zeroing, the
// snapshot runs' buffers, memory_use.
//
// A SIZES struct (all members, one per allocation, the bytes it allocates) annotates every member with exactly one class:
//   gpu_state::kv / gpu_state::state   per-sequence state (what a session reads across steps), planned in memory_line's
//                                      kv / decode_state; it must also say, once each:
//                                        reset     gpu_state::zero_on_reset, or gpu_state::KeptOnReset("why")
//                                        snapshot  Snapshot<E>{E::x} (repeatable: every run tensor that addresses it),
//                                                  or gpu_state::NotSnapshotted("why")
//   gpu_state::scratch                 no step reads what it has not written first (never snapshotted; a reset may
//                                      still zero it: gpu_state::zero_on_reset)
//   gpu_state::constant                written once at construction (never zeroed, never snapshotted)
// sizes_problem<S, E>() names the first member that breaks a rule ("" when none) and snapshot_problem<E, S...>() the
// first tensor of E not claimed by exactly one member; a static_assert after each struct makes either a build error.
//
// A BUFFERS struct holds the allocations: a member of type Mem, std::unique_ptr<Mem> (null: not allocated) or
// std::array<std::unique_ptr<Mem>, N> / std::unique_ptr<Mem>[N] (N allocations, e.g. one per chunk in flight; its sizes member their sum) is a
// buffer. buffers_problem<Mem, B, S...>() pairs them with the sizes members by name - every buffer has exactly one
// sizes member and every sizes member one buffer - so a buffer cannot sit outside the plan, reset or the snapshot. Any
// other member that holds a Mem (a vector, a map) is refused unless it says gpu_state::Outside("why").
// Kernels still take each buffer's ptr() by name (the captures are unchanged); only the lists move here.
#include <cstddef>
#include <array>
#include <cstdint>
#include <memory>
#include <meta>
#include <string>
#include <string_view>
#include <type_traits>

namespace runtime::gpu_state {

enum class Group : uint32_t { Kv, State, Scratch };   // memory_line's kv, decode_state's state and scratch

struct PerSequence {
  Group group;
};
struct Scratch {};
struct Constant {};
inline constexpr PerSequence kv{Group::Kv}, state{Group::State};
inline constexpr Scratch scratch;
inline constexpr Constant constant;

struct ZeroOnReset {};
inline constexpr ZeroOnReset zero_on_reset;
struct KeptOnReset {
  const char* why;
  consteval KeptOnReset(std::string_view w) : why(std::define_static_string(w)) {}
};
template <class E>
struct Snapshot {
  E tensor;
};
struct NotSnapshotted {
  const char* why;
  consteval NotSnapshotted(std::string_view w) : why(std::define_static_string(w)) {}
};

template <class T>
consteval auto fields() {
  return std::define_static_array(std::meta::nonstatic_data_members_of(^^T, std::meta::access_context::unchecked()));
}
template <class A>
consteval size_t count(std::meta::info m) {
  return std::meta::annotations_of_with_type(m, ^^A).size();
}
consteval bool per_sequence(std::meta::info m) { return count<PerSequence>(m) == 1; }
consteval Group group_of(std::meta::info m) {
  for (std::meta::info a : std::meta::annotations_of_with_type(m, ^^PerSequence))
    return std::meta::extract<PerSequence>(a).group;
  return Group::Scratch;
}
consteval bool zeroed(std::meta::info m) { return count<ZeroOnReset>(m) == 1; }
template <class E>
consteval bool claims(std::meta::info m, E t) {
  for (std::meta::info a : std::meta::annotations_of_with_type(m, ^^Snapshot<E>))
    if (std::meta::extract<Snapshot<E>>(a).tensor == t) return true;
  return false;
}
consteval std::string_view says(std::string_view s) { return std::define_static_string(s); }
consteval std::string num(size_t n) { return n < 10 ? std::string(1, char('0' + n)) : num(n / 10) + char('0' + n % 10); }

// "" when every member of the sizes struct S follows the rules above; else what is wrong with the first that does not.
template <class S, class E>
consteval std::string_view sizes_problem() {
  for (std::meta::info m : fields<S>()) {
    const std::string at = std::string(std::meta::identifier_of(^^S)) + "::" + std::string(std::meta::identifier_of(m));
    if (count<PerSequence>(m) + count<Scratch>(m) + count<Constant>(m) != 1)
      return says(at + " has no GPU state annotation: exactly one of gpu_state::kv, state, scratch, constant");
    const size_t reset = count<ZeroOnReset>(m) + count<KeptOnReset>(m);
    const size_t snap = count<Snapshot<E>>(m), not_snap = count<NotSnapshotted>(m);
    if (!per_sequence(m)) {
      if (count<KeptOnReset>(m) + snap + not_snap != 0 || (count<Constant>(m) == 1 && reset != 0))
        return says(at + " is scratch or constant: it takes no snapshot annotation (and a constant is never zeroed)");
      continue;
    }
    if (reset != 1) return says(at + " is per-sequence state: say what reset does (zero_on_reset or KeptOnReset)");
    if ((snap > 0) == (not_snap == 1) || not_snap > 1)
      return says(at + " is per-sequence state: say what the snapshot does (Snapshot{...} or NotSnapshotted)");
  }
  return "";
}

// "" when every tensor of the snapshot enum E is claimed by exactly one member of S...; else the first that is not.
template <class E, class... S>
consteval std::string_view snapshot_problem() {
  for (std::meta::info e : std::meta::enumerators_of(^^E)) {
    const E t = std::meta::extract<E>(e);
    size_t n = 0;
    (
        [&] {
          for (std::meta::info m : fields<S>()) n += claims(m, t) ? 1 : 0;
        }(),
        ...);
    if (n != 1)
      return says("snapshot tensor " + std::string(std::meta::identifier_of(e)) + " is claimed by " + num(n) + " sizes members (one wanted)");
  }
  return "";
}

template <class Mem>
consteval bool is_owned(std::meta::info m) {
  return std::meta::dealias(std::meta::type_of(m)) == std::meta::dealias(^^std::unique_ptr<Mem>);
}
// A std::array<std::unique_ptr<Mem>, N> or std::unique_ptr<Mem>[N]: one buffer of N allocations (e.g. one per chunk
// in flight), its sizes member their sum.
template <class Mem>
consteval bool is_slots(std::meta::info m) {
  const std::meta::info t = std::meta::dealias(std::meta::type_of(m));
  const std::meta::info e = std::meta::is_array_type(t) ? std::meta::remove_extent(t)
                            : std::meta::has_template_arguments(t) && std::meta::template_of(t) == ^^std::array
                                ? std::meta::template_arguments_of(t)[0]
                                : std::meta::info{};
  return e != std::meta::info{} && std::meta::dealias(e) == std::meta::dealias(^^std::unique_ptr<Mem>);
}
template <class Mem>
consteval bool is_buffer(std::meta::info m) {
  return std::meta::dealias(std::meta::type_of(m)) == std::meta::dealias(^^Mem) || is_owned<Mem>(m) ||
         is_slots<Mem>(m);
}
// Whether type t holds a Mem in its template arguments (a container of allocations: a vector, a map).
template <class Mem>
consteval bool holds_mem(std::meta::info t) {
  t = std::meta::dealias(t);
  if (t == std::meta::dealias(^^Mem)) return true;
  if (std::meta::is_array_type(t)) return holds_mem<Mem>(std::meta::remove_extent(t));
  if (!std::meta::has_template_arguments(t)) return false;
  for (std::meta::info a : std::meta::template_arguments_of(t))
    if (std::meta::is_type(a) && holds_mem<Mem>(a)) return true;
  return false;
}
// A buffers struct's (or an engine's) member that holds allocations no sizes member describes, and why (e.g. a map of
// slabs another formula counts, pinned host memory). Without it such a member is refused: the walks could not see it.
// An engine class is checked with no sizes structs (buffers_problem<Mem, Engine>()): its every allocation lives in a
// paired buffers struct or says why not.
struct Outside {
  const char* why;
  consteval Outside(std::string_view w) : why(std::define_static_string(w)) {}
};
// The member of S... named `name` (a null reflection when none is).
template <class... S>
consteval std::meta::info field_named(std::string_view name) {
  std::meta::info out{};
  (
      [&] {
        for (std::meta::info m : fields<S>())
          if (std::meta::identifier_of(m) == name) out = m;
      }(),
      ...);
  return out;
}
template <class... S>
consteval size_t named([[maybe_unused]] std::string_view name) {   // unused for an empty S...
  size_t n = 0;
  (
      [&] {
        for (std::meta::info m : fields<S>()) n += std::meta::identifier_of(m) == name ? 1 : 0;
      }(),
      ...);
  return n;
}

// "" when B's buffers and the members of S... pair one to one by name; else the first that does not.
template <class Mem, class B, class... S>
consteval std::string_view buffers_problem() {
  const std::string b(std::meta::identifier_of(^^B));
  size_t buffers = 0;
  for (std::meta::info m : fields<B>()) {
    if (count<Outside>(m) > 0) continue;
    if (!is_buffer<Mem>(m)) {
      if (holds_mem<Mem>(std::meta::type_of(m)))
        return says(b + "::" + std::string(std::meta::identifier_of(m)) +
                    " holds allocations the walks cannot see: give each its own member (or a std::array of "
                    "std::unique_ptr), or say gpu_state::Outside(\"why\")");
      continue;
    }
    ++buffers;
    if (named<S...>(std::meta::identifier_of(m)) != 1)
      return says(b + "::" + std::string(std::meta::identifier_of(m)) +
                  " is a buffer without exactly one sizes member of its name: the plan, reset and the snapshot "
                  "cannot see it");
  }
  size_t sized = 0;
  ((sized += fields<S>().size()), ...);
  if (sized != buffers) return says(b + ": a sizes member has no buffer of its name");
  return "";
}

// The allocation behind buffer member M of b (null when an owned one is not allocated; const with a const b). Not for
// an array of allocations (is_slots): each() walks those.
template <class Mem, std::meta::info M, class B>
auto* buffer(B& b) {
  if constexpr (is_owned<Mem>(M))
    return b.[:M:].get();
  else
    return &b.[:M:];
}
// f(the allocation) for each allocated one behind buffer member M of b.
template <class Mem, std::meta::info M, class B, class F>
void each(B& b, F&& f) {
  if constexpr (is_slots<Mem>(M)) {
    for (auto& p : b.[:M:])
      if (p) f(*p);
  } else if constexpr (is_owned<Mem>(M)) {
    if (b.[:M:]) f(*b.[:M:]);
  } else {
    f(b.[:M:]);
  }
}

// --- the walks ------------------------------------------------------------------------------------------------------
// Bytes of a sizes struct: every member, or the members of one group (scratch and constant are Group::Scratch).
template <class S>
size_t bytes(const S& s) {
  size_t n = 0;
  template for (constexpr std::meta::info m : fields<S>()) n += s.[:m:];
  return n;
}
template <class S>
size_t bytes(const S& s, Group g) {
  size_t n = 0;
  template for (constexpr std::meta::info m : fields<S>()) {
    if (group_of(m) == g) n += s.[:m:];
  }
  return n;
}
template <class S>
size_t per_sequence_bytes(const S& s) {
  size_t n = 0;
  template for (constexpr std::meta::info m : fields<S>()) {
    if constexpr (per_sequence(m)) n += s.[:m:];
  }
  return n;
}

// Whether m is a buffer of b sized by a member of S... (a walk over S... leaves b's other buffers alone).
template <class Mem, class... S>
consteval bool sized_by(std::meta::info m) {
  return is_buffer<Mem>(m) && named<S...>(std::meta::identifier_of(m)) == 1;
}

// The allocated bytes of b's buffers sized by S...
template <class Mem, class... S, class B>
size_t buffer_bytes(const B& b) {
  size_t n = 0;
  template for (constexpr std::meta::info m : fields<B>()) {
    if constexpr (sized_by<Mem, S...>(m)) each<Mem, m>(b, [&](const Mem& a) { n += a.size(); });
  }
  return n;
}
// The allocated bytes of b's buffers whose sizes member is in group g.
template <class Mem, class... S, class B>
size_t buffer_bytes(const B& b, Group g) {
  size_t n = 0;
  template for (constexpr std::meta::info m : fields<B>()) {
    if constexpr (sized_by<Mem, S...>(m)) {
      constexpr std::meta::info f = field_named<S...>(std::meta::identifier_of(m));
      if (group_of(f) == g) each<Mem, m>(b, [&](const Mem& a) { n += a.size(); });
    }
  }
  return n;
}

// fill(Mem&) on each allocated buffer of b whose sizes member is zero_on_reset, in b's declaration order.
template <class Mem, class... S, class B, class F>
void zero(B& b, F&& fill) {
  template for (constexpr std::meta::info m : fields<B>()) {
    if constexpr (sized_by<Mem, S...>(m)) {
      constexpr std::meta::info f = field_named<S...>(std::meta::identifier_of(m));
      if constexpr (zeroed(f)) each<Mem, m>(b, fill);
    }
  }
}

// The buffer of b whose sizes member claims snapshot tensor t (null when none does or it is not allocated).
template <class Mem, class E, class... S, class B>
auto* snapshot_buffer(B& b, E t) {
  std::conditional_t<std::is_const_v<B>, const Mem*, Mem*> out = nullptr;
  template for (constexpr std::meta::info m : fields<B>()) {
    if constexpr (sized_by<Mem, S...>(m) && !is_slots<Mem>(m)) {
      constexpr std::meta::info f = field_named<S...>(std::meta::identifier_of(m));
      template for (constexpr std::meta::info e : std::define_static_array(std::meta::enumerators_of(^^E))) {
        if constexpr (claims(f, std::meta::extract<E>(e))) {
          if (t == std::meta::extract<E>(e)) out = buffer<Mem, m>(b);
        }
      }
    }
  }
  return out;
}

}  // namespace runtime::gpu_state
