#pragma once
// b70-serve's reconfigure in place (POST /v1/control/reconfigure, server::Options::reconfigure): the settings that
// change only how the loaded model runs, switched between requests without a reload; every other flag is refused as
// needing a restart. A key is a flag of `Flags` (cli/flags.h) without "--", its value a string or a non-negative integer:
//   spec           the proposer the server started with (mtp, dflash; auto when it started
//                  with auto) or off - the MTP head / the DFlash drafter stays loaded while off
//   spec-max       that proposer's K, 1 up to the K it started with (kept while off)
//   host-experts   auto|gpu|cpu (Qwen3.8-Flash-Next, where the CPU route was built: started with auto or cpu)
// The server's speculative options for "on" are the ones it started with (their K set to the current one); "off" is
// server::Options' defaults - what b70-serve sets for --spec off.
#include <cstdint>
#include <functional>
#include <memory>
#include <meta>
#include <string>

#include <nlohmann/json.hpp>

#include "cli/flags.h"
#include "server/server.h"

namespace cli {

// What a b70-serve switches in place, as it started.
struct InPlace {
  std::string spec = "off";   // the proposer started (resolved: mtp, dflash), "off" none
  bool spec_auto = false;     // it came from --spec auto
  uint32_t spec_max = 0;      // its K at startup
  // The adapter: the started proposer at K, or none (on == false).
  std::function<void(uint32_t k, bool on)> set_proposer;
  // --host-experts as started; set_host_experts empty: not switchable, host_experts_why says why.
  std::string host_experts = "auto";
  std::string host_experts_why;
  std::function<void(const std::string&)> set_host_experts;
};

// Why a flag that is not switched in place needs a restart.
inline std::string restart_reason(const std::string& key) {
  if (key == "kv-cache" || key == "max-len" || key == "max-num-seqs" || key == "prefix-cache-gb")
    return "needs a restart: it changes the memory plan";
  if (key == "parallel" || key == "pipeline-split") return "needs a restart: it decides how the model sits on the cards";
  if (key == "prefill-backend")
    return "needs a restart: the prompt pass's buffers (and with l0-int8 the KV cache's form) are planned at startup";
  if (key == "draft-model" || key == "layers")
    return "needs a restart: it changes what is loaded";
  return "needs a restart: it is read at startup only";
}

// The speculative fields of server::Options (the ones server::Reconfigure may change).
inline void copy_spec(server::Options& to, const server::Options& from) {
  to.spec_lookup = from.spec_lookup;
  to.lookup_min_match = from.lookup_min_match;
  to.lookup_history = from.lookup_history;
  to.lookup_adaptive = from.lookup_adaptive;
  to.spec_dflash_lookup = from.spec_dflash_lookup;
  to.spec_mtp_lookup = from.spec_mtp_lookup;
  to.mtp_auto = from.mtp_auto;
  to.mtp_adaptive = from.mtp_adaptive;
}

// The server's reconfigure callback over `in` and the options the server `started` with.
template <class Flags>
server::Reconfigure reconfigure(InPlace in, const server::Options& started) {
  struct State {
    InPlace in;
    server::Options started;
    bool on;
    uint32_t k;
  };
  auto st = std::make_shared<State>(State{std::move(in), started, false, 0});
  st->on = st->in.spec != "off";
  st->k = st->in.spec_max;
  return [st](const nlohmann::json& settings, server::Options& opts) {
    server::Reconfigured r;
    bool spec_changed = false;
    for (const auto& [key, v] : settings.items()) {
      bool known = false;
      template for (constexpr std::meta::info m : cli::flags::members<Flags>())
        known = known || "--" + key == cli::flags::flag_name(std::meta::identifier_of(m));
      if (!known) {
        r.refused.push_back({key, key == "model" ? "serving another model needs a restart" : "not a b70-serve flag"});
        continue;
      }
      std::string value;
      if (v.is_string()) value = v.get<std::string>();
      else if (v.is_number_unsigned() || (v.is_number_integer() && v.get<int64_t>() >= 0)) value = std::to_string(v.get<uint64_t>());
      else {
        r.refused.push_back({key, "expects a string or a non-negative integer, as on the command line"});
        continue;
      }
      InPlace& in = st->in;
      const std::string spec_now = st->on ? in.spec : "off";
      if (key == "spec") {
        const bool is_auto = value == "auto" && in.spec_auto;   // auto is what it resolved to at startup
        const bool on = is_auto ? in.spec != "off" : value != "off";
        if (on && !is_auto && value != in.spec) {
          r.refused.push_back({key, "--spec " + value + " needs a restart: this server " +
                                        (in.spec == "off" ? std::string("loaded no proposer; it stays off")
                                                          : "switches between " + in.spec + " and off")});
          continue;
        }
        if (on != st->on) r.applied.push_back({key, spec_now, on ? in.spec : "off"});
        st->on = on;
        spec_changed = true;
      } else if (key == "spec-max") {
        uint32_t k = 0;
        try {
          k = cli::flags::parse_u32("--spec-max", value);
        } catch (const std::exception& e) {
          r.refused.push_back({key, e.what()});
          continue;
        }
        if (in.spec == "off") {
          r.refused.push_back({key, "this server started with no proposer (--spec off)"});
          continue;
        }
        if (k == 0 || k > in.spec_max) {
          r.refused.push_back({key, "--spec-max " + value + ": this server's " + in.spec + " runs K 1.." +
                                        std::to_string(in.spec_max) + "; a larger K needs a restart"});
          continue;
        }
        if (k != st->k) r.applied.push_back({key, std::to_string(st->k), std::to_string(k)});
        st->k = k;
        spec_changed = true;
      } else if (key == "host-experts") {
        if (value != "auto" && value != "gpu" && value != "cpu") {
          r.refused.push_back({key, "--host-experts " + value + ": use auto, gpu or cpu"});
          continue;
        }
        if (!in.set_host_experts) {
          r.refused.push_back({key, in.host_experts_why});
          continue;
        }
        if (value != in.host_experts) {
          in.set_host_experts(value);
          r.applied.push_back({key, in.host_experts, value});
          in.host_experts = value;
        }
      } else {
        r.refused.push_back({key, restart_reason(key)});
      }
    }
    if (spec_changed) {
      copy_spec(opts, st->on ? st->started : server::Options{});
      if (st->on) {
        if (opts.mtp_auto) opts.mtp_adaptive.max_k = st->k;
        if (opts.spec_lookup) opts.lookup_adaptive.max_k = st->k;
      }
      st->in.set_proposer(st->k, st->on);
    }
    return r;
  };
}

}  // namespace cli
