#include "server/server.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>

#include <httplib.h>

#include "server/batch_engine.h"
#include "server/lookup_in_dflash.h"
#include "server/prompt_lookup.h"
#include "server/scheduler.h"

namespace server {
namespace {

uint64_t unix_time() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
                                    std::chrono::system_clock::now().time_since_epoch())
                                    .count());
}

double steady_seconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string request_id(bool chat, uint64_t sequence) {
  return std::string(chat ? "chatcmpl-" : "cmpl-") + std::to_string(sequence);
}

// Plan 13b: the scheduler's queue is full (the serial server's 503).
struct Busy : std::runtime_error {
  using std::runtime_error::runtime_error;
};

}  // namespace

struct Server::Impl {
  httplib::Server svr;
  std::set<uint32_t> eos;
  mutable std::mutex state_mutex;
  std::condition_variable queue_changed;
  uint64_t next_ticket = 0;
  uint64_t serving = 0;
  size_t waiting = 0;
  int port = 0;
  std::thread thread;
  std::atomic<uint64_t> next_id{0};
  std::atomic<uint64_t> next_log{0};
  uint64_t created = unix_time();
  std::unique_ptr<PrefixSession> prefix;
  // Spec 19e: the request's ids (and, with lookup_history, earlier requests' generated ids).
  PromptLookup lookup;
  std::deque<std::vector<uint32_t>> lookup_history;
  // Spec 19f: --spec dflash-lookup's policy (its state is per request, its counters the server's).
  std::optional<LookupInDFlash> lookup_in_dflash;
  // Plan 13b: the scheduler path (an engine of more than one slot).
  BatchEngineIface* batch = nullptr;
  std::mutex prompt_mutex;   // render + encode, one request at a time (as the serial server's queue made them)
  std::unique_ptr<PrefixSlots> slot_cache;
  std::unique_ptr<Scheduler> sched;
};

Server::Server(Deps deps, Options opts, BatchEngineIface* batch)
    : deps_(deps), opts_(std::move(opts)), impl_(std::make_unique<Impl>()) {
  impl_->eos.insert(opts_.eos_ids.begin(), opts_.eos_ids.end());
  if (batch != nullptr && batch->slots() > 1) {
    impl_->batch = batch;
    if (opts_.spec_dflash_lookup)
      throw std::invalid_argument("--spec dflash-lookup is built for one engine (spec 19f); not under the scheduler");
  }
  if (opts_.spec_lookup) {
    if (deps_.engine.verify_k() == 0)
      throw std::invalid_argument("--spec lookup: this engine cannot verify external drafts");
    if (deps_.engine.mtp_k() != 0)
      throw std::invalid_argument("--spec lookup and --spec mtp are one proposer each; pick one");
    if (opts_.lookup_min_match < PromptLookup::kMinMatch ||
        opts_.lookup_min_match > impl_->lookup.options().max_match)
      throw std::invalid_argument("the lookup minimum match must be in [2, 64]");
  }
  if (opts_.spec_dflash_lookup) {   // spec 19f
    if (opts_.spec_lookup)
      throw std::invalid_argument("--spec dflash-lookup and --spec lookup are one proposer each; pick one");
    if (deps_.engine.mtp_k() == 0 || deps_.engine.dflash_window() == 0)
      throw std::invalid_argument("--spec dflash-lookup: this engine has no DFlash drafter to fall back to");
    LookupInDFlashOptions o;
    o.window = deps_.engine.dflash_window();
    o.max_k = deps_.engine.mtp_k();
    impl_->lookup_in_dflash.emplace(o);
  }
  // Plan 13b: under the scheduler the cache is per slot (PrefixSlots) and deps_.engine is never hooked or driven.
  impl_->prefix = std::make_unique<PrefixSession>(deps_.engine, impl_->batch ? 0 : opts_.prefix_cache_bytes,
                                                  opts_.prefix_alloc);
  if (impl_->batch) {
    BatchEngineIface& b = *impl_->batch;
    SchedulerOptions so;
    so.queue_depth = opts_.queue_depth;
    so.eos_ids = opts_.eos_ids;
    if (b.spec_k() > 0) {   // --spec mtp (or DFlash): the serial loop's K rule, per request
      so.spec = opts_.mtp_adaptive;
      so.spec_fixed_k = opts_.mtp_auto ? 0 : b.spec_k();
    }
    so.lookup = opts_.spec_lookup;
    so.lookup_min_match = opts_.lookup_min_match;
    so.lookup_history = opts_.lookup_history;
    so.lookup_adaptive = opts_.lookup_adaptive;
    if (opts_.prefix_cache_bytes != 0) {
      if (opts_.prefix_alloc == nullptr) throw std::invalid_argument("Server: a prefix cache budget needs an allocator");
      impl_->slot_cache = std::make_unique<PrefixSlots>(b, opts_.prefix_cache_bytes, *opts_.prefix_alloc);
    }
    impl_->sched = std::make_unique<Scheduler>(b, so, impl_->slot_cache.get());
    impl_->sched->start();
  }
  impl_->svr.set_tcp_nodelay(opts_.tcp_nodelay);
  impl_->svr.Post("/v1/chat/completions",
                  [this](const httplib::Request& request, httplib::Response& response) {
                    handle(request, response, true);
                  });
  impl_->svr.Post("/v1/completions",
                  [this](const httplib::Request& request, httplib::Response& response) {
                    handle(request, response, false);
                  });
  impl_->svr.Get("/v1/models", [this](const httplib::Request&, httplib::Response& response) {
    response.set_content(models_body(opts_.served_model, impl_->created), "application/json");
  });
}

Server::~Server() {
  stop();
  if (impl_ && impl_->sched) impl_->sched->stop();
}

bool Server::bind() {
  int bound = 0;
  if (opts_.port == 0) {
    bound = impl_->svr.bind_to_any_port(opts_.host);
    if (bound <= 0) return false;
  } else {
    if (!impl_->svr.bind_to_port(opts_.host, opts_.port)) return false;
    bound = opts_.port;
  }
  std::lock_guard<std::mutex> lock(impl_->state_mutex);
  impl_->port = bound;
  return true;
}

bool Server::listen() {
  if (!bind()) return false;
  return impl_->svr.listen_after_bind();
}

bool Server::start() {
  {
    std::lock_guard<std::mutex> lock(impl_->state_mutex);
    if (impl_->thread.joinable() || impl_->port != 0) return false;
  }
  if (!bind()) return false;
  impl_->thread = std::thread([this] { (void)impl_->svr.listen_after_bind(); });
  return true;
}

void Server::stop() {
  if (!impl_) return;
  impl_->svr.stop();
  if (impl_->thread.joinable()) impl_->thread.join();
}

int Server::bound_port() const {
  std::lock_guard<std::mutex> lock(impl_->state_mutex);
  return impl_->port;
}

bool Server::acquire(uint64_t& ticket) {
  std::unique_lock<std::mutex> lock(impl_->state_mutex);
  const bool busy = impl_->next_ticket != impl_->serving;
  if (busy && impl_->waiting >= opts_.queue_depth) {
    ticket = static_cast<uint64_t>(impl_->waiting);
    return false;
  }
  ticket = impl_->next_ticket++;
  if (ticket == impl_->serving) return true;
  ++impl_->waiting;
  impl_->queue_changed.wait(lock, [this, ticket] { return ticket == impl_->serving; });
  --impl_->waiting;
  return true;
}

void Server::release() {
  {
    std::lock_guard<std::mutex> lock(impl_->state_mutex);
    ++impl_->serving;
  }
  impl_->queue_changed.notify_all();
}

std::string Server::render_prompt(const Request& r) {
  std::string prompt;
  try {
    if (!r.chat) {
      prompt = r.prompt;
    } else if (opts_.chat_format.template_kwargs()) {   // spec 18d (K2-Horizon), 20e (Kolibri-1), 25e (Bolt)
      // Spec 25e: Bolt maps the top-level reasoning_effort onto its template switches (decision 13).
      const nlohmann::json kwargs = request_template_kwargs(opts_.chat_format, r.template_kwargs, r.reasoning_effort);
      check_template_kwargs(opts_.chat_format, kwargs);
      // Spec 25e: the kwargs after the reasoning_effort mapping may carry the enable_thinking the request did not (Bolt: none /
      // minimal); the template variable is that one (for every other format the kwargs' value IS r.enable_thinking).
      const bool thinking = kwargs.is_object() && kwargs.contains("enable_thinking") && kwargs.at("enable_thinking").is_boolean()
                                ? kwargs.at("enable_thinking").get<bool>()
                                : r.enable_thinking;
      prompt = deps_.tmpl.render_with_kwargs(template_messages(opts_.chat_format, r.messages), r.tools, thinking, kwargs);
    } else {
      prompt = deps_.tmpl.render(r.messages, r.tools, r.enable_thinking);
    }
  } catch (const std::exception& error) {
    throw BadRequest(error.what());
  }
  return prompt;
}

Server::Outcome Server::generate(const Request& r,
                                 const std::function<void(const Delta&)>& emit) {
  Outcome outcome;
  const std::string prompt = render_prompt(r);
  outcome.prompt_ids = deps_.tok.encode(prompt);
  const uint32_t max_len = deps_.engine.max_len();
  if (outcome.prompt_ids.empty()) throw BadRequest("prompt encodes to zero tokens");
  if (outcome.prompt_ids.size() + 1 > max_len) {
    throw BadRequest("prompt is " + std::to_string(outcome.prompt_ids.size()) +
                     " tokens; max_model_len is " + std::to_string(max_len));
  }
  const uint32_t budget = static_cast<uint32_t>(max_len - outcome.prompt_ids.size());
  const uint32_t max_tokens = std::min(r.max_tokens.value_or(budget), budget);
  outcome.usage.prompt_tokens = static_cast<uint32_t>(outcome.prompt_ids.size());

  // Spec 7 §3.3: restart from the resident session or the deepest host snapshot, prefill
  // the tail (the block hook writes it through); with the cache off, reset() + prefill.
  PrefixSession& session = *impl_->prefix;
  outcome.prefix = session.begin(outcome.prompt_ids, r.chat && opts_.prefix_split_last);
  outcome.usage.cached_tokens = outcome.prefix.restart;
  if (session.enabled()) {
    std::fprintf(stderr,
                 "prefix: kind %s restart %u of %zu kv_bytes %zu restore_ms %.1f prefill_ms %.1f"
                 " store_ms %.1f\n",
                 plan_kind_name(outcome.prefix.kind), outcome.prefix.restart,
                 outcome.prompt_ids.size(), outcome.prefix.kv_bytes, outcome.prefix.restore_ms,
                 outcome.prefix.prefill_ms, outcome.prefix.store_ms);
    std::fflush(stderr);
  }
  try {
    generate_tail(r, prompt, max_tokens, outcome, emit);
  } catch (...) {
    session.fail();
    throw;
  }
  return outcome;
}

void Server::generate_tail(const Request& r, const std::string& prompt, uint32_t max_tokens,
                           Outcome& outcome, const std::function<void(const Delta&)>& emit) {
  PrefixSession& session = *impl_->prefix;
  // Spec 24e (decision 1): a request with a repetition penalty carries the distinct ids of its prompt and of every id
  // consumed so far, for an engine adapter that applies it (the host sampler, before temperature / top-k / top-p).
  SeenIds seen;
  Sampling sampling = r.sampling;
  const bool penalised = sampling.repetition_penalty != 1.0f;
  if (penalised) {
    for (uint32_t id : outcome.prompt_ids) seen.add(id);
    sampling.seen = &seen;
  }
  std::unique_ptr<StreamerIface> streamer = deps_.tok.streamer();
  if (!streamer) throw BadRequest("tokenizer did not provide a streamer");

  size_t hold = 0;
  for (const std::string& stop : r.stop) {
    hold = std::max(hold, stop.empty() ? 0U : stop.size() - 1);
  }
  // Stop strings apply to content; `pending` holds content not yet emitted.
  std::string pending;
  const auto flush_pending = [&](bool final) {
    const size_t keep = final ? 0 : std::min(hold, pending.size());
    if (pending.size() <= keep) return;
    Delta d;
    d.kind = Delta::Content;
    d.text = pending.substr(0, pending.size() - keep);
    outcome.text += d.text;
    outcome.parsed.content += d.text;
    pending.erase(0, d.text.size());
    emit(d);
  };
  // Chat: the text goes through the tool-call parser (spec 7 §3.5) of the model's format
  // (spec 18d, server/chat_format.h). Qwen: with thinking on, the prompt ends in "<think>\n"
  // and the output opens with reasoning; K2-Horizon: <ifm|think>\n and its own call syntax.
  std::unique_ptr<OutputParser> parser;
  if (r.chat) parser = make_output_parser(opts_.chat_format, prompt, r.template_kwargs, r.tools);
  const auto route = [&](const std::vector<Delta>& deltas) {
    for (const Delta& d : deltas) {
      if (d.kind == Delta::Content) {
        pending += d.text;
        continue;
      }
      flush_pending(true);
      if (d.kind == Delta::Reasoning) outcome.parsed.reasoning += d.text;
      else outcome.parsed.tool_calls.push_back(d.call);
      emit(d);
    }
  };
  const auto add_text = [&](const std::string& text) {
    outcome.raw_text += text;
    if (parser) route(parser->push(text));
    else pending += text;
  };

  outcome.finish_reason = "length";
  bool eos_stop = false;
  std::optional<AdaptiveK> policy;   // --spec mtp (spec 8 §10) or --spec lookup (spec 19e)
  // Spec 8: with MTP a step yields a burst of ids; the loop below still sees them one
  // at a time, and ids past a stop are handed back with truncate_to() after it.
  const bool speculative = deps_.engine.mtp_k() > 0;
  // Spec 19e: with --spec lookup the burst comes from step_drafts(), its drafts from the
  // matcher, which holds exactly the prompt and every id consumed so far (the pending id x
  // is indexed when the engine hands it to the proposer: it is consumed next, always).
  const bool lookup = opts_.spec_lookup;
  if (lookup) {
    lookup_begin(outcome.prompt_ids);
    AdaptiveKOptions o = opts_.lookup_adaptive;
    o.max_k = std::min(o.max_k, deps_.engine.verify_k());
    policy.emplace(o);
  }
  // Spec 19f: --spec dflash-lookup indexes the matcher as --spec lookup does; the DFlash iterations' K comes from the
  // --spec mtp policy below.
  const bool dflash_lookup = opts_.spec_dflash_lookup;
  if (dflash_lookup) {
    lookup_begin(outcome.prompt_ids);
    impl_->lookup_in_dflash->reset();
  }
  bool burst_head_indexed = false;   // burst[0] went into the matcher in the proposer
  // Spec 8 §10: --spec mtp keeps one policy per request, fed only by this request's own
  // acceptance, so the K sequence (and a seeded request's output) is reproducible.
  if (speculative && opts_.mtp_auto) {
    AdaptiveKOptions o = opts_.mtp_adaptive;
    o.max_k = std::min(o.max_k, deps_.engine.mtp_k());
    policy.emplace(o);
  }
  std::vector<uint32_t> burst;
  size_t burst_at = 0;
  PromptLookup& matcher = impl_->lookup;
  const auto next_lookup = [&]() -> uint32_t {
    if (burst_at == burst.size()) {
      const uint32_t k = policy->next();
      uint32_t proposed = 0;
      burst_head_indexed = false;
      burst = deps_.engine.step_drafts(sampling, [&](uint32_t pending) {
        if (burst_head_indexed) throw std::logic_error("EngineIface::step_drafts: propose twice");
        matcher.append(pending);
        burst_head_indexed = true;
        std::vector<uint32_t> d =
            k > 0 ? matcher.propose(k, opts_.lookup_min_match) : std::vector<uint32_t>{};
        proposed = static_cast<uint32_t>(d.size());
        return d;
      });
      burst_at = 0;
      if (burst.empty()) throw std::runtime_error("EngineIface::step_drafts returned no ids");
      if (burst.size() > size_t(proposed) + 1)
        throw std::runtime_error("EngineIface::step_drafts returned more ids than drafts + 1");
      policy->observe(proposed, uint32_t(burst.size() - 1));
    }
    const uint32_t id = burst[burst_at];
    if (burst_at > 0 || !burst_head_indexed) matcher.append(id);
    ++burst_at;
    return id;
  };
  // Spec 19f: one iteration asks the policy first (the pending id indexed in the proposer, as next_lookup's); lookup
  // drafts feed the policy's STAY / EXIT, a DFlash iteration feeds AdaptiveK - each proposer learns from its own.
  const auto next_dflash_lookup = [&]() -> uint32_t {
    if (burst_at == burst.size()) {
      LookupInDFlash& lid = *impl_->lookup_in_dflash;
      const uint32_t k = policy ? policy->next() : deps_.engine.mtp_k();
      uint32_t proposed = 0;
      burst_head_indexed = false;
      burst = deps_.engine.step_drafts_or_many(sampling, k, [&](uint32_t pending) {
        if (burst_head_indexed) throw std::logic_error("EngineIface::step_drafts_or_many: propose twice");
        matcher.append(pending);
        burst_head_indexed = true;
        const std::vector<uint32_t> d = lid.propose(matcher);
        proposed = static_cast<uint32_t>(d.size());
        return d;
      });
      burst_at = 0;
      if (burst.empty()) throw std::runtime_error("EngineIface::step_drafts_or_many returned no ids");
      const uint32_t kept = static_cast<uint32_t>(burst.size() - 1);
      if (proposed > 0) {
        if (kept > proposed)
          throw std::runtime_error("EngineIface::step_drafts_or_many returned more ids than drafts + 1");
        lid.observe(proposed, kept);
      } else if (policy) {
        policy->observe(k, std::min(k, kept));
      }
    }
    const uint32_t id = burst[burst_at];
    if (burst_at > 0 || !burst_head_indexed) matcher.append(id);
    ++burst_at;
    return id;
  };
  const auto next_id = [&]() -> uint32_t {
    if (lookup) return next_lookup();
    if (dflash_lookup) return next_dflash_lookup();
    if (!speculative) return deps_.engine.step(sampling);
    if (burst_at == burst.size()) {
      const uint32_t k = policy ? policy->next() : deps_.engine.mtp_k();
      burst = deps_.engine.step_many(sampling, k);
      burst_at = 0;
      if (burst.empty()) throw std::runtime_error("EngineIface::step_many returned no ids");
      if (policy) policy->observe(k, std::min<uint32_t>(k, uint32_t(burst.size() - 1)));
    }
    return burst[burst_at++];
  };
  for (uint32_t generated = 0; generated < max_tokens; ++generated) {
    const uint32_t id = next_id();
    session.fed(id);
    if (penalised) seen.add(id);
    if (generated == 0) outcome.t_first_token = steady_seconds();
    const bool is_eos = impl_->eos.count(id) != 0;
    if (is_eos && !r.ignore_eos && outcome.out_ids.size() >= r.min_tokens) {
      outcome.finish_reason = "stop";
      eos_stop = true;
      break;
    }
    outcome.out_ids.push_back(id);
    if (!is_eos) add_text(streamer->push(id));

    bool stopped = false;
    for (const std::string& stop : r.stop) {
      const size_t at = pending.find(stop);
      if (at != std::string::npos) {
        pending.erase(at);
        stopped = true;
        break;
      }
    }
    if (stopped) {
      outcome.finish_reason = "stop";
      flush_pending(true);
      break;
    }
    flush_pending(false);
  }
  if (burst_at < burst.size())
    deps_.engine.truncate_to(deps_.engine.pos() - static_cast<uint32_t>(burst.size() - burst_at));
  if (outcome.finish_reason == "length") add_text(streamer->flush());
  if (parser && (eos_stop || outcome.finish_reason == "length")) route(parser->finish());
  flush_pending(true);
  if (eos_stop && !outcome.parsed.tool_calls.empty()) outcome.finish_reason = "tool_calls";
  outcome.usage.completion_tokens = static_cast<uint32_t>(outcome.out_ids.size());
  if (lookup) lookup_end(outcome.out_ids);
}

// Spec 19e: index the request. Without history the matcher is reused from the longest
// common prefix of what it holds (the previous request's prompt and kept ids) and this
// prompt - an agentic turn re-sends the conversation, so the tail is all that is new. With
// history it is rebuilt: the earlier requests' generated ids, each behind a boundary, then
// the prompt (the prompt's matches are the more recent on ties).
void Server::lookup_begin(const std::vector<uint32_t>& prompt_ids) {
  PromptLookup& lk = impl_->lookup;
  if (opts_.lookup_history == 0) {
    const std::vector<uint32_t>& held = lk.ids();
    size_t common = 0;
    const size_t n = std::min(held.size(), prompt_ids.size());
    while (common < n && held[common] == prompt_ids[common]) ++common;
    lk.truncate(common);
    for (size_t i = common; i < prompt_ids.size(); ++i) lk.append(prompt_ids[i]);
    return;
  }
  lk.clear();
  for (const std::vector<uint32_t>& earlier : impl_->lookup_history) {
    lk.append(earlier);
    lk.boundary();
  }
  lk.append(prompt_ids);
}

void Server::lookup_end(const std::vector<uint32_t>& out_ids) {
  if (!opts_.spec_lookup || opts_.lookup_history == 0) return;
  impl_->lookup_history.push_back(out_ids);
  while (impl_->lookup_history.size() > opts_.lookup_history) impl_->lookup_history.pop_front();
}

void Server::finish_request() {
  if (!impl_->prefix->enabled()) return;
  try {
    const double ms = impl_->prefix->end();
    std::fprintf(stderr, "prefix: request end, pos %u, store_ms %.1f\n", deps_.engine.pos(), ms);
    std::fflush(stderr);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "b70-serve: request-end snapshot failed: %s\n", error.what());
  }
}

void Server::log_request(bool chat, const std::string& body, double t_start,
                         const Outcome& outcome) {
  if (opts_.log_requests_dir.empty()) return;
  nlohmann::json record = {
      {"t_start", t_start},
      {"t_first_token", outcome.t_first_token == 0 ? t_start : outcome.t_first_token},
      {"t_end", steady_seconds()},
      {"endpoint", chat ? "/v1/chat/completions" : "/v1/completions"},
      {"request", nlohmann::json::parse(body, nullptr, false)},
      {"prompt_ids", outcome.prompt_ids},
      {"out_ids", outcome.out_ids},
      {"response_text", outcome.raw_text},
  };
  if (outcome.slot >= 0) record["slot"] = outcome.slot;   // plan 13b: which card (the serial record is unchanged)
  char name[32];
  std::snprintf(name, sizeof(name), "/%06llu.json",
                static_cast<unsigned long long>(++impl_->next_log));
  const std::string path = opts_.log_requests_dir + name;
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) {
    std::fprintf(stderr, "b70-serve: cannot write request log %s\n", path.c_str());
    return;
  }
  const std::string text = record.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
  std::fwrite(text.data(), 1, text.size(), f);
  std::fclose(f);
}

void Server::handle(const httplib::Request& request, httplib::Response& response, bool chat) {
  const double t_start = steady_seconds();
  Request parsed;
  try {
    parsed = parse_request(request.body, chat, opts_.sampling_defaults.value_or(Sampling{}));
    // Spec 23e: generation_config.json's stop_strings beside the request's own stop.
    for (const std::string& s : opts_.stop_strings)
      if (!s.empty() && std::find(parsed.stop.begin(), parsed.stop.end(), s) == parsed.stop.end()) parsed.stop.push_back(s);
  } catch (const BadRequest& error) {
    response.status = 400;
    response.set_content(error_body(error.what(), "invalid_request_error"), "application/json");
    return;
  }
  if (impl_->sched) {   // plan 13b: up to slots() requests at once
    handle_batched(request, response, parsed, t_start);
    return;
  }

  uint64_t ticket = 0;
  if (!acquire(ticket)) {
    response.status = 503;
    response.set_content(error_body("server busy: " + std::to_string(ticket) + " requests queued",
                                    "server_overloaded"),
                         "application/json");
    return;
  }

  if (!parsed.stream) {
    try {
      const Outcome outcome = generate(parsed, [](const Delta&) {});
      log_request(parsed.chat, request.body, t_start, outcome);
      const auto* prompt_ids = parsed.return_token_ids ? &outcome.prompt_ids : nullptr;
      const auto* out_ids = parsed.return_token_ids ? &outcome.out_ids : nullptr;
      const std::string id = request_id(parsed.chat, ++impl_->next_id);
      response.set_content(completion_body(parsed, id, unix_time(), outcome.text,
                                           outcome.finish_reason, outcome.usage, prompt_ids, out_ids,
                                           parsed.chat ? &outcome.parsed : nullptr),
                           "application/json");
      finish_request();
    } catch (const BadRequest& error) {
      response.status = 400;
      response.set_content(error_body(error.what(), "invalid_request_error"), "application/json");
    } catch (const std::exception& error) {
      response.status = 500;
      response.set_content(error_body(error.what(), "server_error"), "application/json");
    }
    release();
    return;
  }

  response.set_header("Cache-Control", "no-cache");
  response.set_header("X-Accel-Buffering", "no");
  response.set_chunked_content_provider(
      "text/event-stream",
      [this, parsed, body = request.body, t_start](size_t, httplib::DataSink& sink) mutable {
        const std::string id = request_id(parsed.chat, ++impl_->next_id);
        const uint64_t created = unix_time();
        const auto write = [&sink](const std::string& frame) {
          return sink.write(frame.data(), frame.size());
        };
        // The role frame goes out with the first delta, after prefill, as vLLM
        // sends it: clients time prefill to the first chunk with `choices`
        // (llama-benchy's est_ppt), so a frame before prefill reads as ~0 ms.
        bool role_sent = !parsed.chat;
        bool ok = false;
        const auto send_role = [&] {
          if (role_sent) return;
          role_sent = true;
          (void)write(stream_frame_role(parsed, id, created));
        };
        try {
          const Outcome outcome = generate(parsed, [&](const Delta& d) {
            if (!d.text.empty() || d.kind == Delta::Call) send_role();
            if (d.kind == Delta::Call) {
              (void)write(stream_frame_tool_call(parsed, id, created, d.call, d.index));
            } else if (d.text.empty()) {
            } else if (d.kind == Delta::Reasoning) {
              (void)write(stream_frame_reasoning(parsed, id, created, d.text));
            } else {
              (void)write(stream_frame_text(parsed, id, created, d.text));
            }
          });
          send_role();
          (void)write(stream_frame_finish(parsed, id, created, outcome.finish_reason));
          if (parsed.include_usage) (void)write(stream_frame_usage(parsed, id, created, outcome.usage));
          log_request(parsed.chat, body, t_start, outcome);
          ok = true;
        } catch (const BadRequest& error) {
          (void)write(std::string("data: ") +
                      error_body(error.what(), "invalid_request_error") + "\n\n");
        } catch (const std::exception& error) {
          (void)write(std::string("data: ") + error_body(error.what(), "server_error") + "\n\n");
        }
        (void)write(kDone);
        if (ok) finish_request();
        sink.done();
        release();
        return true;
      });
}


// ---- Plan 13b: the scheduler path -------------------------------------------------------------------------------
//
// One request under server::Scheduler. The scheduler's tick thread runs generate_tail's per-id rules for it in
// on_token / on_done (the streamer, the stop strings, the tool-call parser, the repetition penalty's seen ids) -
// the same steps in the same order as the serial loop - and queues the deltas; the request's own HTTP thread waits
// for them and writes them out. A write that fails (the client is gone) cancels the request: its slot is free at
// the next tick (Review Focus 6).
struct Server::BatchedRun {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<Delta> deltas;   // mu
    bool done = false;          // mu
    // The tick thread's until done; the HTTP thread's after.
    Outcome out;
    Finish fin;
    SeenIds seen;
    std::unique_ptr<StreamerIface> streamer;
    std::unique_ptr<OutputParser> parser;
    std::vector<std::string> stops;
    std::string pending;
    size_t hold = 0;
    bool stopped = false;   // a stop string ended it
    // The scheduler's Job holds the callbacks, which hold this run: a weak reference back, no cycle.
    std::weak_ptr<Job> job;
};

std::shared_ptr<Server::BatchedRun> Server::submit_batched(const Request& r) {
  using Run = BatchedRun;
  auto run = std::make_shared<Run>();
  Outcome& outcome = run->out;
  std::string prompt;
  {
    std::lock_guard<std::mutex> lock(impl_->prompt_mutex);
    prompt = render_prompt(r);
    outcome.prompt_ids = deps_.tok.encode(prompt);
  }
  const uint32_t max_len = impl_->batch->max_len();
  if (outcome.prompt_ids.empty()) throw BadRequest("prompt encodes to zero tokens");
  if (outcome.prompt_ids.size() + 1 > max_len) {
    throw BadRequest("prompt is " + std::to_string(outcome.prompt_ids.size()) +
                     " tokens; max_model_len is " + std::to_string(max_len));
  }
  outcome.usage.prompt_tokens = static_cast<uint32_t>(outcome.prompt_ids.size());
  outcome.finish_reason = "length";
  run->streamer = deps_.tok.streamer();
  if (!run->streamer) throw BadRequest("tokenizer did not provide a streamer");
  run->stops = r.stop;
  for (const std::string& stop : r.stop) run->hold = std::max(run->hold, stop.empty() ? 0U : stop.size() - 1);
  if (r.chat) run->parser = make_output_parser(opts_.chat_format, prompt, r.template_kwargs, r.tools);

  SchedRequest sr;
  sr.prompt = outcome.prompt_ids;
  sr.max_tokens = r.max_tokens.value_or(UINT32_MAX);
  sr.min_tokens = r.min_tokens;
  sr.ignore_eos = r.ignore_eos;
  sr.split_last = r.chat && opts_.prefix_split_last;
  sr.sampling = r.sampling;
  const bool penalised = sr.sampling.repetition_penalty != 1.0f;
  if (penalised) {
    for (uint32_t id : outcome.prompt_ids) run->seen.add(id);
    sr.sampling.seen = &run->seen;
  }
  const std::shared_ptr<Run> rp = run;   // the callbacks keep the run alive until the scheduler drops them
  const auto push = [rp](const Delta& d) {
    std::lock_guard<std::mutex> lock(rp->mu);
    rp->deltas.push_back(d);
    rp->cv.notify_all();
  };
  // generate_tail's flush_pending / route / add_text over the run's state (the tick thread).
  const auto flush_pending = [rp, push](bool final) {
    const size_t keep = final ? 0 : std::min(rp->hold, rp->pending.size());
    if (rp->pending.size() <= keep) return;
    Delta d;
    d.kind = Delta::Content;
    d.text = rp->pending.substr(0, rp->pending.size() - keep);
    rp->out.text += d.text;
    rp->out.parsed.content += d.text;
    rp->pending.erase(0, d.text.size());
    push(d);
  };
  const auto route = [rp, push, flush_pending](const std::vector<Delta>& deltas) {
    for (const Delta& d : deltas) {
      if (d.kind == Delta::Content) {
        rp->pending += d.text;
        continue;
      }
      flush_pending(true);
      if (d.kind == Delta::Reasoning) rp->out.parsed.reasoning += d.text;
      else rp->out.parsed.tool_calls.push_back(d.call);
      push(d);
    }
  };
  const auto add_text = [rp, route](const std::string& text) {
    rp->out.raw_text += text;
    if (rp->parser) route(rp->parser->push(text));
    else rp->pending += text;
  };
  const std::set<uint32_t>* eos = &impl_->eos;
  sr.on_token = [rp, eos, penalised, add_text, flush_pending](uint32_t id) {
    Outcome& o = rp->out;
    if (penalised) rp->seen.add(id);
    if (o.out_ids.empty() && o.t_first_token == 0) o.t_first_token = steady_seconds();
    const bool is_eos = eos->count(id) != 0;
    o.out_ids.push_back(id);
    if (!is_eos) add_text(rp->streamer->push(id));
    for (const std::string& stop : rp->stops) {
      const size_t at = rp->pending.find(stop);
      if (at != std::string::npos) {
        rp->pending.erase(at);
        o.finish_reason = "stop";
        rp->stopped = true;
        flush_pending(true);
        return false;
      }
    }
    flush_pending(false);
    return true;
  };
  sr.on_done = [rp, add_text, route, flush_pending](const Finish& fin) {
    Outcome& o = rp->out;
    rp->fin = fin;
    if (fin.reason == Finish::Stop || fin.reason == Finish::Length) {
      const bool eos_stop = fin.reason == Finish::Stop && !rp->stopped;
      if (eos_stop) {
        o.finish_reason = "stop";
        if (o.t_first_token == 0) o.t_first_token = steady_seconds();   // the EOS was the first id
      }
      if (o.finish_reason == "length") add_text(rp->streamer->flush());
      if (rp->parser && (eos_stop || o.finish_reason == "length")) route(rp->parser->finish());
      flush_pending(true);
      if (eos_stop && !o.parsed.tool_calls.empty()) o.finish_reason = "tool_calls";
    }
    o.usage.completion_tokens = static_cast<uint32_t>(o.out_ids.size());
    o.usage.cached_tokens = fin.cached_tokens;
    std::lock_guard<std::mutex> lock(rp->mu);
    rp->done = true;
    rp->cv.notify_all();
  };

  std::shared_ptr<Job> job;   // the scheduler holds the run's callbacks until on_done
  try {
    job = impl_->sched->submit(std::move(sr));
  } catch (const std::invalid_argument& error) {
    throw BadRequest(error.what());
  }
  if (!job) throw Busy("server busy: " + std::to_string(impl_->sched->waiting()) + " requests queued");
  run->job = job;
  return run;
}

void Server::cancel_batched(BatchedRun& run) {
  if (std::shared_ptr<Job> job = run.job.lock()) impl_->sched->cancel(job);
}

Server::Outcome Server::wait_batched(BatchedRun& r, const std::function<bool(const Delta&)>& emit) {
  BatchedRun* const run = &r;
  const Outcome& outcome = run->out;
  bool alive = true;
  for (;;) {
    std::deque<Delta> deltas;
    bool done = false;
    {
      std::unique_lock<std::mutex> lock(run->mu);
      run->cv.wait(lock, [&] { return !run->deltas.empty() || run->done; });
      deltas.swap(run->deltas);
      done = run->done;
    }
    for (const Delta& d : deltas) {
      if (alive && !emit(d)) {
        alive = false;
        cancel_batched(*run);
      }
    }
    if (done) break;
  }
  const Finish& fin = run->fin;
  if (impl_->slot_cache) {
    std::fprintf(stderr, "prefix: slot %d kind %s restart %u of %zu\n", fin.slot,
                 fin.slot >= 0 ? plan_kind_name(impl_->slot_cache->last_kind(uint32_t(fin.slot))) : "none",
                 fin.cached_tokens, outcome.prompt_ids.size());
    std::fflush(stderr);
  }
  if (fin.reason == Finish::Error) throw std::runtime_error(fin.error);
  run->out.slot = fin.slot;
  return std::move(run->out);
}

void Server::handle_batched(const httplib::Request& request, httplib::Response& response, const Request& parsed,
                            double t_start) {
  // The submit comes first, as the serial server's acquire(): a full queue is a 503 before any stream starts. A
  // prompt the template or the length check refuses is a 400 - inside the stream for a streaming request, as the
  // serial server reports it.
  std::shared_ptr<BatchedRun> run;
  std::string refused;
  try {
    run = submit_batched(parsed);
  } catch (const Busy& error) {
    response.status = 503;
    response.set_content(error_body(error.what(), "server_overloaded"), "application/json");
    return;
  } catch (const BadRequest& error) {
    if (!parsed.stream) {
      response.status = 400;
      response.set_content(error_body(error.what(), "invalid_request_error"), "application/json");
      return;
    }
    refused = error.what();
  }
  if (!parsed.stream) {
    try {
      const Outcome outcome = wait_batched(*run, [](const Delta&) { return true; });
      log_request(parsed.chat, request.body, t_start, outcome);
      const auto* prompt_ids = parsed.return_token_ids ? &outcome.prompt_ids : nullptr;
      const auto* out_ids = parsed.return_token_ids ? &outcome.out_ids : nullptr;
      const std::string id = request_id(parsed.chat, ++impl_->next_id);
      response.set_content(completion_body(parsed, id, unix_time(), outcome.text, outcome.finish_reason,
                                           outcome.usage, prompt_ids, out_ids,
                                           parsed.chat ? &outcome.parsed : nullptr),
                           "application/json");
    } catch (const std::exception& error) {
      response.status = 500;
      response.set_content(error_body(error.what(), "server_error"), "application/json");
    }
    return;
  }
  response.set_header("Cache-Control", "no-cache");
  response.set_header("X-Accel-Buffering", "no");
  response.set_chunked_content_provider(
      "text/event-stream",
      [this, parsed, run, refused, body = request.body, t_start](size_t, httplib::DataSink& sink) mutable {
        const std::string id = request_id(parsed.chat, ++impl_->next_id);
        const uint64_t created = unix_time();
        const auto write = [&sink](const std::string& frame) { return sink.write(frame.data(), frame.size()); };
        bool role_sent = !parsed.chat;
        const auto send_role = [&] {
          if (role_sent) return true;
          role_sent = true;
          return write(stream_frame_role(parsed, id, created));
        };
        if (!run) {
          (void)write(std::string("data: ") + error_body(refused, "invalid_request_error") + "\n\n");
        } else {
          try {
            const Outcome outcome = wait_batched(*run, [&](const Delta& d) {
              bool ok = true;
              if (!d.text.empty() || d.kind == Delta::Call) ok = send_role();
              if (d.kind == Delta::Call) {
                ok = write(stream_frame_tool_call(parsed, id, created, d.call, d.index)) && ok;
              } else if (d.text.empty()) {
              } else if (d.kind == Delta::Reasoning) {
                ok = write(stream_frame_reasoning(parsed, id, created, d.text)) && ok;
              } else {
                ok = write(stream_frame_text(parsed, id, created, d.text)) && ok;
              }
              return ok;
            });
            (void)send_role();
            (void)write(stream_frame_finish(parsed, id, created, outcome.finish_reason));
            if (parsed.include_usage) (void)write(stream_frame_usage(parsed, id, created, outcome.usage));
            log_request(parsed.chat, body, t_start, outcome);
          } catch (const std::exception& error) {
            (void)write(std::string("data: ") + error_body(error.what(), "server_error") + "\n\n");
          }
        }
        (void)write(kDone);
        sink.done();
        return true;
      },
      // A client that left before the stream began never reaches the provider: its request is cancelled here
      // (a no-op once it finished).
      [this, run](bool) {
        if (run) cancel_batched(*run);
      });
}

}  // namespace server
