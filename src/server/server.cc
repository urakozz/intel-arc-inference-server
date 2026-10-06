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

#include "server/prompt_lookup.h"

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
};

Server::Server(Deps deps, Options opts)
    : deps_(deps), opts_(std::move(opts)), impl_(std::make_unique<Impl>()) {
  impl_->eos.insert(opts_.eos_ids.begin(), opts_.eos_ids.end());
  if (opts_.spec_lookup) {
    if (deps_.engine.verify_k() == 0)
      throw std::invalid_argument("--spec lookup: this engine cannot verify external drafts");
    if (deps_.engine.mtp_k() != 0)
      throw std::invalid_argument("--spec lookup and --mtp are one proposer each; pick one");
    if (opts_.lookup_min_match < PromptLookup::kMinMatch ||
        opts_.lookup_min_match > impl_->lookup.options().max_match)
      throw std::invalid_argument("--spec-min-match must be in [2, 64]");
  }
  impl_->prefix = std::make_unique<PrefixSession>(deps_.engine, opts_.prefix_cache_bytes,
                                                  opts_.prefix_alloc);
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

Server::~Server() { stop(); }

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

Server::Outcome Server::generate(const Request& r,
                                 const std::function<void(const Delta&)>& emit) {
  Outcome outcome;
  std::string prompt;
  try {
    if (!r.chat) {
      prompt = r.prompt;
    } else if (opts_.chat_format.template_kwargs()) {   // spec 18d (K2-Horizon), spec 20e (Kolibri-1)
      check_template_kwargs(opts_.chat_format, r.template_kwargs);
      prompt = deps_.tmpl.render_with_kwargs(template_messages(opts_.chat_format, r.messages), r.tools,
                                             r.enable_thinking, r.template_kwargs);
    } else {
      prompt = deps_.tmpl.render(r.messages, r.tools, r.enable_thinking);
    }
  } catch (const std::exception& error) {
    throw BadRequest(error.what());
  }
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
  std::optional<AdaptiveK> policy;   // --mtp auto (spec 8 §10) or --spec lookup (spec 19e)
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
  bool burst_head_indexed = false;   // burst[0] went into the matcher in the proposer
  // Spec 8 §10: --mtp auto keeps one policy per request, fed only by this request's own
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
      burst = deps_.engine.step_drafts(r.sampling, [&](uint32_t pending) {
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
  const auto next_id = [&]() -> uint32_t {
    if (lookup) return next_lookup();
    if (!speculative) return deps_.engine.step(r.sampling);
    if (burst_at == burst.size()) {
      const uint32_t k = policy ? policy->next() : deps_.engine.mtp_k();
      burst = deps_.engine.step_many(r.sampling, k);
      burst_at = 0;
      if (burst.empty()) throw std::runtime_error("EngineIface::step_many returned no ids");
      if (policy) policy->observe(k, std::min<uint32_t>(k, uint32_t(burst.size() - 1)));
    }
    return burst[burst_at++];
  };
  for (uint32_t generated = 0; generated < max_tokens; ++generated) {
    const uint32_t id = next_id();
    session.fed(id);
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
  } catch (const BadRequest& error) {
    response.status = 400;
    response.set_content(error_body(error.what(), "invalid_request_error"), "application/json");
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

}  // namespace server
