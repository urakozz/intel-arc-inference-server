#include "server/server.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <set>
#include <thread>
#include <utility>

#include <httplib.h>

namespace server {
namespace {

uint64_t unix_time() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
                                    std::chrono::system_clock::now().time_since_epoch())
                                    .count());
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
  uint64_t created = unix_time();
};

Server::Server(Deps deps, Options opts)
    : deps_(deps), opts_(std::move(opts)), impl_(std::make_unique<Impl>()) {
  impl_->eos.insert(opts_.eos_ids.begin(), opts_.eos_ids.end());
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
                                 const std::function<void(const std::string&)>& emit) {
  Outcome outcome;
  std::string prompt;
  try {
    prompt = r.chat ? deps_.tmpl.render(r.messages, r.tools, r.enable_thinking) : r.prompt;
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

  deps_.engine.reset();
  deps_.engine.prefill(outcome.prompt_ids);
  std::unique_ptr<StreamerIface> streamer = deps_.tok.streamer();
  if (!streamer) throw BadRequest("tokenizer did not provide a streamer");

  size_t hold = 0;
  for (const std::string& stop : r.stop) {
    hold = std::max(hold, stop.empty() ? 0U : stop.size() - 1);
  }
  std::string pending;
  const auto flush_pending = [&](bool final) {
    const size_t keep = final ? 0 : std::min(hold, pending.size());
    if (pending.size() <= keep) return;
    const std::string piece = pending.substr(0, pending.size() - keep);
    outcome.text += piece;
    emit(piece);
    pending.erase(0, piece.size());
  };

  outcome.finish_reason = "length";
  for (uint32_t generated = 0; generated < max_tokens; ++generated) {
    const uint32_t id = deps_.engine.step(r.sampling);
    const bool is_eos = impl_->eos.count(id) != 0;
    if (is_eos && !r.ignore_eos && outcome.out_ids.size() >= r.min_tokens) {
      outcome.finish_reason = "stop";
      break;
    }
    outcome.out_ids.push_back(id);
    if (!is_eos) pending += streamer->push(id);

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
  if (outcome.finish_reason == "length") {
    pending += streamer->flush();
    flush_pending(true);
  }
  outcome.usage.completion_tokens = static_cast<uint32_t>(outcome.out_ids.size());
  return outcome;
}

void Server::handle(const httplib::Request& request, httplib::Response& response, bool chat) {
  Request parsed;
  try {
    parsed = parse_request(request.body, chat);
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
      const Outcome outcome = generate(parsed, [](const std::string&) {});
      const auto* prompt_ids = parsed.return_token_ids ? &outcome.prompt_ids : nullptr;
      const auto* out_ids = parsed.return_token_ids ? &outcome.out_ids : nullptr;
      const std::string id = request_id(parsed.chat, ++impl_->next_id);
      response.set_content(completion_body(parsed, id, unix_time(), outcome.text,
                                           outcome.finish_reason, outcome.usage, prompt_ids, out_ids),
                           "application/json");
    } catch (const BadRequest& error) {
      response.status = 400;
      response.set_content(error_body(error.what(), "invalid_request_error"), "application/json");
    }
    release();
    return;
  }

  response.set_header("Cache-Control", "no-cache");
  response.set_header("X-Accel-Buffering", "no");
  response.set_chunked_content_provider(
      "text/event-stream",
      [this, parsed](size_t, httplib::DataSink& sink) mutable {
        const std::string id = request_id(parsed.chat, ++impl_->next_id);
        const uint64_t created = unix_time();
        const auto write = [&sink](const std::string& frame) {
          return sink.write(frame.data(), frame.size());
        };
        if (parsed.chat) (void)write(stream_frame_role(parsed, id, created));
        try {
          const Outcome outcome = generate(parsed, [&](const std::string& piece) {
            if (!piece.empty()) (void)write(stream_frame_text(parsed, id, created, piece));
          });
          (void)write(stream_frame_finish(parsed, id, created, outcome.finish_reason));
          if (parsed.include_usage) (void)write(stream_frame_usage(parsed, id, created, outcome.usage));
        } catch (const BadRequest& error) {
          (void)write(std::string("data: ") +
                      error_body(error.what(), "invalid_request_error") + "\n\n");
        }
        (void)write(kDone);
        sink.done();
        release();
        return true;
      });
}

}  // namespace server
