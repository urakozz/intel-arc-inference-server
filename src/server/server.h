#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "server/adaptive_k.h"
#include "server/chat_format.h"
#include "server/deps.h"
#include "server/openai.h"
#include "server/prefix_cache.h"
#include "server/toolcall.h"

namespace httplib {
struct Request;
struct Response;
}  // namespace httplib

namespace server {

struct Options {
  std::string host = "0.0.0.0";
  int port = 8000;
  std::string served_model = "b70";
  size_t queue_depth = 4;
  bool tcp_nodelay = true;
  std::vector<uint32_t> eos_ids;
  // Spec 18d: the model's chat format (template variables, reasoning tags, tool-call syntax);
  // b70-serve sets it from config.json's model_type. The default is the Qwen path.
  ChatFormat chat_format;
  // Spec 20e: what a request that names no temperature / top_p / top_k samples with (parse_request's
  // defaults). b70-serve sets it for Kolibri-1 only, from generation_config.json (T 1.0, top-p 0.97,
  // top-k 128, sampled); unset = Sampling{} (greedy unless the request says otherwise), as before.
  std::optional<Sampling> sampling_defaults;
  // Spec 23e: generation_config.json's stop_strings (server::generation_stop_strings), applied to every request beside
  // its own `stop` (content only, as every stop string). b70-serve sets them from the snapshot's file.
  std::vector<std::string> stop_strings;
  // Non-empty: write DIR/NNNNNN.json per served request (spec 7 P0): timings, the
  // request body, prompt and generated ids, the generated text.
  std::string log_requests_dir;
  // Spec 7: the prefix cache's byte budget (0 = off: reset() and a full prefill per
  // request, exactly the server before it) and the pinned host allocator it draws from.
  size_t prefix_cache_bytes = 0;
  HostAlloc* prefix_alloc = nullptr;
  // Opt-in: chat requests feed the last prompt id by a decode replay so that the
  // prompt-end snapshot sits at len - 1 (PrefixSession::begin). Completions never do.
  bool prefix_split_last = false;
  // Spec 8 §10 (`--spec mtp`): with an MTP engine (mtp_k() > 0), choose each iteration's K
  // per request (server::AdaptiveK, K <= min(mtp_adaptive.max_k, mtp_k())). Off: every
  // iteration drafts mtp_k(), as before.
  bool mtp_auto = false;
  AdaptiveKOptions mtp_adaptive;
  // Spec 19e (`--spec lookup`): prompt-lookup speculative decoding. Each iteration's drafts
  // are the continuation of the longest earlier match (>= lookup_min_match ids) of the
  // request's own ids (server::PromptLookup over the prompt and every id kept so far),
  // verified by EngineIface::step_drafts; K per iteration from a per-request
  // server::AdaptiveK over lookup_adaptive (its cost table's drafts free: MtpCost::
  // with_free_drafts). Needs an engine with verify_k() > 0 and mtp_k() == 0 (one proposer
  // per server); the Server constructor refuses anything else. Off: nothing here runs.
  bool spec_lookup = false;
  uint32_t lookup_min_match = 3;
  // Seed each request's matcher with the generated ids of this many earlier requests (each
  // behind a PromptLookup::boundary(), before the prompt): what a client's re-rendered
  // history drops (reasoning) stays matchable. 0: the matcher holds the prompt only and is
  // reused across requests from the common prefix.
  uint32_t lookup_history = 0;
  AdaptiveKOptions lookup_adaptive;
  // Spec 19f (`--spec dflash-lookup`, spec 19 §13.1): prompt lookup inside DFlash. Each iteration first asks
  // server::LookupInDFlash (ENTER >= 8 ids beyond the drafter's window, STAY >= 3 kept) over the request's matcher
  // (indexed as --spec lookup's, lookup_history 0): its drafts, when it has any, are verified at K up to mtp_k() with
  // the drafter's draft pass skipped; otherwise the iteration is a DFlash one at K from mtp_auto's AdaptiveK (which
  // only DFlash iterations feed). Through EngineIface::step_drafts_or_many; needs an engine with mtp_k() > 0 and
  // dflash_window() > 0, and not spec_lookup - the Server constructor refuses anything else. Off: nothing here runs.
  bool spec_dflash_lookup = false;
  // How b70-serve runs the model on the cards, for the startup line ("1 card", "pp - ...", "replica - ..."); empty
  // when the family's own runner says it.
  std::string parallel_desc;
};

struct BatchEngineIface;
class Scheduler;

class Server {
 public:
  // Plan 13b (spec 13 §11, b70-serve --parallel replica): with `batch` (an engine of more than one slot, e.g.
  // cli::ReplicaEngine - one engine per card) every request goes through server::Scheduler: up to slots() at once,
  // each granted a slot, its prefix cache per slot (PrefixSlots over the one host store), --prefix-split-last,
  // MTP / prompt lookup per request, the request log, a client disconnect cancelling its request. `deps.engine`
  // is then not driven (only its properties are read). Null, or one slot: today's serial server, unchanged.
  Server(Deps deps, Options opts, BatchEngineIface* batch = nullptr);
  ~Server();

  bool listen();
  bool start();
  void stop();
  int bound_port() const;

 private:
  struct Outcome {
    std::string finish_reason;
    Usage usage;
    std::vector<uint32_t> prompt_ids;
    std::vector<uint32_t> out_ids;
    std::string text;      // the content text (completions: the raw text)
    ParsedOutput parsed;   // chat: reasoning, content and tool calls
    std::string raw_text;  // every generated piece, before any parsing or stop
    double t_first_token = 0;
    PrefixSession::Report prefix;
    int slot = -1;         // plan 13b: the slot (card) that served it under the scheduler; -1 on the serial path
  };
  struct Impl;

  Outcome generate(const Request& r, const std::function<void(const Delta&)>& emit);
  std::string render_prompt(const Request& r);
  // Plan 13b: the scheduler path - submit (throws BadRequest, or Busy when the queue is full), then wait for the
  // request's deltas; `emit` returns false when the client is gone (the request is then cancelled).
  struct BatchedRun;
  std::shared_ptr<BatchedRun> submit_batched(const Request& r);
  Outcome wait_batched(BatchedRun& run, const std::function<bool(const Delta&)>& emit);
  void cancel_batched(BatchedRun& run);
  void handle_batched(const httplib::Request& request, httplib::Response& response, const Request& parsed,
                      double t_start);
  void generate_tail(const Request& r, const std::string& prompt, uint32_t max_tokens,
                     Outcome& outcome, const std::function<void(const Delta&)>& emit);
  bool acquire(uint64_t& ticket);
  void release();
  bool bind();
  void handle(const httplib::Request& request, httplib::Response& response, bool chat);
  void log_request(bool chat, const std::string& body, double t_start, const Outcome& outcome);
  void finish_request();   // the request-end snapshot, after the last frame
  void lookup_begin(const std::vector<uint32_t>& prompt_ids);   // spec 19e
  void lookup_end(const std::vector<uint32_t>& out_ids);

  Deps deps_;
  Options opts_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace server
