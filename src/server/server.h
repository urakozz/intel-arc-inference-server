#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "server/deps.h"
#include "server/openai.h"

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
};

class Server {
 public:
  Server(Deps deps, Options opts);
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
    std::string text;
  };
  struct Impl;

  Outcome generate(const Request& r, const std::function<void(const std::string&)>& emit);
  bool acquire(uint64_t& ticket);
  void release();
  bool bind();
  void handle(const httplib::Request& request, httplib::Response& response, bool chat);

  Deps deps_;
  Options opts_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace server
