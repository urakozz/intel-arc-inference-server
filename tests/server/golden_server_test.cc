#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <spawn.h>

extern char** environ;

namespace {

using json = nlohmann::json;
constexpr uint32_t kGenerated = 32;
const char* const kPrompts[] = {"prose", "code", "cjk"};

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

std::string read_text(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) fail("cannot open '" + path + "'");
  std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  while (!text.empty() && text.back() == '\n') text.pop_back();
  return text;
}

uint32_t parse_id(const std::string& word, const std::string& path) {
  if (word.empty() || word.find_first_not_of("0123456789") != std::string::npos)
    fail("invalid id in '" + path + "': '" + word + "'");
  const unsigned long long value = std::stoull(word);
  if (value > 0xFFFFFFFFull) fail("out-of-range id in '" + path + "': '" + word + "'");
  return static_cast<uint32_t>(value);
}

std::vector<uint32_t> read_ids(const std::string& path) {
  std::ifstream input(path);
  if (!input) fail("cannot open '" + path + "'");
  std::vector<uint32_t> ids;
  std::string word;
  while (input >> word) ids.push_back(parse_id(word, path));
  if (!input.eof()) fail("cannot parse ids in '" + path + "'");
  if (ids.empty()) fail("no ids in '" + path + "'");
  return ids;
}

std::string id_list(const std::vector<uint32_t>& ids) {
  std::ostringstream output;
  output << '[';
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i != 0) output << ", ";
    output << ids[i];
  }
  output << ']';
  return output.str();
}

void require_equal(const std::string& prompt, const char* label, const std::vector<uint32_t>& actual,
                   const std::vector<uint32_t>& expected) {
  if (actual == expected) return;
  std::fprintf(stderr, "%s %s mismatch\n  actual:   %s\n  expected: %s\n", prompt.c_str(), label,
               id_list(actual).c_str(), id_list(expected).c_str());
  fail(prompt + " " + label + " mismatch");
}

uint16_t free_port() {
  const int socket_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (socket_fd < 0) fail("socket: " + std::string(std::strerror(errno)));
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  if (::bind(socket_fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
    const std::string error = std::strerror(errno);
    ::close(socket_fd);
    fail("bind: " + error);
  }
  socklen_t length = sizeof(address);
  if (::getsockname(socket_fd, reinterpret_cast<sockaddr*>(&address), &length) != 0) {
    const std::string error = std::strerror(errno);
    ::close(socket_fd);
    fail("getsockname: " + error);
  }
  ::close(socket_fd);
  return ntohs(address.sin_port);
}

pid_t spawn(const std::vector<std::string>& args, posix_spawn_file_actions_t* actions = nullptr) {
  std::vector<char*> argv;
  argv.reserve(args.size() + 1);
  for (const std::string& arg : args) argv.push_back(const_cast<char*>(arg.c_str()));
  argv.push_back(nullptr);
  pid_t pid = -1;
  const int result = posix_spawn(&pid, argv.front(), actions, nullptr, argv.data(), environ);
  if (result != 0) fail("posix_spawn " + args.front() + ": " + std::strerror(result));
  return pid;
}

void stop(pid_t pid) {
  if (pid <= 0) return;
  if (::kill(pid, SIGTERM) != 0 && errno != ESRCH)
    fail("SIGTERM " + std::to_string(pid) + ": " + std::strerror(errno));
  int status = 0;
  while (::waitpid(pid, &status, 0) < 0) {
    if (errno != EINTR) fail("waitpid " + std::to_string(pid) + ": " + std::strerror(errno));
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
    fail("server exited unsuccessfully after SIGTERM");
}

void wait_ready(uint16_t port) {
  httplib::Client client("127.0.0.1", port);
  for (uint32_t attempt = 0; attempt < 240; ++attempt) {
    const auto response = client.Get("/v1/models");
    if (response && response->status == 200) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
  fail("b70-serve did not answer GET /v1/models within 120 seconds");
}

std::vector<uint32_t> run_decode(const std::string& decode, const std::string& snapshot,
                                 const std::string& ids_path) {
  int output[2] = {-1, -1};
  if (::pipe(output) != 0) fail("pipe: " + std::string(std::strerror(errno)));
  posix_spawn_file_actions_t actions;
  if (posix_spawn_file_actions_init(&actions) != 0) fail("posix_spawn_file_actions_init failed");
  const int dup_result = posix_spawn_file_actions_adddup2(&actions, output[1], STDOUT_FILENO);
  const int close_read = posix_spawn_file_actions_addclose(&actions, output[0]);
  const int close_write = posix_spawn_file_actions_addclose(&actions, output[1]);
  if (dup_result != 0 || close_read != 0 || close_write != 0) {
    posix_spawn_file_actions_destroy(&actions);
    ::close(output[0]);
    ::close(output[1]);
    fail("posix_spawn_file_actions setup failed");
  }
  const pid_t pid = spawn({decode, snapshot, "--ids", ids_path, "--n", "32"}, &actions);
  posix_spawn_file_actions_destroy(&actions);
  ::close(output[1]);
  std::string text;
  char buffer[4096];
  for (;;) {
    const ssize_t count = ::read(output[0], buffer, sizeof(buffer));
    if (count > 0) {
      text.append(buffer, static_cast<size_t>(count));
    } else if (count == 0) {
      break;
    } else if (errno != EINTR) {
      ::close(output[0]);
      fail("read b70-decode: " + std::string(std::strerror(errno)));
    }
  }
  ::close(output[0]);
  int status = 0;
  while (::waitpid(pid, &status, 0) < 0) {
    if (errno != EINTR) fail("waitpid b70-decode: " + std::string(std::strerror(errno)));
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) fail("b70-decode failed for '" + ids_path + "'");

  std::istringstream input(text);
  std::vector<uint32_t> ids;
  std::string word;
  while (input >> word) ids.push_back(parse_id(word, "b70-decode stdout"));
  if (!input.eof()) fail("b70-decode stdout is not ids");
  return ids;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: golden_server_test <b70-serve> <b70-decode> <prompts-dir> <snapshot>\n");
    return 2;
  }

  pid_t server = -1;
  try {
    const std::string serve = argv[1];
    const std::string decode = argv[2];
    const std::string prompts = argv[3];
    const std::string snapshot = argv[4];
    const uint16_t port = free_port();
    server = spawn({serve, snapshot, "--host", "127.0.0.1", "--port", std::to_string(port)});
    wait_ready(port);
    httplib::Client client("127.0.0.1", port);

    for (const char* prompt_name : kPrompts) {
      const std::string name = prompt_name;
      const std::string ids_path = prompts + "/" + name + ".ids";
      const std::vector<uint32_t> expected_prompt = read_ids(ids_path);
      const json request = {{"prompt", read_text(prompts + "/" + name + ".txt")},
                            {"max_tokens", kGenerated},
                            {"return_token_ids", true}};
      const auto response = client.Post("/v1/completions", request.dump(), "application/json");
      if (!response) fail(name + " /v1/completions did not return a response");
      if (response->status != 200)
        fail(name + " /v1/completions status " + std::to_string(response->status) + ": " + response->body);
      const json body = json::parse(response->body);
      const std::vector<uint32_t> prompt_ids = body.at("prompt_token_ids").get<std::vector<uint32_t>>();
      const std::vector<uint32_t> served_ids =
          body.at("choices").at(0).at("token_ids").get<std::vector<uint32_t>>();
      require_equal(name, "prompt ids", prompt_ids, expected_prompt);
      if (served_ids.size() > kGenerated) fail(name + " server returned more than 32 generated ids");
      if (served_ids.size() < kGenerated)
        std::printf("%s: stopped on EOS after %zu/%u generated ids\n", name.c_str(), served_ids.size(), kGenerated);

      const std::vector<uint32_t> decoded_ids = run_decode(decode, snapshot, ids_path);
      if (decoded_ids.size() != kGenerated)
        fail(name + " b70-decode returned " + std::to_string(decoded_ids.size()) + " ids, expected 32");
      const size_t compared = std::min(served_ids.size(), decoded_ids.size());
      require_equal(name, "generated ids", std::vector<uint32_t>(served_ids.begin(), served_ids.begin() + compared),
                    std::vector<uint32_t>(decoded_ids.begin(), decoded_ids.begin() + compared));
      std::printf("%s: prompt ids identical, %zu/%u generated ids identical\n", name.c_str(), compared, kGenerated);
    }

    stop(server);
    return 0;
  } catch (const std::exception& error) {
    const std::string message = error.what();
    try {
      stop(server);
    } catch (const std::exception& cleanup_error) {
      std::fprintf(stderr, "golden_server_test cleanup: %s\n", cleanup_error.what());
    }
    std::fprintf(stderr, "golden_server_test: %s\n", message.c_str());
    return 1;
  }
}
