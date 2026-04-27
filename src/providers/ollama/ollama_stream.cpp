#include "ollama_stream.h"

#include "ai/logger.h"

#include <chrono>
#include <mutex>

namespace {
constexpr auto kEventTimeout = static_cast<std::chrono::seconds>(30);
constexpr auto kSleepInterval = std::chrono::milliseconds(1);
constexpr auto kConnectionTimeout = 30;  // seconds
constexpr auto kReadTimeout = 300;       // 5 minutes for long generations
}  // namespace

namespace ai {
namespace ollama {

OllamaStreamImpl::~OllamaStreamImpl() {
  stop_stream();
}

void OllamaStreamImpl::start_stream(const std::string& url,
                                    const httplib::Headers& headers,
                                    const nlohmann::json& request_body) {
  ai::logger::log_debug("Starting Ollama stream - URL: {}", url);

  std::lock_guard<std::mutex> lock(thread_mutex_);

  if (stream_thread_.joinable()) {
    return;
  }

  should_stop_ = false;
  is_complete_ = false;
  finish_event_pushed_ = false;

  ai::logger::log_info("Launching Ollama stream thread");

  stream_thread_ = std::thread([this, url, headers, request_body]() {
    run_stream(url, headers, request_body);
  });
}

StreamEvent OllamaStreamImpl::get_next_event() {
  StreamEvent event("");
  auto start_time = std::chrono::steady_clock::now();

  while (!event_queue_.try_dequeue(event)) {
    if (is_complete_ && event_queue_.size_approx() == 0) {
      return StreamEvent("");
    }

    if (std::chrono::steady_clock::now() - start_time > kEventTimeout) {
      ai::logger::log_error("Timeout waiting for Ollama stream event after {} seconds",
                            kEventTimeout.count());
      return StreamEvent(kStreamEventTypeError, "Timeout waiting for next event");
    }

    std::this_thread::sleep_for(kSleepInterval);
  }

  return event;
}

bool OllamaStreamImpl::has_more_events() const {
  return event_queue_.size_approx() > 0 || !is_complete_;
}

void OllamaStreamImpl::stop_stream() {
  should_stop_ = true;
  std::lock_guard<std::mutex> lock(thread_mutex_);
  if (stream_thread_.joinable()) {
    stream_thread_.join();
  }
}

void OllamaStreamImpl::run_stream(const std::string& url,
                                  const httplib::Headers& headers,
                                  const nlohmann::json& request_body) {
  // Parse scheme://host[:port]/path
  std::string scheme = "http";
  std::string_view url_view(url);

  if (auto pos = url_view.find("://"); pos != std::string_view::npos) {
    scheme = std::string(url_view.substr(0, pos));
    url_view.remove_prefix(pos + 3);
  }

  auto slash_pos = url_view.find('/');
  std::string host(url_view.substr(0, slash_pos));
  std::string path = (slash_pos != std::string_view::npos)
                         ? std::string(url_view.substr(slash_pos))
                         : "/v1/chat/completions";

  ai::logger::log_debug("Ollama stream: {}://{}{}", scheme, host, path);

  // Shared request execution logic; works with both httplib::Client and
  // httplib::SSLClient since both expose the same send() interface.
  auto execute_request = [&](auto& client) {
    client.set_connection_timeout(kConnectionTimeout);
    client.set_read_timeout(kReadTimeout);

    std::string accumulated_data;

    httplib::Request req;
    req.method = "POST";
    req.path = path;
    req.headers = headers;
    req.body = request_body.dump();
    req.set_header("Content-Type", "application/json");

    req.content_receiver = [this, &accumulated_data](
                               const char* data, size_t data_length,
                               uint64_t /*offset*/,
                               uint64_t /*total_length*/) {
      accumulated_data.append(data, data_length);

      size_t pos = 0;
      while ((pos = accumulated_data.find('\n')) != std::string::npos) {
        std::string line = accumulated_data.substr(0, pos);
        accumulated_data.erase(0, pos + 1);

        if (!line.empty() && line.back() == '\r') {
          line.pop_back();
        }

        if (should_stop_) {
          return false;
        }

        parse_sse_line(line);
      }

      return true;
    };

    httplib::Response res;
    httplib::Error error;

    if (!client.send(req, res, error)) {
      push_event(
          create_error_event("Network error: " + httplib::to_string(error)));
    } else if (res.status != 200) {
      ai::logger::log_error("Ollama stream returned HTTP {} - body: {}",
                            res.status, res.body);
      push_event(create_error_event("HTTP " + std::to_string(res.status) +
                                    " error: " + res.body));
    }
  };

  try {
    if (scheme == "https") {
      httplib::SSLClient client(host);
      client.enable_server_certificate_verification(true);
      execute_request(client);
    } else {
      httplib::Client client(host);
      execute_request(client);
    }
  } catch (const std::exception& e) {
    ai::logger::log_error("Exception in Ollama stream thread: {}", e.what());
    push_event(create_error_event(e.what()));
  }

  mark_complete();
}

void OllamaStreamImpl::parse_sse_line(const std::string& line) {
  if (!line.starts_with("data: ")) {
    return;
  }

  auto data = line.substr(6);

  if (data == "[DONE]") {
    ai::logger::log_debug("Ollama stream received [DONE]");
    push_finish_event_if_needed();
    mark_complete();
    return;
  }

  try {
    auto json = nlohmann::json::parse(data);
    auto& choices = json["choices"];

    if (!choices.empty() && choices[0].contains("delta")) {
      auto& delta = choices[0]["delta"];
      if (delta.contains("content") && !delta["content"].is_null()) {
        push_event(StreamEvent(delta["content"].get<std::string>()));
      }
    }

    // Capture finish_reason; don't push yet — usage arrives in the next chunk
    // when stream_options.include_usage is set.
    if (!choices.empty() && choices[0].contains("finish_reason") &&
        !choices[0]["finish_reason"].is_null()) {
      pending_finish_reason_ =
          parse_finish_reason(choices[0]["finish_reason"].get<std::string>());
    }

    // Capture usage (may arrive in the same chunk as finish_reason, or later).
    if (json.contains("usage") && !json["usage"].is_null()) {
      pending_usage_ = parse_usage(json["usage"]);
    }

    // Push finish event once both pieces are available.
    if (pending_finish_reason_.has_value() && pending_usage_.has_value()) {
      push_finish_event_if_needed();
    }
  } catch (const std::exception& e) {
    ai::logger::log_error("Failed to parse Ollama SSE line: {}", e.what());
  }
}

void OllamaStreamImpl::push_event(StreamEvent event) {
  event_queue_.enqueue(std::move(event));
}

void OllamaStreamImpl::push_finish_event_if_needed() {
  bool expected = false;
  if (!finish_event_pushed_.compare_exchange_strong(expected, true)) {
    return;
  }
  if (pending_finish_reason_.has_value() && pending_usage_.has_value()) {
    event_queue_.enqueue(StreamEvent(kStreamEventTypeFinish, *pending_usage_,
                                     *pending_finish_reason_));
  } else {
    event_queue_.enqueue(StreamEvent(kStreamEventTypeFinish));
  }
}

void OllamaStreamImpl::mark_complete() {
  is_complete_ = true;
}

StreamEvent OllamaStreamImpl::create_error_event(const std::string& message) {
  return StreamEvent(kStreamEventTypeError, message);
}

FinishReason OllamaStreamImpl::parse_finish_reason(
    const std::string& reason_str) {
  if (reason_str == "stop") return kFinishReasonStop;
  if (reason_str == "length") return kFinishReasonLength;
  if (reason_str == "content_filter") return kFinishReasonContentFilter;
  return kFinishReasonStop;
}

Usage OllamaStreamImpl::parse_usage(const nlohmann::json& usage_json) {
  Usage usage;
  usage.prompt_tokens = usage_json.value("prompt_tokens", 0);
  usage.completion_tokens = usage_json.value("completion_tokens", 0);
  usage.total_tokens = usage_json.value("total_tokens", 0);
  return usage;
}

}  // namespace ollama
}  // namespace ai
