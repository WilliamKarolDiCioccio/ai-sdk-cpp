#pragma once

#include "ai/types/stream_result.h"

#include <atomic>
#include <concurrentqueue.h>
#include <httplib.h>
#include <mutex>
#include <thread>

#include <nlohmann/json.hpp>

namespace ai {
namespace ollama {

/// SSE stream implementation for Ollama's OpenAI-compatible streaming endpoint.
///
/// Supports both plain HTTP (default local Ollama) and HTTPS (proxied/remote
/// deployments) by inspecting the URL scheme at stream start.
class OllamaStreamImpl : public internal::StreamResultImpl {
 public:
  OllamaStreamImpl() = default;
  ~OllamaStreamImpl();

  OllamaStreamImpl(const OllamaStreamImpl&) = delete;
  OllamaStreamImpl& operator=(const OllamaStreamImpl&) = delete;
  OllamaStreamImpl(OllamaStreamImpl&&) = delete;
  OllamaStreamImpl& operator=(OllamaStreamImpl&&) = delete;

  void start_stream(const std::string& url,
                    const httplib::Headers& headers,
                    const nlohmann::json& request_body);

  StreamEvent get_next_event() override;
  bool has_more_events() const override;
  void stop_stream() override;

 private:
  void run_stream(const std::string& url,
                  const httplib::Headers& headers,
                  const nlohmann::json& request_body);
  void parse_sse_line(const std::string& line);
  void push_event(StreamEvent event);
  void push_finish_event_if_needed();
  void mark_complete();

  StreamEvent create_error_event(const std::string& message);
  FinishReason parse_finish_reason(const std::string& reason_str);
  Usage parse_usage(const nlohmann::json& usage_json);

  moodycamel::ConcurrentQueue<StreamEvent> event_queue_;
  std::thread stream_thread_;
  std::mutex thread_mutex_;
  std::atomic<bool> is_complete_{false};
  std::atomic<bool> should_stop_{false};
  std::atomic<bool> finish_event_pushed_{false};

  // Deferred finish state — Ollama sends finish_reason and usage in separate
  // consecutive chunks when stream_options.include_usage is true.
  std::optional<FinishReason> pending_finish_reason_;
  std::optional<Usage> pending_usage_;
};

}  // namespace ollama
}  // namespace ai
