#pragma once

#include "ai/retry/retry_policy.h"
#include "ai/types/stream_options.h"
#include "providers/base_provider_client.h"

#include <string>
#include <vector>

namespace ai {
namespace ollama {

class OllamaClient : public providers::BaseProviderClient {
 public:
  explicit OllamaClient(const std::string& base_url,
                        const std::string& api_key = "");

  explicit OllamaClient(const std::string& base_url,
                        const std::string& api_key,
                        const retry::RetryConfig& retry_config);

  StreamResult stream_text(const StreamOptions& options) override;

  // Valid as long as base_url is set; no API key required for local Ollama
  bool is_valid() const override;

  std::string provider_name() const override;
  std::vector<std::string> supported_models() const override;
  // Always true — Ollama serves any locally installed model
  bool supports_model(const std::string& model_name) const override;
  std::string config_info() const override;
  std::string default_model() const override;

  const std::string& get_base_url() const { return config_.base_url; }
};

}  // namespace ollama
}  // namespace ai
