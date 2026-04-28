#include "ollama_client.h"

#include "ai/logger.h"
#include "ai/ollama.h"
#include "ollama_request_builder.h"
#include "ollama_response_parser.h"
#include "ollama_stream.h"

#include <memory>

namespace ai {
namespace ollama {

OllamaClient::OllamaClient(const std::string& base_url,
                            const std::string& api_key)
    : BaseProviderClient(
          providers::ProviderConfig{
              .api_key = api_key,
              .base_url = base_url,
              .completions_endpoint_path = "/v1/chat/completions",
              .embeddings_endpoint_path = "/v1/embeddings",
              .auth_header_name = "Authorization",
              .auth_header_prefix = "Bearer ",
              .extra_headers = {},
              .retry_config = {}},
          std::make_unique<OllamaRequestBuilder>(),
          std::make_unique<OllamaResponseParser>()) {
  ai::logger::log_debug("Ollama client initialized with base_url: {}", base_url);
}

OllamaClient::OllamaClient(const std::string& base_url,
                            const std::string& api_key,
                            const retry::RetryConfig& retry_config)
    : BaseProviderClient(
          providers::ProviderConfig{
              .api_key = api_key,
              .base_url = base_url,
              .completions_endpoint_path = "/v1/chat/completions",
              .embeddings_endpoint_path = "/v1/embeddings",
              .auth_header_name = "Authorization",
              .auth_header_prefix = "Bearer ",
              .extra_headers = {},
              .retry_config = retry_config},
          std::make_unique<OllamaRequestBuilder>(),
          std::make_unique<OllamaResponseParser>()) {
  ai::logger::log_debug(
      "Ollama client initialized with base_url: {} and custom retry config",
      base_url);
}

StreamResult OllamaClient::stream_text(const StreamOptions& options) {
  ai::logger::log_debug("Starting Ollama streaming - model: {}", options.model);

  auto request_json = request_builder_->build_request_json(options);
  request_json["stream"] = true;
  // Request usage stats in the final chunk (Ollama 0.4+)
  request_json["stream_options"]["include_usage"] = true;

  auto headers = request_builder_->build_headers(config_);
  headers.emplace("Accept", "text/event-stream");

  auto impl = std::make_unique<OllamaStreamImpl>();
  impl->start_stream(config_.base_url + config_.completions_endpoint_path,
                     headers, request_json);

  ai::logger::log_info("Ollama streaming started - model: {}", options.model);

  return StreamResult(std::move(impl));
}

bool OllamaClient::is_valid() const {
  return !config_.base_url.empty();
}

std::string OllamaClient::provider_name() const {
  return "ollama";
}

std::vector<std::string> OllamaClient::supported_models() const {
  // Ollama serves whatever models are installed locally — the real list is
  // only knowable via a live /api/tags call.  Return empty rather than an
  // arbitrarily incomplete snapshot; use supports_model() for validation.
  return {};
}

bool OllamaClient::supports_model(const std::string& /*model_name*/) const {
  // Any model installed in the Ollama instance is valid; we cannot enumerate
  // them without a live API call, so accept all
  return true;
}

std::string OllamaClient::config_info() const {
  return "Ollama (base_url: " + config_.base_url + ")";
}

std::string OllamaClient::default_model() const {
  return models::kDefaultModel;
}

}  // namespace ollama
}  // namespace ai
