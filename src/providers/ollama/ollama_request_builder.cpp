#include "ollama_request_builder.h"

namespace ai {
namespace ollama {

nlohmann::json OllamaRequestBuilder::build_request_json(
    const GenerateOptions& options) {
  auto request = openai::OpenAIRequestBuilder::build_request_json(options);
  // Remap OpenAI's newer field name to the one Ollama accepts across all versions
  if (request.contains("max_completion_tokens")) {
    request["max_tokens"] = request["max_completion_tokens"];
    request.erase("max_completion_tokens");
  }
  return request;
}

httplib::Headers OllamaRequestBuilder::build_headers(
    const providers::ProviderConfig& config) {
  httplib::Headers headers;
  // Only emit Authorization when a key is explicitly configured
  if (!config.api_key.empty()) {
    headers.emplace(config.auth_header_name,
                    config.auth_header_prefix + config.api_key);
  }
  for (const auto& [key, value] : config.extra_headers) {
    headers.emplace(key, value);
  }
  return headers;
}

}  // namespace ollama
}  // namespace ai
