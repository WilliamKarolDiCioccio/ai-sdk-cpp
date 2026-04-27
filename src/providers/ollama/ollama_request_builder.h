#pragma once

#include "providers/openai/openai_request_builder.h"

namespace ai {
namespace ollama {

/// OpenAI-compatible request builder for Ollama.
///
/// Inherits all JSON serialization from OpenAIRequestBuilder and overrides:
///  - build_headers: omits Authorization when api_key is empty (local Ollama)
///  - build_request_json(GenerateOptions): remaps max_completion_tokens →
///    max_tokens for broader Ollama version compatibility
class OllamaRequestBuilder : public openai::OpenAIRequestBuilder {
 public:
  nlohmann::json build_request_json(const GenerateOptions& options) override;
  httplib::Headers build_headers(
      const providers::ProviderConfig& config) override;
};

}  // namespace ollama
}  // namespace ai
