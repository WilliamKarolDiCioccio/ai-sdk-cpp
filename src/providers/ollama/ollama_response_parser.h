#pragma once

#include "providers/openai/openai_response_parser.h"

namespace ai {
namespace ollama {

// Extends the OpenAI parser to handle thinking-mode models (e.g. Gemma4).
// When the /v1/chat/completions response has an empty "content" but a
// non-empty "reasoning" field, the model placed its answer inside reasoning.
// This parser extracts the last ```json...``` block from reasoning and
// promotes it to result.text so callers see a normal response.
class OllamaResponseParser : public openai::OpenAIResponseParser {
 public:
  GenerateResult parse_success_completion_response(
      const nlohmann::json& response) override;

 private:
  static std::string extractJsonBlockFromReasoning(const std::string& reasoning);
};

}  // namespace ollama
}  // namespace ai
