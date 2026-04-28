#include "ollama_response_parser.h"

#include "ai/logger.h"

namespace ai {
namespace ollama {

std::string OllamaResponseParser::extractJsonBlockFromReasoning(
    const std::string& reasoning)
{
    // Find the last ```json fence — thinking models usually place the final
    // clean answer in the last code block after the chain-of-thought.
    auto pos = reasoning.rfind("```json");
    if (pos != std::string::npos)
    {
        pos += 7; // skip "```json"
        while (pos < reasoning.size() &&
               (reasoning[pos] == '\n' || reasoning[pos] == '\r'))
            ++pos;
        auto end = reasoning.find("```", pos);
        return reasoning.substr(pos, end == std::string::npos
                                         ? reasoning.size() - pos
                                         : end - pos);
    }

    // Fallback: find the last top-level '{' and take everything up to
    // any trailing fence (handles models that emit bare JSON in reasoning).
    auto brace = reasoning.rfind('{');
    if (brace == std::string::npos) return {};
    auto extracted = reasoning.substr(brace);
    auto fence = extracted.find("```");
    if (fence != std::string::npos) extracted = extracted.substr(0, fence);
    return extracted;
}

GenerateResult OllamaResponseParser::parse_success_completion_response(
    const nlohmann::json& response)
{
    GenerateResult result = OpenAIResponseParser::parse_success_completion_response(response);

    if (!result.text.empty()) return result;

    // Content was empty — check for reasoning field (thinking-mode model).
    try
    {
        if (!response.contains("choices") || response["choices"].empty())
            return result;
        auto& msg = response["choices"][0]["message"];
        if (!msg.contains("reasoning") || !msg["reasoning"].is_string())
            return result;
        auto reasoning = msg["reasoning"].get<std::string>();
        if (reasoning.empty()) return result;

        result.text = extractJsonBlockFromReasoning(reasoning);
        if (!result.text.empty())
            ai::logger::log_debug(
                "OllamaResponseParser: recovered {} chars from reasoning field",
                result.text.size());
    }
    catch (...) {}

    return result;
}

}  // namespace ollama
}  // namespace ai
