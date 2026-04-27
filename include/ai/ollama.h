#pragma once

#ifndef AI_SDK_HAS_OLLAMA
#error \
    "Ollama component not available. Link with ai::ollama or ai::sdk to use Ollama functionality."
#endif

#include "retry/retry_policy.h"
#include "types/client.h"

#include <string>

namespace ai {
namespace ollama {

namespace models {
// Llama 3 family (Meta)
constexpr const char* kLlama32 = "llama3.2";
constexpr const char* kLlama32_1b = "llama3.2:1b";
constexpr const char* kLlama32_3b = "llama3.2:3b";
constexpr const char* kLlama33 = "llama3.3";
constexpr const char* kLlama33_70b = "llama3.3:70b";

// Gemma 3 family (Google)
constexpr const char* kGemma3_1b = "gemma3:1b";
constexpr const char* kGemma3_4b = "gemma3:4b";
constexpr const char* kGemma3_12b = "gemma3:12b";
constexpr const char* kGemma3_27b = "gemma3:27b";

// Qwen 3 family (Alibaba)
constexpr const char* kQwen3_0_6b = "qwen3:0.6b";
constexpr const char* kQwen3_1_7b = "qwen3:1.7b";
constexpr const char* kQwen3_4b = "qwen3:4b";
constexpr const char* kQwen3_8b = "qwen3:8b";
constexpr const char* kQwen3_14b = "qwen3:14b";
constexpr const char* kQwen3_32b = "qwen3:32b";

// DeepSeek R1 reasoning models
constexpr const char* kDeepseekR1_7b = "deepseek-r1:7b";
constexpr const char* kDeepseekR1_14b = "deepseek-r1:14b";
constexpr const char* kDeepseekR1_32b = "deepseek-r1:32b";
constexpr const char* kDeepseekR1_70b = "deepseek-r1:70b";

// Mistral family
constexpr const char* kMistral = "mistral";
constexpr const char* kMistralNemo = "mistral-nemo";
constexpr const char* kMistralSmall = "mistral-small3.1";

// Phi 4 family (Microsoft)
constexpr const char* kPhi4 = "phi4";
constexpr const char* kPhi4Mini = "phi4-mini";
constexpr const char* kPhi4Reasoning = "phi4-reasoning";

// Code-focused models
constexpr const char* kCodestral = "codestral";
constexpr const char* kCodeLlama = "codellama";
constexpr const char* kQwenCoder2_5 = "qwen2.5-coder";

// Embedding models
constexpr const char* kNomicEmbedText = "nomic-embed-text";
constexpr const char* kMxbaiEmbedLarge = "mxbai-embed-large";
constexpr const char* kAllMinilm = "all-minilm";

/// Default model used when none is specified
constexpr const char* kDefaultModel = kLlama32;
}  // namespace models

/// Create an Ollama client connecting to the default local endpoint.
/// No API key required. Connects to http://localhost:11434.
Client create_client();

/// Create an Ollama client with a custom base URL.
/// @param base_url Ollama endpoint (e.g. "http://192.168.1.100:11434" or
///                 "https://ollama.example.com" for a proxied deployment)
Client create_client(const std::string& base_url);

/// Create an Ollama client with a custom base URL and retry configuration.
Client create_client(const std::string& base_url,
                     const retry::RetryConfig& retry_config);

/// Create an Ollama client for authenticated remote deployments.
/// @param base_url Remote Ollama endpoint
/// @param api_key Bearer token used in the Authorization header
Client create_client_with_key(const std::string& base_url,
                              const std::string& api_key);

/// Create an Ollama client for authenticated remote deployments with retry.
Client create_client_with_key(const std::string& base_url,
                              const std::string& api_key,
                              const retry::RetryConfig& retry_config);

}  // namespace ollama
}  // namespace ai
