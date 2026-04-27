#include "ollama_factory.h"

#include "ollama_client.h"

#include <memory>

namespace ai {
namespace ollama {

namespace {
constexpr const char* kDefaultBaseUrl = "http://localhost:11434";
}  // namespace

Client create_client() {
  return Client(std::make_unique<OllamaClient>(kDefaultBaseUrl));
}

Client create_client(const std::string& base_url) {
  return Client(std::make_unique<OllamaClient>(
      base_url.empty() ? kDefaultBaseUrl : base_url));
}

Client create_client(const std::string& base_url,
                     const retry::RetryConfig& retry_config) {
  return Client(std::make_unique<OllamaClient>(
      base_url.empty() ? kDefaultBaseUrl : base_url, "", retry_config));
}

Client create_client_with_key(const std::string& base_url,
                              const std::string& api_key) {
  return Client(std::make_unique<OllamaClient>(base_url, api_key));
}

Client create_client_with_key(const std::string& base_url,
                              const std::string& api_key,
                              const retry::RetryConfig& retry_config) {
  return Client(std::make_unique<OllamaClient>(base_url, api_key, retry_config));
}

}  // namespace ollama
}  // namespace ai
