#pragma once

#include "ai/retry/retry_policy.h"
#include "ai/types/client.h"

#include <string>

namespace ai {
namespace ollama {

Client create_client();
Client create_client(const std::string& base_url);
Client create_client(const std::string& base_url,
                     const retry::RetryConfig& retry_config);
Client create_client_with_key(const std::string& base_url,
                              const std::string& api_key);
Client create_client_with_key(const std::string& base_url,
                              const std::string& api_key,
                              const retry::RetryConfig& retry_config);

}  // namespace ollama
}  // namespace ai
