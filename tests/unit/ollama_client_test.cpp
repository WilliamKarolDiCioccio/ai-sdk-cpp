#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "ai/ollama.h"
#include "ai/types/generate_options.h"
#include "ai/types/stream_options.h"
#include "providers/ollama/ollama_client.h"
#include "providers/ollama/ollama_request_builder.h"

namespace ai {
namespace test {

// ---------------------------------------------------------------------------
// OllamaClient unit tests
// ---------------------------------------------------------------------------

class OllamaClientTest : public ::testing::Test {
 protected:
  static constexpr const char* kLocalUrl = "http://localhost:11434";
  static constexpr const char* kRemoteUrl = "https://ollama.example.com";
};

TEST_F(OllamaClientTest, ValidWithBaseUrlNoApiKey) {
  ollama::OllamaClient client(kLocalUrl);
  EXPECT_TRUE(client.is_valid());
}

TEST_F(OllamaClientTest, ValidWithBaseUrlAndApiKey) {
  ollama::OllamaClient client(kRemoteUrl, "some-bearer-token");
  EXPECT_TRUE(client.is_valid());
}

TEST_F(OllamaClientTest, InvalidWithEmptyBaseUrl) {
  ollama::OllamaClient client("");
  EXPECT_FALSE(client.is_valid());
}

TEST_F(OllamaClientTest, ProviderNameIsOllama) {
  ollama::OllamaClient client(kLocalUrl);
  EXPECT_EQ(client.provider_name(), "ollama");
}

TEST_F(OllamaClientTest, ConfigInfoContainsBaseUrl) {
  ollama::OllamaClient client(kLocalUrl);
  EXPECT_THAT(client.config_info(), testing::HasSubstr("Ollama"));
  EXPECT_THAT(client.config_info(), testing::HasSubstr(kLocalUrl));
}

TEST_F(OllamaClientTest, DefaultModelIsSet) {
  ollama::OllamaClient client(kLocalUrl);
  EXPECT_FALSE(client.default_model().empty());
  EXPECT_EQ(client.default_model(), ollama::models::kDefaultModel);
}

TEST_F(OllamaClientTest, SupportedModelsEmpty) {
  // Ollama serves whatever is installed locally; the list is unknowable without
  // a live API call, so supported_models() returns empty by design.
  ollama::OllamaClient client(kLocalUrl);
  EXPECT_TRUE(client.supported_models().empty());
}

TEST_F(OllamaClientTest, SupportsAnyModel) {
  // Ollama accepts any locally installed model — always returns true
  ollama::OllamaClient client(kLocalUrl);
  EXPECT_TRUE(client.supports_model("llama3.2"));
  EXPECT_TRUE(client.supports_model("gemma4:26b"));
  EXPECT_TRUE(client.supports_model("some-future-model:latest"));
  EXPECT_TRUE(client.supports_model(""));
}

TEST_F(OllamaClientTest, GetBaseUrlAccessor) {
  ollama::OllamaClient client(kLocalUrl);
  EXPECT_EQ(client.get_base_url(), kLocalUrl);
}

// ---------------------------------------------------------------------------
// OllamaRequestBuilder unit tests
// ---------------------------------------------------------------------------

class OllamaRequestBuilderTest : public ::testing::Test {
 protected:
  ollama::OllamaRequestBuilder builder_;

  providers::ProviderConfig make_config(const std::string& api_key = "") {
    return providers::ProviderConfig{
        .api_key = api_key,
        .base_url = "http://localhost:11434",
        .completions_endpoint_path = "/v1/chat/completions",
        .embeddings_endpoint_path = "/v1/embeddings",
        .auth_header_name = "Authorization",
        .auth_header_prefix = "Bearer ",
        .extra_headers = {},
        .retry_config = {}};
  }
};

TEST_F(OllamaRequestBuilderTest, BuildHeadersNoAuthWhenKeyEmpty) {
  auto headers = builder_.build_headers(make_config(""));
  EXPECT_EQ(headers.find("Authorization"), headers.end());
}

TEST_F(OllamaRequestBuilderTest, BuildHeadersWithAuthWhenKeyProvided) {
  auto headers = builder_.build_headers(make_config("my-secret-token"));
  auto it = headers.find("Authorization");
  ASSERT_NE(it, headers.end());
  EXPECT_EQ(it->second, "Bearer my-secret-token");
}

TEST_F(OllamaRequestBuilderTest, ExtraHeadersAreForwarded) {
  auto config = make_config("");
  config.extra_headers.emplace("X-Custom", "value");
  auto headers = builder_.build_headers(config);
  auto it = headers.find("X-Custom");
  ASSERT_NE(it, headers.end());
  EXPECT_EQ(it->second, "value");
}

TEST_F(OllamaRequestBuilderTest, MaxTokensRemappedFromMaxCompletionTokens) {
  GenerateOptions opts("llama3.2", "hello");
  opts.max_tokens = 128;

  auto json = builder_.build_request_json(opts);

  // Ollama expects max_tokens, not the newer OpenAI field name
  EXPECT_TRUE(json.contains("max_tokens"));
  EXPECT_EQ(json["max_tokens"], 128);
  EXPECT_FALSE(json.contains("max_completion_tokens"));
}

TEST_F(OllamaRequestBuilderTest, MaxTokensAbsentWhenNotSet) {
  GenerateOptions opts("llama3.2", "hello");
  // max_tokens not set

  auto json = builder_.build_request_json(opts);

  EXPECT_FALSE(json.contains("max_tokens"));
  EXPECT_FALSE(json.contains("max_completion_tokens"));
}

TEST_F(OllamaRequestBuilderTest, ModelAndPromptPresentInRequest) {
  GenerateOptions opts("gemma4:26b", "test prompt");
  auto json = builder_.build_request_json(opts);

  EXPECT_EQ(json["model"], "gemma4:26b");
  EXPECT_TRUE(json.contains("messages"));
  EXPECT_FALSE(json["messages"].empty());
}

TEST_F(OllamaRequestBuilderTest, SystemPromptIncludedInMessages) {
  GenerateOptions opts("llama3.2", "You are helpful.", "What is 2+2?");
  auto json = builder_.build_request_json(opts);

  auto& msgs = json["messages"];
  ASSERT_GE(msgs.size(), 2u);
  EXPECT_EQ(msgs[0]["role"], "system");
  EXPECT_EQ(msgs[0]["content"], "You are helpful.");
  EXPECT_EQ(msgs[1]["role"], "user");
  EXPECT_EQ(msgs[1]["content"], "What is 2+2?");
}

// ---------------------------------------------------------------------------
// Factory function unit tests
// ---------------------------------------------------------------------------

TEST(OllamaFactoryTest, CreateClientDefaultsToLocalhost) {
  auto client = ollama::create_client();
  EXPECT_EQ(client.provider_name(), "ollama");
  EXPECT_TRUE(client.is_valid());
}

TEST(OllamaFactoryTest, CreateClientWithCustomUrl) {
  auto client = ollama::create_client("http://192.168.1.5:11434");
  EXPECT_TRUE(client.is_valid());
  EXPECT_THAT(client.config_info(), testing::HasSubstr("192.168.1.5"));
}

TEST(OllamaFactoryTest, CreateClientEmptyUrlFallsBackToDefault) {
  auto client = ollama::create_client("");
  EXPECT_TRUE(client.is_valid());
  EXPECT_THAT(client.config_info(), testing::HasSubstr("localhost:11434"));
}

TEST(OllamaFactoryTest, CreateClientWithKeyHasValidConfig) {
  auto client =
      ollama::create_client_with_key("https://ollama.example.com", "tok123");
  EXPECT_TRUE(client.is_valid());
  EXPECT_THAT(client.config_info(), testing::HasSubstr("ollama.example.com"));
}

}  // namespace test
}  // namespace ai
