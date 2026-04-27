#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <httplib.h>

#include "ai/ollama.h"
#include "ai/types/generate_options.h"
#include "ai/types/stream_options.h"

#include <optional>
#include <string>
#include <string_view>

namespace ai {
namespace test {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool ping_ollama(const std::string& base_url) {
  // Parse host and port out of http[s]://host[:port][/...]
  std::string_view v(base_url);
  if (auto pos = v.find("://"); pos != std::string_view::npos)
    v.remove_prefix(pos + 3);

  auto slash = v.find('/');
  auto host_port = std::string(v.substr(0, slash));

  std::string host = host_port;
  int port = 11434;
  if (auto colon = host_port.rfind(':'); colon != std::string::npos) {
    host = host_port.substr(0, colon);
    try {
      port = std::stoi(host_port.substr(colon + 1));
    } catch (...) {
    }
  }

  httplib::Client client(host, port);
  client.set_connection_timeout(2);
  auto res = client.Get("/api/version");
  return res && res->status == 200;
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class OllamaIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* url_env = std::getenv("OLLAMA_BASE_URL");
    base_url_ = url_env ? url_env : "http://localhost:11434";

    const char* model_env = std::getenv("OLLAMA_TEST_MODEL");
    test_model_ = model_env ? model_env : "gemma4:26b";

    ollama_available_ = ping_ollama(base_url_);
  }

  std::string base_url_;
  std::string test_model_;
  bool ollama_available_ = false;
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_F(OllamaIntegrationTest, BasicTextGeneration) {
  if (!ollama_available_)
    GTEST_SKIP() << "Ollama not reachable at " << base_url_;

  auto client = ollama::create_client(base_url_);
  EXPECT_TRUE(client.is_valid());

  GenerateOptions opts(test_model_, "Reply with exactly one word: OK");
  auto result = client.generate_text(opts);

  EXPECT_TRUE(result.is_success()) << result.error_message();
  EXPECT_FALSE(result.text.empty());
  EXPECT_GT(result.usage.total_tokens, 0);
}

TEST_F(OllamaIntegrationTest, TextGenerationWithSystemPrompt) {
  if (!ollama_available_)
    GTEST_SKIP() << "Ollama not reachable at " << base_url_;

  auto client = ollama::create_client(base_url_);

  GenerateOptions opts(test_model_,
                       "You are a calculator. Answer only with numbers.",
                       "What is 1 + 1?");
  auto result = client.generate_text(opts);

  EXPECT_TRUE(result.is_success()) << result.error_message();
  EXPECT_FALSE(result.text.empty());
  // Model should mention "2" somewhere
  EXPECT_THAT(result.text, testing::HasSubstr("2"));
}

TEST_F(OllamaIntegrationTest, MaxTokensIsRespected) {
  if (!ollama_available_)
    GTEST_SKIP() << "Ollama not reachable at " << base_url_;

  auto client = ollama::create_client(base_url_);

  GenerateOptions opts(test_model_, "Count from 1 to 1000, one number per line.");
  opts.max_tokens = 10;  // Very tight budget
  auto result = client.generate_text(opts);

  EXPECT_TRUE(result.is_success()) << result.error_message();
  // The response must be short — finish_reason should be "length"
  EXPECT_EQ(result.finish_reason, kFinishReasonLength);
  EXPECT_LE(result.usage.completion_tokens, 15);  // Small slack for tokenization
}

TEST_F(OllamaIntegrationTest, StreamingTextGeneration) {
  if (!ollama_available_)
    GTEST_SKIP() << "Ollama not reachable at " << base_url_;

  auto client = ollama::create_client(base_url_);

  StreamOptions opts(GenerateOptions(test_model_, "Say: hello stream"));
  auto stream = client.stream_text(opts);

  int text_chunks = 0;
  bool got_finish = false;
  std::string accumulated;

  for (const auto& event : stream) {
    if (event.type == kStreamEventTypeTextDelta && !event.text_delta.empty()) {
      ++text_chunks;
      accumulated += event.text_delta;
    } else if (event.type == kStreamEventTypeFinish) {
      got_finish = true;
    } else if (event.type == kStreamEventTypeError) {
      FAIL() << "Stream error: "
             << (event.error ? *event.error : "unknown");
    }
  }

  EXPECT_GT(text_chunks, 0) << "No text chunks received from stream";
  EXPECT_TRUE(got_finish) << "Stream did not send a finish event";
  EXPECT_FALSE(accumulated.empty());
}

TEST_F(OllamaIntegrationTest, PlainHttpTransportWorks) {
  // Ollama runs over plain HTTP by default — verify the stream impl
  // correctly uses httplib::Client (not SSLClient) for http:// URLs
  if (!ollama_available_)
    GTEST_SKIP() << "Ollama not reachable at " << base_url_;

  // Ensure the base URL is http (not https)
  ASSERT_THAT(base_url_, testing::StartsWith("http://"))
      << "This test specifically exercises the plain-HTTP path; set "
         "OLLAMA_BASE_URL=http://... to run it";

  auto client = ollama::create_client(base_url_);
  GenerateOptions opts(test_model_, "Reply with: HTTP OK");
  auto result = client.generate_text(opts);

  EXPECT_TRUE(result.is_success()) << result.error_message();
}

TEST_F(OllamaIntegrationTest, ProviderMetadataPresent) {
  if (!ollama_available_)
    GTEST_SKIP() << "Ollama not reachable at " << base_url_;

  auto client = ollama::create_client(base_url_);
  GenerateOptions opts(test_model_, "Say: metadata test");
  auto result = client.generate_text(opts);

  EXPECT_TRUE(result.is_success()) << result.error_message();
  // provider_metadata should contain the raw JSON response
  EXPECT_TRUE(result.provider_metadata.has_value());
  // Should look like JSON
  EXPECT_THAT(*result.provider_metadata, testing::StartsWith("{"));
}

}  // namespace test
}  // namespace ai
