# ---------------------------------------------------------------------------
# Ollama-specific test executable
# Does not require ClickHouse; uses a longer timeout for local LLM inference.
# ---------------------------------------------------------------------------
add_executable(ai_ollama_tests
    unit/ollama_client_test.cpp
    integration/ollama_integration_test.cpp
)

target_link_libraries(ai_ollama_tests
    PRIVATE
        ai-sdk-cpp-ollama
        ai-sdk-cpp-openai   # OllamaRequestBuilder extends OpenAIRequestBuilder
        ai-sdk-cpp-core
        httplib::httplib
        concurrentqueue
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
)

target_include_directories(ai_ollama_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
        ${CMAKE_SOURCE_DIR}/src
)

set_target_properties(ai_ollama_tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_compile_options(ai_ollama_tests PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>
    $<$<CXX_COMPILER_ID:MSVC>:/W3>
)

target_compile_definitions(ai_ollama_tests PRIVATE
    AI_SDK_TESTING=1
    AI_SDK_HAS_OLLAMA=1
    AI_SDK_HAS_OPENAI=1
    CPPHTTPLIB_OPENSSL_SUPPORT=1
    CPPHTTPLIB_BROTLI_SUPPORT=1
    CPPHTTPLIB_THREAD_POOL_COUNT=8
)

include(GoogleTest)
gtest_discover_tests(ai_ollama_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTIES
        TIMEOUT 120   # Local LLM inference can be slow
)

# ---------------------------------------------------------------------------
# Full test suite (requires ClickHouse)
# ---------------------------------------------------------------------------

if(NOT TARGET ClickHouse::Client)
    message(STATUS "ClickHouse::Client not available - skipping ai_tests (initialize third_party/clickhouse-cpp submodule to enable)")
    return()
endif()

# Test executable for AI SDK C++
add_executable(ai_tests
    # Unit tests
    unit/openai_client_test.cpp
    unit/anthropic_client_test.cpp
    unit/types_test.cpp
    unit/openai_stream_test.cpp
    unit/anthropic_stream_test.cpp
    unit/openai_embeddings_test.cpp

    # Integration tests
    integration/openai_integration_test.cpp
    integration/anthropic_integration_test.cpp
    integration/tool_calling_integration_test.cpp
    integration/clickhouse_integration_test.cpp
    integration/openai_embeddings_integration_test.cpp
    integration/multi_step_duplicate_execution_test.cpp

    # Utility classes
    utils/mock_openai_client.cpp
    utils/mock_anthropic_client.cpp
    utils/test_fixtures.cpp
)

# Link against the main library and test dependencies
target_link_libraries(ai_tests
    PRIVATE
        ai-sdk-cpp
        httplib::httplib  # Need direct access to httplib for internal headers
        concurrentqueue   # Need direct access to concurrentqueue
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
        ClickHouse::Client  # ClickHouse C++ client for integration tests
)

# Include directories for tests
target_include_directories(ai_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
        ${CMAKE_SOURCE_DIR}/src  # Access to private headers
)

# Set test properties
set_target_properties(ai_tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Compiler options for tests (less strict than main library)
if(MSVC)
    target_compile_options(ai_tests PRIVATE /W3)
else()
    target_compile_options(ai_tests PRIVATE -Wall -Wextra)
endif()

# Define preprocessor macros for tests
target_compile_definitions(ai_tests PRIVATE
    AI_SDK_TESTING=1
    $<$<CONFIG:Debug>:AI_SDK_DEBUG_TESTS=1>
    # httplib configuration (must match main library)
    CPPHTTPLIB_OPENSSL_SUPPORT=1
    CPPHTTPLIB_BROTLI_SUPPORT=1
    CPPHTTPLIB_THREAD_POOL_COUNT=8
)

# Discover tests for CTest
include(GoogleTest)
gtest_discover_tests(ai_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTIES 
        TIMEOUT 30
)
