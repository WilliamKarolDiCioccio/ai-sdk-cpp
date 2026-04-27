# If nlohmann_json is already provided by the parent project (e.g. via vcpkg),
# skip adding the patched copy to avoid duplicate-target conflicts.
if(TARGET nlohmann_json::nlohmann_json)
    return()
endif()

# Configure nlohmann/json build options
set(JSON_BuildTests OFF CACHE INTERNAL "Disable JSON tests")
set(JSON_Install    OFF CACHE INTERNAL "Disable JSON installation (embedded use)")

# Add nlohmann/json as subdirectory
# PATCHED: Using patched version without localeconv dependency
add_subdirectory(
    ${AI_SDK_THIRD_PARTY_DIR}/nlohmann_json_patched
    ${CMAKE_CURRENT_BINARY_DIR}/nlohmann_json
)
