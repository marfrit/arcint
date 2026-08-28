cmake_minimum_required(VERSION 3.20)
project(ligence VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(LIGENCE_OPENVINO "Build the OpenVINO executor backend (M1+)" OFF)
option(LIGENCE_WERROR   "Treat compiler warnings as errors"         OFF)
option(LIGENCE_TESTS    "Build the test binaries"                   ON)

find_package(Threads REQUIRED)

# ---------------------------------------------------------------- build info
# Resolved at configure time, not build time: reconfigure to refresh the sha.
set(LIGENCE_GIT_SHA "unknown")
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse --short=12 HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE LIGENCE_GIT_SHA
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" status --porcelain --untracked-files=no
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE LIGENCE_GIT_DIRTY
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(NOT LIGENCE_GIT_DIRTY STREQUAL "")
    set(LIGENCE_GIT_SHA "${LIGENCE_GIT_SHA}-dirty")
  endif()
endif()
configure_file(cmake/build_info.h.in "${CMAKE_CURRENT_BINARY_DIR}/generated/build_info.h" @ONLY)

# ------------------------------------------------------------------- vendored
add_library(ligence_vendor INTERFACE)
target_include_directories(ligence_vendor SYSTEM INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

# ----------------------------------------------------------------------- core
add_library(ligence_core STATIC
  src/config.cpp
  src/util/log.cpp
  src/util/sha256.cpp
  src/util/utf8.cpp
  src/util/text.cpp
  src/core/artifact.cpp
  src/core/model_registry.cpp
  src/core/prefix_cache.cpp
  src/core/sampler.cpp
  src/core/sampling.cpp
  src/core/stop.cpp
  src/core/toolcall.cpp
  src/core/chat.cpp
  src/exec/backend_stub.cpp
  src/api/error.cpp
  src/api/handlers.cpp
  src/http/server.cpp
)
target_include_directories(ligence_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(ligence_core PUBLIC ligence_vendor Threads::Threads)
target_compile_options(ligence_core PRIVATE -Wall -Wextra -Wpedantic)
if(LIGENCE_WERROR)
  target_compile_options(ligence_core PRIVATE -Werror)
endif()

if(LIGENCE_OPENVINO)
  find_package(OpenVINO REQUIRED COMPONENTS Runtime)
  target_sources(ligence_core PRIVATE src/exec/backend_ov.cpp)
  target_link_libraries(ligence_core PUBLIC openvino::runtime)
  target_compile_definitions(ligence_core PUBLIC LIGENCE_OPENVINO=1)
endif()

# ----------------------------------------------------------------------- exe
add_executable(ligence src/main.cpp)
target_link_libraries(ligence PRIVATE ligence_core)
target_compile_options(ligence PRIVATE -Wall -Wextra -Wpedantic)

# --------------------------------------------------------------------- tests
if(LIGENCE_TESTS)
  enable_testing()
  add_executable(ligence-test
    tests/test_main.cpp
    tests/test_utf8.cpp
    tests/test_toolcall.cpp
    tests/test_sampler.cpp
    tests/test_sampling.cpp
    tests/test_stop.cpp
    tests/test_registry.cpp
    tests/test_chat.cpp
    tests/test_text.cpp
    tests/test_config.cpp
    tests/test_prefix_cache.cpp
    tests/test_provenance.cpp
    tests/test_sha256.cpp
  )
  target_link_libraries(ligence-test PRIVATE ligence_core)
  target_include_directories(ligence-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
  # The provenance test reads models/allowlist-raw.json from the source tree.
  target_compile_definitions(ligence-test PRIVATE
    LIGENCE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  target_compile_options(ligence-test PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME unit COMMAND ligence-test)

  add_test(NAME roundtrip
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/roundtrip.sh" "$<TARGET_FILE:ligence>")
  set_tests_properties(roundtrip PROPERTIES TIMEOUT 120)
endif()
