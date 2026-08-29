cmake_minimum_required(VERSION 3.20)
project(arcint VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(ARCINT_OPENVINO "Build the OpenVINO executor backend (M1+)" OFF)
option(ARCINT_WERROR   "Treat compiler warnings as errors"         OFF)
option(ARCINT_TESTS    "Build the test binaries"                   ON)

# The tokenizers extension ships with the OpenVINO wheel rather than with the
# runtime, so its path is deployment configuration. Baking the configure-time
# location in is what lets the installed binary start without a wrapper script
# that has to know where a Python virtualenv lives.
set(ARCINT_TOKENIZERS_SO "" CACHE FILEPATH
    "Path to libopenvino_tokenizers.so (default: found next to the OpenVINO runtime)")

find_package(Threads REQUIRED)

# ---------------------------------------------------------------- build info
# Resolved at configure time, not build time: reconfigure to refresh the sha.
# A deployed binary must be able to say which commit it is. Builds from an
# exported tree have no .git to ask, so the sha can be passed in instead:
#   cmake -S . -B build -DARCINT_GIT_SHA=$(git rev-parse --short=12 HEAD)
set(ARCINT_GIT_SHA "" CACHE STRING "Git sha to stamp in, when there is no .git to read")
if(ARCINT_GIT_SHA)
  message(STATUS "arcint: build stamped with the supplied sha ${ARCINT_GIT_SHA}")
endif()
if(NOT ARCINT_GIT_SHA)
set(ARCINT_GIT_SHA "unknown")
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse --short=12 HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE ARCINT_GIT_SHA
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" status --porcelain --untracked-files=no
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE ARCINT_GIT_DIRTY
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(NOT ARCINT_GIT_DIRTY STREQUAL "")
    set(ARCINT_GIT_SHA "${ARCINT_GIT_SHA}-dirty")
  endif()
endif()
endif()

# ------------------------------------------------------------------- vendored
add_library(arcint_vendor INTERFACE)
target_include_directories(arcint_vendor SYSTEM INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

# ----------------------------------------------------------------------- core
add_library(arcint_core STATIC
  src/config.cpp
  src/util/log.cpp
  src/util/sha256.cpp
  src/util/utf8.cpp
  src/util/text.cpp
  src/core/artifact.cpp
  src/core/block_pool.cpp
  src/core/model_registry.cpp
  src/core/prefix_cache.cpp
  src/core/sampler.cpp
  src/core/sampling.cpp
  src/core/stop.cpp
  src/core/toolcall.cpp
  src/core/chat.cpp
  src/core/drafter.cpp
  src/core/turnstile.cpp
  src/exec/backend_stub.cpp
  src/api/error.cpp
  src/api/handlers.cpp
  src/http/server.cpp
)
target_include_directories(arcint_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(arcint_core PUBLIC arcint_vendor Threads::Threads)
target_compile_options(arcint_core PRIVATE -Wall -Wextra -Wpedantic)
if(ARCINT_WERROR)
  target_compile_options(arcint_core PRIVATE -Werror)
endif()

if(ARCINT_OPENVINO)
  find_package(OpenVINO REQUIRED COMPONENTS Runtime)
  target_sources(arcint_core PRIVATE src/exec/backend_ov.cpp)
  target_link_libraries(arcint_core PUBLIC openvino::runtime)
  target_compile_definitions(arcint_core PUBLIC ARCINT_OPENVINO=1)

  # Where OpenVINO's own shared objects live; the installed binary needs to
  # find them without LD_LIBRARY_PATH being set by hand at every start.
  get_target_property(_ov_lib openvino::runtime IMPORTED_LOCATION_RELEASE)
  if(NOT _ov_lib)
    get_target_property(_ov_lib openvino::runtime IMPORTED_LOCATION)
  endif()
  if(_ov_lib)
    get_filename_component(ARCINT_OV_LIBDIR "${_ov_lib}" DIRECTORY)
  endif()

  if(NOT ARCINT_TOKENIZERS_SO AND ARCINT_OV_LIBDIR)
    # The wheel puts it in a sibling package, not beside the runtime.
    file(GLOB _tok
         "${ARCINT_OV_LIBDIR}/../../openvino_tokenizers/lib/libopenvino_tokenizers.so"
         "${ARCINT_OV_LIBDIR}/libopenvino_tokenizers.so")
    if(_tok)
      list(GET _tok 0 _tok0)
      get_filename_component(ARCINT_TOKENIZERS_SO "${_tok0}" REALPATH)
    endif()
  endif()
  if(ARCINT_TOKENIZERS_SO)
    message(STATUS "arcint: tokenizers extension ${ARCINT_TOKENIZERS_SO}")
  else()
    message(WARNING "arcint: libopenvino_tokenizers.so not found; set "
                    "ARCINT_TOKENIZERS_SO at configure time or in the environment")
  endif()
endif()

# Generated last on purpose: it bakes in the tokenizers path discovered above,
# and a configure_file that runs before the discovery writes an empty one (which
# is a binary that cannot start without a wrapper -- exactly what the baked path
# exists to avoid).
configure_file(cmake/build_info.h.in "${CMAKE_CURRENT_BINARY_DIR}/generated/build_info.h" @ONLY)

# ----------------------------------------------------------------------- exe
add_executable(arcint src/main.cpp)
target_link_libraries(arcint PRIVATE arcint_core)
target_compile_options(arcint PRIVATE -Wall -Wextra -Wpedantic)

# ------------------------------------------------------------------- install
#
# Enough to run the engine from a systemd unit: a binary that resolves its own
# shared objects, so the unit carries flags rather than a library search path.
if(ARCINT_OV_LIBDIR)
  get_filename_component(_tok_dir "${ARCINT_TOKENIZERS_SO}" DIRECTORY)
  set_target_properties(arcint PROPERTIES
    INSTALL_RPATH "${ARCINT_OV_LIBDIR};${_tok_dir}"
    BUILD_WITH_INSTALL_RPATH FALSE)
endif()

include(GNUInstallDirs)
install(TARGETS arcint RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(FILES models/allowlist-raw.json
        DESTINATION "${CMAKE_INSTALL_DATADIR}/arcint")
install(PROGRAMS tools/export_mtp.py
        DESTINATION "${CMAKE_INSTALL_DATADIR}/arcint/tools")

# --------------------------------------------------------------------- tests
if(ARCINT_TESTS)
  enable_testing()
  add_executable(arcint-test
    tests/test_main.cpp
    tests/test_utf8.cpp
    tests/test_toolcall.cpp
    tests/test_sampler.cpp
    tests/test_sampling.cpp
    tests/test_stop.cpp
    tests/test_registry.cpp
    tests/test_chat.cpp
    tests/test_drafter.cpp
    tests/test_text.cpp
    tests/test_config.cpp
    tests/test_prefix_cache.cpp
    tests/test_block_pool.cpp
    tests/test_turnstile.cpp
    tests/test_provenance.cpp
    tests/test_sha256.cpp
  )
  target_link_libraries(arcint-test PRIVATE arcint_core)
  target_include_directories(arcint-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
  # The provenance test reads models/allowlist-raw.json from the source tree.
  target_compile_definitions(arcint-test PRIVATE
    ARCINT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  target_compile_options(arcint-test PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME unit COMMAND arcint-test)

  add_test(NAME roundtrip
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/roundtrip.sh" "$<TARGET_FILE:arcint>")
  set_tests_properties(roundtrip PROPERTIES TIMEOUT 120)

  # Lane accounting under load. Stub-only on purpose: it is the half of §4.1
  # that needs no card and therefore can run in CI and under the sanitizers,
  # where a use-after-free in the lease shows up as a report rather than as a
  # rare hang on the GPU box. The card-side gates live in tests/concurrency/run.py.
  add_test(NAME stress
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/concurrency/stress.sh"
                   "$<TARGET_FILE:arcint>" 60 12 4)
  set_tests_properties(stress PROPERTIES TIMEOUT 300)
endif()
