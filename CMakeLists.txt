cmake_minimum_required(VERSION 3.20)
project(ligence VERSION 0.1.0 LANGUAGES CXX)

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

# The tokenizers extension ships with the OpenVINO wheel rather than with the
# runtime, so its path is deployment configuration. Baking the configure-time
# location in is what lets the installed binary start without a wrapper script
# that has to know where a Python virtualenv lives.
set(LIGENCE_TOKENIZERS_SO "" CACHE FILEPATH
    "Path to libopenvino_tokenizers.so (default: found next to the OpenVINO runtime)")

find_package(Threads REQUIRED)

# ---------------------------------------------------------------- build info
# Resolved at configure time, not build time: reconfigure to refresh the sha.
# A deployed binary must be able to say which commit it is. Builds from an
# exported tree have no .git to ask, so the sha can be passed in instead:
#   cmake -S . -B build -DLIGENCE_GIT_SHA=$(git rev-parse --short=12 HEAD)
set(LIGENCE_GIT_SHA "" CACHE STRING "Git sha to stamp in, when there is no .git to read")
if(LIGENCE_GIT_SHA)
  message(STATUS "ligence: build stamped with the supplied sha ${LIGENCE_GIT_SHA}")
endif()
if(NOT LIGENCE_GIT_SHA)
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
  src/core/drafter.cpp
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

  # Where OpenVINO's own shared objects live; the installed binary needs to
  # find them without LD_LIBRARY_PATH being set by hand at every start.
  get_target_property(_ov_lib openvino::runtime IMPORTED_LOCATION_RELEASE)
  if(NOT _ov_lib)
    get_target_property(_ov_lib openvino::runtime IMPORTED_LOCATION)
  endif()
  if(_ov_lib)
    get_filename_component(LIGENCE_OV_LIBDIR "${_ov_lib}" DIRECTORY)
  endif()

  if(NOT LIGENCE_TOKENIZERS_SO AND LIGENCE_OV_LIBDIR)
    # The wheel puts it in a sibling package, not beside the runtime.
    file(GLOB _tok
         "${LIGENCE_OV_LIBDIR}/../../openvino_tokenizers/lib/libopenvino_tokenizers.so"
         "${LIGENCE_OV_LIBDIR}/libopenvino_tokenizers.so")
    if(_tok)
      list(GET _tok 0 _tok0)
      get_filename_component(LIGENCE_TOKENIZERS_SO "${_tok0}" REALPATH)
    endif()
  endif()
  if(LIGENCE_TOKENIZERS_SO)
    message(STATUS "ligence: tokenizers extension ${LIGENCE_TOKENIZERS_SO}")
  else()
    message(WARNING "ligence: libopenvino_tokenizers.so not found; set "
                    "LIGENCE_TOKENIZERS_SO at configure time or in the environment")
  endif()
endif()

# ----------------------------------------------------------------------- exe
add_executable(ligence src/main.cpp)
target_link_libraries(ligence PRIVATE ligence_core)
target_compile_options(ligence PRIVATE -Wall -Wextra -Wpedantic)

# ------------------------------------------------------------------- install
#
# Enough to run the engine from a systemd unit: a binary that resolves its own
# shared objects, so the unit carries flags rather than a library search path.
if(LIGENCE_OV_LIBDIR)
  get_filename_component(_tok_dir "${LIGENCE_TOKENIZERS_SO}" DIRECTORY)
  set_target_properties(ligence PROPERTIES
    INSTALL_RPATH "${LIGENCE_OV_LIBDIR};${_tok_dir}"
    BUILD_WITH_INSTALL_RPATH FALSE)
endif()

include(GNUInstallDirs)
install(TARGETS ligence RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(FILES models/allowlist-raw.json
        DESTINATION "${CMAKE_INSTALL_DATADIR}/ligence")
install(PROGRAMS tools/export_mtp.py
        DESTINATION "${CMAKE_INSTALL_DATADIR}/ligence/tools")

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
    tests/test_drafter.cpp
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
