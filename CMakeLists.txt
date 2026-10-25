cmake_minimum_required(VERSION 3.20)
project(arcint VERSION 1.2.6 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# The libllama executor -- ggml's OpenCL backend runs the GGUF.
# ARCINT_LLAMA_DIR is a llama.cpp source tree at the pinned commit with
# contrib/llama.cpp/patches applied (contrib/llama.cpp/README.md); it is built
# as a subproject, static, OpenCL on.
option(ARCINT_LLAMA "Build the libllama (ggml OpenCL) executor backend" OFF)
set(ARCINT_LLAMA_DIR "" CACHE PATH "llama.cpp source tree for ARCINT_LLAMA")
option(ARCINT_WERROR   "Treat compiler warnings as errors"         OFF)
option(ARCINT_TESTS    "Build the test binaries"                   ON)

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

# The pinned llama.cpp's PEG grammar closure (clean-turns step 2): builds the
# Qwen-coder tool-call GBNF from the declared schemas. Copied verbatim from the
# pin's common/ (MIT, THIRD_PARTY.md), made self-contained by
# third_party/llama-peg/llama-peg-shim.h. Vendored code: no -Werror, no
# arcint warnings.
add_library(arcint_peg STATIC
  third_party/llama-peg/peg-parser.cpp
  third_party/llama-peg/json.cpp
  third_party/llama-peg/json-schema.cpp
  third_party/llama-peg/json-schema-to-grammar.cpp
  third_party/llama-peg/trie.cpp
  third_party/llama-peg/unicode.cpp
)
target_include_directories(arcint_peg PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama-peg")
target_link_libraries(arcint_peg PUBLIC arcint_vendor)

# ----------------------------------------------------------------------- core
add_library(arcint_core STATIC
  src/config.cpp
  src/util/log.cpp
  src/util/utf8.cpp
  src/util/text.cpp
  src/core/reasoning.cpp
  src/core/sampler.cpp
  src/core/sampling.cpp
  src/core/stop.cpp
  src/core/toolcall.cpp
  src/core/toolgrammar.cpp
  src/core/chat.cpp
  src/exec/backend_stub.cpp
  src/api/error.cpp
  src/api/handlers.cpp
  src/http/server.cpp
)
target_include_directories(arcint_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(arcint_core PUBLIC arcint_vendor Threads::Threads arcint_peg)
target_compile_options(arcint_core PRIVATE -Wall -Wextra -Wpedantic)
if(ARCINT_WERROR)
  target_compile_options(arcint_core PRIVATE -Werror)
endif()

if(ARCINT_LLAMA)
  if(NOT ARCINT_LLAMA_DIR OR NOT EXISTS "${ARCINT_LLAMA_DIR}/include/llama.h")
    message(FATAL_ERROR "ARCINT_LLAMA=ON needs ARCINT_LLAMA_DIR, a llama.cpp source tree")
  endif()
  if(NOT EXISTS "${ARCINT_LLAMA_DIR}/ggml/src/ggml-opencl/kernels/mul_mv_kq_q8_1.cl")
    message(WARNING "ARCINT_LLAMA_DIR has no contrib/llama.cpp/patches: ggml's stock OpenCL kernels, "
                    "a K-quant MoE's experts on the CPU (contrib/llama.cpp/README.md)")
  endif()
  set(GGML_OPENCL ON CACHE BOOL "" FORCE)
  set(GGML_OPENCL_USE_ADRENO_KERNELS OFF CACHE BOOL "" FORCE)
  set(GGML_OPENCL_EMBED_KERNELS ON CACHE BOOL "" FORCE)
  set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
  # the multimodal library (vision input, audio input, TTS audio generation):
  # libmtmd links ggml + llama only, never llama-common (tools/mtmd/CMakeLists.txt),
  # so one arcint process can serve it with GGML_MAX_NAME at the pin's 64
  # (docs/milestone-1.2.0.md §3). MTMD_VIDEO needs the subprocess tool and is off.
  set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
  set(MTMD_VIDEO OFF CACHE BOOL "" FORCE)
  # not llama.cpp's common library: it compiles another cpp-httplib (0.58)
  # than arcint's (0.18) into the same binary, and the server then fails to
  # bind (measured-here); the MTP drafter is rebuilt in llama_spec.cpp
  set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  add_subdirectory("${ARCINT_LLAMA_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/llama.cpp" EXCLUDE_FROM_ALL)
  find_package(OpenCL REQUIRED)
  target_sources(arcint_core PRIVATE src/exec/backend_llama.cpp src/exec/llama_spec.cpp src/exec/llama_dflash.cpp
                                     src/exec/llama_toolgrammar.cpp)
  # The DFlash2 drafter's red cases (tests/dflash/legs.sh): a build with a
  # mutant compiled in, which says so at load. 0 is the served build; the
  # cases are listed in src/exec/llama_dflash.cpp and tests/dflash/legs.sh
  set(ARCINT_DFLASH_RED "0" CACHE STRING "DFlash2 red-case mutant (0: none; a test build only)")
  if(NOT ARCINT_DFLASH_RED STREQUAL "0")
    message(WARNING "ARCINT_DFLASH_RED=${ARCINT_DFLASH_RED}: a DFlash2 red-case build, not for serving")
  endif()
  target_compile_definitions(arcint_core PRIVATE ARCINT_DFLASH_RED=${ARCINT_DFLASH_RED})
  # llama-ext.h: the pinned llama.cpp's staging API (nextn embeddings, MTP)
  target_include_directories(arcint_core PRIVATE "${ARCINT_LLAMA_DIR}/src" "${ARCINT_LLAMA_DIR}/tools/mtmd")
  target_link_libraries(arcint_core PUBLIC llama mtmd OpenCL::OpenCL)
  target_compile_definitions(arcint_core PUBLIC ARCINT_LLAMA=1)
  # libmtmd present in the tree (docs/milestone-1.2.0.md A1): the mmproj flag
  # and the multimodal branch are compiled only then
  if(EXISTS "${ARCINT_LLAMA_DIR}/tools/mtmd/mtmd.h")
    target_compile_definitions(arcint_core PUBLIC ARCINT_MTMD=1)
  else()
    message(WARNING "ARCINT_LLAMA_DIR lacks tools/mtmd: --mmproj is refused")
  endif()
  # contrib/llama.cpp 0021 (the expert cache) adds llama_model_params fields;
  # without it --llama-expert-cache is refused at load
  file(READ "${ARCINT_LLAMA_DIR}/include/llama.h" ARCINT_LLAMA_H)
  if(ARCINT_LLAMA_H MATCHES "expert_cache_bytes")
    target_compile_definitions(arcint_core PUBLIC ARCINT_LLAMA_EXPERT_CACHE=1)
    # 0024: an expert cache per card under the layer split
    if(ARCINT_LLAMA_H MATCHES "llama_model_expert_cache_count")
      target_compile_definitions(arcint_core PUBLIC ARCINT_LLAMA_EXPERT_CACHE_DEV=1)
    endif()
  else()
    message(WARNING "ARCINT_LLAMA_DIR lacks contrib/llama.cpp patch 0021: --llama-expert-cache is refused")
  endif()
endif()

configure_file(cmake/build_info.h.in "${CMAKE_CURRENT_BINARY_DIR}/generated/build_info.h" @ONLY)

# ----------------------------------------------------------------------- exe
add_executable(arcint src/main.cpp)
target_link_libraries(arcint PRIVATE arcint_core)
target_compile_options(arcint PRIVATE -Wall -Wextra -Wpedantic)

# ------------------------------------------------------------------- install
#
# Enough to run the engine from a systemd unit: the unit carries flags, the
# binary links ggml statically and finds the OpenCL ICD loader by itself.
include(GNUInstallDirs)
install(TARGETS arcint RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
# The unit ships as a template rather than being installed into a user's
# systemd directory: arcint runs as a *user* unit, and on the fleet that
# directory is a git repo owned by the unit manager. Installed from here so the
# package and a `cmake --install` agree on one file instead of the packaging
# copying it separately and drifting.
install(FILES packaging/arcint.service
        DESTINATION "${CMAKE_INSTALL_DATADIR}/arcint")
# 1.2.0: the vision and speech units run this binary too (docs/speech-service.md),
# shipped beside it as templates; the enrolment tool writes a voice profile on
# the host (the API never takes a reference recording, D5).
install(FILES contrib/systemd/arcint-vlm.service contrib/systemd/arcint-tts.service
              tools/tts_enrol.py
        DESTINATION "${CMAKE_INSTALL_DATADIR}/arcint")

# --------------------------------------------------------------------- tests
#
# Two ctest targets, not one (docs/design-0.3.1-test-ladder.md): everything
# below with LABELS unit is device-free and is what bare `ctest` runs; the
# card-requiring acceptance cells, only ever registered under ARCINT_ACCEPTANCE,
# carry LABELS acceptance instead so the partition is assertable
# (`ctest -N -L acceptance` must list zero tests in a default build).
option(ARCINT_ACCEPTANCE
       "Register the card-requiring acceptance cells (tests/acceptance/cells.json); needs ARCINT_LLAMA" OFF)

if(ARCINT_ACCEPTANCE AND NOT ARCINT_TESTS)
  message(FATAL_ERROR
    "ARCINT_ACCEPTANCE=ON requires ARCINT_TESTS=ON: without enable_testing() the "
    "cells register nowhere and `ctest -L acceptance` reports zero tests, green.")
endif()

if(ARCINT_ACCEPTANCE AND NOT ARCINT_LLAMA)
  message(FATAL_ERROR
    "ARCINT_ACCEPTANCE=ON requires ARCINT_LLAMA=ON. A stub binary would "
    "'pass' a card cell by serving stub bytes, which is worse than refusing "
    "to configure at all (docs/design-0.3.1-test-ladder.md §7).")
endif()

if(ARCINT_TESTS)
  enable_testing()
  add_executable(arcint-test
    tests/test_main.cpp
    tests/test_utf8.cpp
    tests/test_toolcall.cpp
    tests/test_sampler.cpp
    tests/test_sampling.cpp
    tests/test_stop.cpp
    tests/test_chat.cpp
    tests/test_text.cpp
    tests/test_config.cpp
    tests/test_verify_walk.cpp
    tests/test_dflash_walk.cpp
    tests/test_spec_state.cpp
    tests/test_token_ids.cpp
    tests/test_props.cpp
    tests/test_sampler_modes.cpp
    tests/test_loop_guard.cpp
    tests/test_lanes.cpp
    tests/test_reasoning.cpp
    tests/test_stream_utf8.cpp
    tests/test_stop_think_ids.cpp
    tests/test_decode_stats.cpp
    tests/test_toolgrammar.cpp
  )
  target_link_libraries(arcint-test PRIVATE arcint_core)
  target_include_directories(arcint-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
  if(ARCINT_LLAMA)
    # test_toolgrammar.cpp runs the generated GBNF through libllama's grammar
    # engine on text (llama-grammar.h, the entry points upstream's own grammar
    # tests use), so it needs the pinned tree's src/ on its include path.
    # Review B5: the toolgrammar_llama_* refusal cases exist in an ARCINT_LLAMA
    # build only. The device-free unit gate above (LABELS unit) runs the
    # generator's shape and the sampler's constraint hook without them; the
    # record is in tests/test_toolgrammar.cpp's header and in the campaign
    # (docs/campaigns/clean-turns.md, step 2 "the gate").
    target_include_directories(arcint-test PRIVATE "${ARCINT_LLAMA_DIR}/src")
    # clean-turns step 2's forced-malformed gate on a real vocabulary (vocab-only
    # load, no device). Not a ctest case: it needs the GGUF; run by hand.
    add_executable(arcint-tool-grammar-probe tools/tool_grammar_probe.cpp)
    target_link_libraries(arcint-tool-grammar-probe PRIVATE arcint_core)
    target_compile_options(arcint-tool-grammar-probe PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  # test_props.cpp reads src/exec/backend_llama.cpp (the KV-precision tripwire:
  # the llama backend cannot be constructed without a card).
  target_compile_definitions(arcint-test PRIVATE
    ARCINT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  target_compile_options(arcint-test PRIVATE -Wall -Wextra -Wpedantic)

  # --max-skips 0: a skip is a failure here unless the case is named with
  # --allow-skip <case> (tests/harness.h's SKIP_UNLESS). Local, filtered runs
  # (`./build/arcint-test some_filter`) are unaffected; this only binds ctest's
  # own invocation.
  add_test(NAME unit COMMAND arcint-test --max-skips 0)
  set_tests_properties(unit PROPERTIES LABELS unit)

  add_test(NAME roundtrip
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/roundtrip.sh" "$<TARGET_FILE:arcint>")
  # roundtrip.sh exits 77 with a reason when curl or python3 is missing. That
  # is deliberately NOT a ctest skip here: the package recipe gates on
  # `-L unit`, and a builder without curl must fail that gate, not pass it with
  # the 64 round-trip checks unrun (docs/design-0.3.1-test-ladder.md §7).
  set_tests_properties(roundtrip PROPERTIES TIMEOUT 120 LABELS unit)

  # Lane accounting under load. Stub-only on purpose: it is the half of §4.1
  # that needs no card and therefore can run in CI and under the sanitizers,
  # where a use-after-free in the lease shows up as a report rather than as a
  # rare hang on the GPU box. The card-side gates live in tests/concurrency/run.py.
  add_test(NAME stress
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/concurrency/stress.sh"
                   "$<TARGET_FILE:arcint>" 60 12 4)
  set_tests_properties(stress PROPERTIES TIMEOUT 300 LABELS unit)

  # Both device-free: the generator's own consistency check, and the
  # acceptance runner's self-test against a fixture manifest. Neither needs a
  # card or ARCINT_ACCEPTANCE, so both run in every build.
  add_test(NAME acceptance-enumeration
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tools/acceptance_manifest.py" --check)
  set_tests_properties(acceptance-enumeration PROPERTIES LABELS unit)

  add_test(NAME acceptance-runner
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/acceptance/selftest.py")
  set_tests_properties(acceptance-runner PROPERTIES LABELS unit)

  # The release speed gate's decision function and parser (tests/acceptance/ab.py).
  add_test(NAME acceptance-ab
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/acceptance/ab_selftest.py")
  set_tests_properties(acceptance-ab PROPERTIES LABELS unit)
endif()

if(ARCINT_ACCEPTANCE)
  # Parameters for the card-requiring cells, resolved at configure time into a
  # generated run manifest alongside $<TARGET_FILE:arcint> -- not environment,
  # which does not appear in the ctest log (docs/design-0.3.1-test-ladder.md §2).
  set(ARCINT_ACCEPTANCE_GGUF_AGENT "" CACHE FILEPATH
      "The agent GGUF (the dense 27B) the acceptance cells serve")
  set(ARCINT_ACCEPTANCE_GGUF_CODER "" CACHE FILEPATH
      "The coder GGUF the acceptance cells serve")
  set(ARCINT_ACCEPTANCE_GGUF_FLASHNEXT "" CACHE FILEPATH
      "The Flash-Next GGUF (first shard) the acceptance cells serve")
  set(ARCINT_ACCEPTANCE_PROMPT_DIR "" CACHE PATH
      "The served cells' assets: corpus.txt, needle-20k.json, expert-profile.bin, mtp-flash-next.gguf, mtp-vocab.json (tests/acceptance/cells/served_check.py)")
  set(ARCINT_ACCEPTANCE_DEVICE_LARGE "GPU.0" CACHE STRING
      "Device string (--device) for the 24 GB card in acceptance cells")
  set(ARCINT_ACCEPTANCE_DEVICE_SMALL "GPU.1" CACHE STRING
      "Device string (--device) for the 16 GB card in acceptance cells")
  # The one external cell's command (docs/design-pruefstand-cell.md §1): the
  # operator's wrapper that reaches the Prüfstand harness from this host and
  # prints `ACCEPTANCE-METRIC score <n> points`. Empty (the default) means the
  # cell skips by name. Lives in the cache and the generated manifest only --
  # never in a tracked file, which is the point of configuring it here.
  set(ARCINT_ACCEPTANCE_PRUEFSTAND "" CACHE STRING
      "Command line of the operator's Prüfstand wrapper for the external acceptance cell (empty: the cell skips; backslashes and double quotes are escaped into the manifest, generator expressions are not supported)")
  set(ARCINT_ACCEPTANCE_PRUEFSTAND_AGENT "" CACHE STRING
      "Command line of the operator's Prüfstand wrapper for the agent cell (empty: the cell skips; escaped like ARCINT_ACCEPTANCE_PRUEFSTAND)")
  # A command line, unlike the paths above, may well carry a double
  # quote; unescaped it would break the whole manifest, and run.py would die
  # in json.load before any cell ran.
  string(REPLACE "\\" "\\\\" _arcint_pruefstand_json "${ARCINT_ACCEPTANCE_PRUEFSTAND}")
  string(REPLACE "\"" "\\\"" _arcint_pruefstand_json "${_arcint_pruefstand_json}")
  string(REPLACE "\\" "\\\\" _arcint_pruefstand_agent_json "${ARCINT_ACCEPTANCE_PRUEFSTAND_AGENT}")
  string(REPLACE "\"" "\\\"" _arcint_pruefstand_agent_json "${_arcint_pruefstand_agent_json}")

  set(ARCINT_ACCEPTANCE_MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/acceptance/run_manifest.json")
  file(GENERATE OUTPUT "${ARCINT_ACCEPTANCE_MANIFEST}" CONTENT
"{
  \"binary\": \"$<TARGET_FILE:arcint>\",
  \"device_large\": \"${ARCINT_ACCEPTANCE_DEVICE_LARGE}\",
  \"device_small\": \"${ARCINT_ACCEPTANCE_DEVICE_SMALL}\",
  \"gguf_agent\": \"${ARCINT_ACCEPTANCE_GGUF_AGENT}\",
  \"gguf_coder\": \"${ARCINT_ACCEPTANCE_GGUF_CODER}\",
  \"gguf_flashnext\": \"${ARCINT_ACCEPTANCE_GGUF_FLASHNEXT}\",
  \"prompt_dir\": \"${ARCINT_ACCEPTANCE_PROMPT_DIR}\",
  \"pruefstand\": \"${_arcint_pruefstand_json}\",
  \"pruefstand_agent\": \"${_arcint_pruefstand_agent_json}\"
}
")

  set(ARCINT_ACCEPTANCE_CELLS "${CMAKE_CURRENT_SOURCE_DIR}/tests/acceptance/cells.json")
  file(READ "${ARCINT_ACCEPTANCE_CELLS}" _arcint_cells_json)
  string(JSON _arcint_cell_count LENGTH "${_arcint_cells_json}")
  if(_arcint_cell_count GREATER 0)
    math(EXPR _arcint_cell_last "${_arcint_cell_count} - 1")
    foreach(_arcint_i RANGE ${_arcint_cell_last})
      string(JSON _arcint_cell_name GET "${_arcint_cells_json}" ${_arcint_i} "name")
      add_test(NAME "acceptance-${_arcint_cell_name}"
               COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/acceptance/run.py"
                       --manifest "${ARCINT_ACCEPTANCE_MANIFEST}"
                       --cells "${ARCINT_ACCEPTANCE_CELLS}"
                       --cell "${_arcint_cell_name}")
      # No SKIP_RETURN_CODE here on purpose: a per-cell ctest entry must not be
      # able to go green on a skip by itself. `tests/acceptance/run.py --all
      # --allow-skip <cell>` is the only entry point that may treat a skip as
      # anything but a failure (docs/design-0.3.1-test-ladder.md §7).
      set_tests_properties("acceptance-${_arcint_cell_name}" PROPERTIES LABELS acceptance)

      # Per-cell TIMEOUT from cells.json's own timeout_seconds. A cell may
      # omit the key (the external cell did until docs/design-pruefstand-
      # cell.md gave it one) -- ERROR_VARIABLE turns that absent-key GET into
      # a normal (non-fatal) miss instead of a configure error, and the guard
      # checks the error, not only the value: such a cell keeps ctest's
      # default. The value check stays too, defensively, in case some future
      # cell carries an explicit JSON null there instead of omitting the key.
      string(JSON _arcint_cell_timeout ERROR_VARIABLE _arcint_timeout_err
             GET "${_arcint_cells_json}" ${_arcint_i} "timeout_seconds")
      if(NOT _arcint_timeout_err AND NOT _arcint_cell_timeout STREQUAL "")
        set_tests_properties("acceptance-${_arcint_cell_name}" PROPERTIES
                              TIMEOUT "${_arcint_cell_timeout}")
      endif()
    endforeach()
  endif()
endif()
