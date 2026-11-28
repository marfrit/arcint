cmake_minimum_required(VERSION 3.20)
project(arcint VERSION 1.8.3 LANGUAGES CXX)

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
  src/util/canonical_json.cpp
  src/util/sha256.cpp
  src/util/utf8.cpp
  src/util/text.cpp
  src/core/admission.cpp
  src/core/spill_guard.cpp
  src/core/dory.cpp
  src/core/reasoning.cpp
  src/core/renderer_material.cpp
  src/core/sampler.cpp
  src/core/sampling.cpp
  src/core/stop.cpp
  src/core/toolcall.cpp
  src/core/toolgrammar.cpp
  src/core/chat.cpp
  src/exec/backend_stub.cpp
  src/exec/opencl_card_select.cpp
  src/api/auth.cpp
  src/api/error.cpp
  src/api/handlers.cpp
  src/api/generate.cpp
  src/api/raw_trace.cpp
  src/api/metrics.cpp
  src/api/renderer.cpp
  src/api/request_id.cpp
  src/api/responses.cpp
  src/api/responses_stream.cpp
  src/api/responses_state.cpp
  src/api/slots.cpp
  src/api/tokenize.cpp
  src/http/server.cpp
)
target_include_directories(arcint_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${CMAKE_CURRENT_BINARY_DIR}/generated")
# cpp-httplib's select()-based helpers: without poll, process_request refuses
# every connection whose fd is >= FD_SETSIZE (1024) with a 500, and select()
# returns -1 there, which the client-gone probe (detail::is_socket_alive) would
# read as "gone". poll has neither the fd limit nor the errno ambiguity.
target_compile_definitions(arcint_core PUBLIC CPPHTTPLIB_USE_POLL)
target_link_libraries(arcint_core PUBLIC arcint_vendor Threads::Threads arcint_peg)

# TLS (--tls-cert/--tls-key) needs httplib's OpenSSL server. Off by default: the
# default build neither finds nor links OpenSSL, so the package's Depends stay
# unchanged (no libssl). ON is an extra arm, not the recipe: it exists so a
# deployment that wants TLS in-process can build one.
option(ARCINT_TLS "Serve HTTPS through httplib's OpenSSL server (needs OpenSSL)" OFF)
if(ARCINT_TLS)
  find_package(OpenSSL REQUIRED)
  target_compile_definitions(arcint_core PUBLIC CPPHTTPLIB_OPENSSL_SUPPORT ARCINT_TLS=1)
  target_link_libraries(arcint_core PUBLIC OpenSSL::SSL OpenSSL::Crypto)
endif()

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
                                     src/exec/llama_dspark.cpp src/exec/llama_toolgrammar.cpp)
  # The DFlash2 drafter's red cases (tests/dflash/legs.sh): a build with a
  # mutant compiled in, which says so at load. 0 is the served build; the
  # cases are listed in src/exec/llama_dflash.cpp and tests/dflash/legs.sh
  set(ARCINT_DFLASH_RED "0" CACHE STRING "DFlash2 red-case mutant (0: none; a test build only)")
  if(NOT ARCINT_DFLASH_RED STREQUAL "0")
    message(WARNING "ARCINT_DFLASH_RED=${ARCINT_DFLASH_RED}: a DFlash2 red-case build, not for serving")
  endif()
  target_compile_definitions(arcint_core PRIVATE ARCINT_DFLASH_RED=${ARCINT_DFLASH_RED})
  # The DSpark drafter's red cases (src/exec/dspark_ring.h, src/exec/llama_dspark.cpp):
  # 1 the anchor reads its own Markov context row, 2 the ring keeps rejected rows,
  # 3 the anchor's row drafted, 4 the llama.cpp port's n_max + 1-row block,
  # 5 a checkpoint carries later rows, 6 the step's budget ignored, 7 the anchor reads the row one
  # position early, 8 any target model taken, 9 a checkpoint payload read without its bounds, 10 failed
  # restores neither counted nor warned. A test build only.
  set(ARCINT_DSPARK_RED "0" CACHE STRING "DSpark red-case mutant (0: none; a test build only)")
  if(NOT ARCINT_DSPARK_RED STREQUAL "0")
    message(WARNING "ARCINT_DSPARK_RED=${ARCINT_DSPARK_RED}: a DSpark red-case build, not for serving")
  endif()
  target_compile_definitions(arcint_core PRIVATE ARCINT_DSPARK_RED=${ARCINT_DSPARK_RED})
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
  # contrib/llama.cpp 0108 (the dory architecture) adds llama_model_n_layer_physical to llama-ext.h;
  # without it a dory file cannot load (the architecture is unknown), and the admission geometry
  # keeps llama_model_n_layer
  file(READ "${ARCINT_LLAMA_DIR}/src/llama-ext.h" ARCINT_LLAMA_EXT_H)
  if(ARCINT_LLAMA_EXT_H MATCHES "llama_model_n_layer_physical")
    target_compile_definitions(arcint_core PUBLIC ARCINT_LLAMA_DORY=1)
  endif()
  # contrib/llama.cpp 0118: Dory's loop count per context (llama_set_dory_loops in llama.h);
  # without it --llama-dory-max-loops is refused at start
  if(ARCINT_LLAMA_H MATCHES "llama_set_dory_loops")
    target_compile_definitions(arcint_core PUBLIC ARCINT_LLAMA_DORY_TURN_LOOPS=1)
    message(STATUS "contrib/llama.cpp 0118 present: --llama-dory-max-loops is built")
  else()
    message(STATUS "ARCINT_LLAMA_DIR lacks contrib/llama.cpp patch 0118: --llama-dory-max-loops is refused at start")
  endif()
  # contrib/llama.cpp 0111: the weights the device declined at load, placed in host memory,
  # counted (llama_placement_downgrades in llama-ext.h); /props downgrades carries them
  if(ARCINT_LLAMA_EXT_H MATCHES "llama_placement_downgrades")
    target_compile_definitions(arcint_core PUBLIC ARCINT_LLAMA_PLACEMENT=1)
  else()
    message(WARNING "ARCINT_LLAMA_DIR lacks contrib/llama.cpp patch 0111: a weight the card declines at load is not reported")
  endif()
  # contrib/llama.cpp 0160: the driver's card figure (llama_device_card_memory) and the
  # reserved graph stats (llama_context_graph_stats). The backend uses the patch's types
  # outside the #ifdef, so a tree without it cannot compile: refused at configure.
  if(ARCINT_LLAMA_EXT_H MATCHES "llama_device_card_memory")
    message(STATUS "contrib/llama.cpp 0160 present: the driver's card figure is built")
  else()
    message(FATAL_ERROR "ARCINT_LLAMA_DIR lacks contrib/llama.cpp patch 0160 (llama_device_card_memory): the spill guard's backend needs its types")
  endif()
  # contrib/llama.cpp 0173: the expert cache's projected bytes at a budget; the guard's cache
  # suggestion searches it. The backend calls it outside the #ifdef, so a missing patch refuses.
  if(ARCINT_LLAMA_EXT_H MATCHES "llama_model_expert_cache_projected_bytes")
    message(STATUS "contrib/llama.cpp 0173 present: the expert cache projection is built")
  else()
    message(FATAL_ERROR "ARCINT_LLAMA_DIR lacks contrib/llama.cpp patch 0173 (llama_model_expert_cache_projected_bytes): the guard's cache suggestion needs it")
  endif()
  # contrib/llama.cpp 0153: the attention part's own oldest retained position
  # (llama_memory_attn_seq_pos_min in llama-ext.h), which a window-sized SWA
  # cache needs to decide whether a prefix trim is exact; without it
  # the build stops (the backend's prefix cuts call it)
  if(ARCINT_LLAMA_EXT_H MATCHES "llama_memory_attn_seq_pos_min")
    target_compile_definitions(arcint_core PUBLIC ARCINT_LLAMA_SWA_WINDOW=1)
    message(STATUS "contrib/llama.cpp 0153 present: --llama-swa-window is built")
  else()
    message(FATAL_ERROR "ARCINT_LLAMA_DIR lacks contrib/llama.cpp patch 0153 (llama_memory_attn_seq_pos_min): apply the series")
  endif()
  # contrib/llama.cpp 0209: the OpenCL backend writes through a view of a q8_0 / q4_0 tensor
  # (a checkpoint restore's K / V cells). 0209 adds no API, so the marker is its function in
  # ggml-opencl.cpp. Without it a q8_0 / q4_0 sliding-window cache keeps no checkpoints
  # (core/swa_window.h, 1.8.1's refusal): such a restore would write positions and no bytes.
  file(STRINGS "${ARCINT_LLAMA_DIR}/ggml/src/ggml-opencl/ggml-opencl.cpp" ARCINT_LLAMA_0209
       REGEX "ggml_cl_warn_view_write_dropped" LIMIT_COUNT 1)
  # the scan is a configure step: a tree that changes under an existing build directory (0209
  # reverted, another branch) must re-run it, or the define outlives the patch
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${ARCINT_LLAMA_DIR}/ggml/src/ggml-opencl/ggml-opencl.cpp")
  if(ARCINT_LLAMA_0209)
    target_compile_definitions(arcint_core PUBLIC ARCINT_LLAMA_KV_VIEW_WRITES=1)
    message(STATUS "contrib/llama.cpp 0209 present: q8_0 / q4_0 sliding-window checkpoints are restored")
  else()
    message(WARNING "ARCINT_LLAMA_DIR lacks contrib/llama.cpp patch 0209: a q8_0 / q4_0 sliding-window cache "
                    "keeps no context checkpoints on OpenCL")
  endif()
  # the series this source carries has 0209: the unit suite checks the tree agrees (a
  # renamed marker would otherwise turn the checkpoints off with every test green)
  file(GLOB ARCINT_SERIES_0209 "${CMAKE_CURRENT_SOURCE_DIR}/contrib/llama.cpp/patches/0209-*.patch")
  if(ARCINT_SERIES_0209)
    target_compile_definitions(arcint_core PUBLIC ARCINT_SERIES_HAS_0209=1)
  endif()
  # A red-case build: restore q8_0 / q4_0 sliding-window checkpoints although the tree lacks
  # 0209, to show the stale restore the refusal prevents (the served gate's red arm). Never
  # for serving.
  option(ARCINT_ASSUME_KV_VIEW_WRITES "red case: restore q8_0/q4_0 SWA checkpoints without 0209 (a test build only)" OFF)
  if(ARCINT_ASSUME_KV_VIEW_WRITES)
    target_compile_definitions(arcint_core PUBLIC ARCINT_ASSUME_KV_VIEW_WRITES=1)
    message(WARNING "ARCINT_ASSUME_KV_VIEW_WRITES=ON: a red-case build that restores q8_0 / q4_0 sliding-window "
                    "checkpoints whether or not the tree has 0209; not for serving")
  endif()
  if(ARCINT_LLAMA_H MATCHES "expert_cache_bytes")
    target_compile_definitions(arcint_core PUBLIC ARCINT_LLAMA_EXPERT_CACHE=1)
    # 0024: an expert cache per card under the layer split
    if(ARCINT_LLAMA_H MATCHES "llama_model_expert_cache_count")
      target_compile_definitions(arcint_core PUBLIC ARCINT_LLAMA_EXPERT_CACHE_DEV=1)
    endif()
    # 0068: the prefill-slot-loan (docs/campaigns/prefill-slot-loan.md)
    if(ARCINT_LLAMA_H MATCHES "slot_loan_bytes")
      target_compile_definitions(arcint_core PUBLIC ARCINT_LLAMA_SLOT_LOAN=1)
    else()
      message(WARNING "ARCINT_LLAMA_DIR lacks contrib/llama.cpp patch 0068: --llama-slot-loan is refused")
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
    tests/test_sampler_golden.cpp
    tests/test_sampling.cpp
    tests/test_stop.cpp
    tests/test_chat.cpp
    tests/test_minja_empty.cpp
    tests/test_minja_scope.cpp
    tests/test_chat_render.cpp
    tests/test_text.cpp
    tests/test_config.cpp
    tests/test_swa_window.cpp
    tests/test_verify_walk.cpp
    tests/test_walk_feed.cpp
    tests/test_mtp_trace.cpp
    tests/test_mtp_capture.cpp
    tests/test_draft_policy.cpp
    tests/test_dflash_walk.cpp
    tests/test_dspark.cpp
    tests/test_spec_state.cpp
    tests/test_bytes.cpp
    tests/test_token_ids.cpp
    tests/test_mtp_head_file.cpp
    tests/test_props.cpp
    tests/test_placement.cpp
    tests/test_planes.cpp
    tests/test_sampler_modes.cpp
    tests/test_fast_topk.cpp
    tests/test_mtp_depth.cpp
    tests/test_loop_guard.cpp
    tests/test_context_guard.cpp
    tests/test_lanes.cpp
    tests/test_reasoning.cpp
    tests/test_think_lead.cpp
    tests/test_raw_trace.cpp
    tests/test_stream_golden.cpp
    tests/test_admission.cpp
    tests/test_spill_guard.cpp
    tests/test_dory.cpp
    tests/test_dory_request.cpp
    tests/test_gemma4.cpp
    tests/test_inkling.cpp
    tests/test_markers.cpp
    tests/test_stream_utf8.cpp
    tests/test_stop_think_ids.cpp
    tests/test_decode_stats.cpp
    tests/test_slot_loan.cpp
    tests/test_loan_book.cpp
    tests/test_toolgrammar.cpp
    tests/test_inkling_grammar.cpp
    tests/test_force_content.cpp
    tests/test_tokenize_once.cpp
    tests/test_disconnect.cpp
    tests/test_responses.cpp
    tests/test_metrics.cpp
    tests/test_slots.cpp
    tests/test_auth.cpp
    tests/test_reqid.cpp
    tests/test_tokenize.cpp
    tests/test_opencl_card_select.cpp
    tests/test_cache_prompt.cpp
    tests/test_completion_timings.cpp
    tests/test_ignored_fields.cpp
    tests/test_request_defaults.cpp
    tests/test_refusal_table.cpp
    tests/test_request_shapes.cpp
    tests/test_template_file.cpp
  )
  target_link_libraries(arcint-test PRIVATE arcint_core)
  target_include_directories(arcint-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
  # tests/test_dspark.cpp runs dspark_ring.h's rules, a header: its red cases need the switch here too
  if(DEFINED ARCINT_DSPARK_RED)
    target_compile_definitions(arcint-test PRIVATE ARCINT_DSPARK_RED=${ARCINT_DSPARK_RED})
  endif()
  # round 13: test_loan_book reads the loan-KL driver's rule header
  # (contrib/llama-loan-kld/loan_kld_book.h), a device-free plain-data header
  target_include_directories(arcint-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/contrib/llama-loan-kld")
  if(ARCINT_LLAMA)
    # test_toolgrammar.cpp runs the generated GBNF through libllama's grammar
    # engine on text (llama-grammar.h, the entry points upstream's own grammar
    # tests use), so it needs the pinned tree's src/ on its include path.
    # Review B5: the toolgrammar_llama_* refusal cases exist in an ARCINT_LLAMA
    # build only. The device-free unit gate above (LABELS unit) runs the
    # generator's shape and the sampler's constraint hook without them; the
    # record is in tests/test_toolgrammar.cpp's header and in the campaign
    # (docs/campaigns/clean-turns.md, step 2 "the gate").
    # ... and its include/ for llama.h: test_loan_book's rule cases read
    # LLAMA_SLOT_LOAN_BYTES_ENV and llama_model_default_params (finding 8's rule
    # lives in the pinned tree's init and its default params, not in the tool)
    target_include_directories(arcint-test PRIVATE "${ARCINT_LLAMA_DIR}/src" "${ARCINT_LLAMA_DIR}/include")
    # test_inkling_grammar.cpp drives the Inkling call grammar through arcint's
    # Sampler on a real vocabulary: the pinned tree's vocab-only GGUF (part of
    # the llama.cpp source tarball), loaded with no device. A missing file
    # fails the case, it does not skip.
    target_compile_definitions(arcint-test PRIVATE
      ARCINT_LLAMA_VOCAB_GGUF="${ARCINT_LLAMA_DIR}/models/ggml-vocab-gpt-2.gguf")
    # contrib/llama.cpp 0116: test_planes.cpp runs the OpenCL backend's planar layout (the low-bit types
    # and Q3_K), a device-free header of the pinned tree's ggml-opencl
    if(EXISTS "${ARCINT_LLAMA_DIR}/ggml/src/ggml-opencl/ggml-opencl-planes.h")
      target_include_directories(arcint-test PRIVATE "${ARCINT_LLAMA_DIR}/ggml/src/ggml-opencl" "${ARCINT_LLAMA_DIR}/ggml/src")
      target_compile_definitions(arcint-test PRIVATE ARCINT_LLAMA_PLANES=1)
    endif()
    # clean-turns step 2's forced-malformed gate on a real vocabulary (vocab-only
    # load, no device). Not a ctest case: it needs the GGUF; run by hand.
    add_executable(arcint-tool-grammar-probe tools/tool_grammar_probe.cpp)
    target_link_libraries(arcint-tool-grammar-probe PRIVATE arcint_core)
    target_compile_options(arcint-tool-grammar-probe PRIVATE -Wall -Wextra -Wpedantic)
    # --llama-swa-window on a real model: a window-sized SWA cache's
    # checkpoint resume against a fresh full prefill, and its red arms (the
    # trim-only cut, a checkpoint without the SWA part). Not a ctest case: it
    # needs a GGUF; run by hand, CPU is enough (tools/swa_ckpt_probe.cpp).
    add_executable(arcint-swa-ckpt-probe tools/swa_ckpt_probe.cpp)
    target_link_libraries(arcint-swa-ckpt-probe PRIVATE arcint_core)
    target_include_directories(arcint-swa-ckpt-probe PRIVATE "${ARCINT_LLAMA_DIR}/src" "${ARCINT_LLAMA_DIR}/include")
    target_compile_options(arcint-swa-ckpt-probe PRIVATE -Wall -Wextra -Wpedantic)
    # Renders a chat template read from a file with the engine's own minja setup,
    # over a matrix of tools / thinking / exchange shapes (tools/template_probe.cpp).
    # Not a ctest case: it needs a template taken from a GGUF, which is never
    # committed.
    add_executable(arcint-template-probe tools/template_probe.cpp)
    target_link_libraries(arcint-template-probe PRIVATE arcint_core)
    target_compile_options(arcint-template-probe PRIVATE -Wall -Wextra -Wpedantic)
    # audit 15's number: a vocab-only GGUF, the old two-call encode against the
    # one-pass encode, on the host or the card host (tools/tokenize_cost.cpp,
    # the campaign's command). Not a ctest case: it needs the GGUF.
    add_executable(arcint-tokenize-cost tools/tokenize_cost.cpp)
    target_link_libraries(arcint-tokenize-cost PRIVATE arcint_core)
    target_compile_options(arcint-tokenize-cost PRIVATE -Wall -Wextra -Wpedantic)
    # campaign expert-foresight's layer-ahead step 1: the router-input capture
    # (tools/expert_router_capture.cpp), CPU only, reads the graph's
    # ffn_moe_logits/ffn_moe_topk tensors through llama_context_params.cb_eval
    # and self-checks the first decode token's ids against the ffn_gate_inp
    # weight read from the run's own GGUF. Not a ctest case: it needs a GGUF and
    # a prompt file; the device-free writer/reader round trip is
    # expert-router-capture below. llama-ext.h (llama_model_n_expert) is in the
    # pinned tree's src.
    add_executable(arcint-expert-router-capture tools/expert_router_capture.cpp)
    target_link_libraries(arcint-expert-router-capture PRIVATE arcint_core)
    target_include_directories(arcint-expert-router-capture PRIVATE "${ARCINT_LLAMA_DIR}/src")
    target_compile_options(arcint-expert-router-capture PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  # test_props.cpp reads src/exec/backend_llama.cpp (the KV-precision tripwire:
  # the llama backend cannot be constructed without a card).
  target_compile_definitions(arcint-test PRIVATE
    ARCINT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  if(ARCINT_LLAMA)
    # test_loan_book.cpp reads the cache's own init: the destructor's buffer list
    # is built inside a function a unit build cannot call (finding 7, third review)
    target_compile_definitions(arcint-test PRIVATE
      ARCINT_LLAMA_SRC_DIR="${ARCINT_LLAMA_DIR}")
  endif()
  target_compile_options(arcint-test PRIVATE -Wall -Wextra -Wpedantic)

  # --max-skips 0: a skip is a failure here unless the case is named with
  # --allow-skip <case> (tests/harness.h's SKIP_UNLESS). Local, filtered runs
  # (`./build/arcint-test some_filter`) are unaffected; this only binds ctest's
  # own invocation.
  add_test(NAME unit COMMAND arcint-test --max-skips 0)
  set_tests_properties(unit PROPERTIES LABELS unit)

  add_test(NAME roundtrip
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/roundtrip.sh" "$<TARGET_FILE:arcint>")

  # Review R5 (docs/campaigns/feyntalk.md): W1 splits prepare_chat, and a green
  # suite does not prove the served bytes stayed the same. The checked-in
  # capture of the chat and completions routes on the stub backend is compared
  # byte for byte, with only the ids and `created` normalised (normalise.py
  # names them and why). A refactor that moves one byte of these bodies fails
  # here.
  add_test(NAME feyntalk-golden
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/golden/feyntalk-w1/compare.sh"
                   "$<TARGET_FILE:arcint>")
  set_tests_properties(feyntalk-golden PROPERTIES TIMEOUT 120 LABELS unit)
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

  # ops item 1: GET /metrics is opt-in and arcint.timings is opt-in. The HTTP
  # half (route presence, Content-Type, the response object) on the stub.
  add_test(NAME metrics
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/metrics.sh"
                   "$<TARGET_FILE:arcint>")
  set_tests_properties(metrics PROPERTIES TIMEOUT 120 LABELS unit)

  # ops: GET /slots is opt-in (--slots) and state-only. The HTTP half (route
  # presence, the 401, one entry per slot, the busy slot's request id, and that
  # no request text reaches the body) on the stub.
  add_test(NAME slots
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/slots.sh"
                   "$<TARGET_FILE:arcint>")
  set_tests_properties(slots PROPERTIES TIMEOUT 120 LABELS unit)

  # ops item 2: the bearer key and the TLS flag pairing. The HTTP half (401s,
  # the /health exemption, preflight, the key file) on the stub.
  add_test(NAME auth
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/auth.sh"
                   "$<TARGET_FILE:arcint>")
  set_tests_properties(auth PROPERTIES TIMEOUT 120 LABELS unit)

  # X-Request-Id: the header contract (a valid inbound id is echoed, an absent
  # one is generated as req-<16 hex>, an invalid one is replaced and its bytes
  # appear nowhere) and the `rid=` carry-through of one request's own log
  # lines. Stub-only, device-free, so it belongs to the unit gate.
  add_test(NAME reqid
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/reqid.sh"
                   "$<TARGET_FILE:arcint>")
  set_tests_properties(reqid PROPERTIES TIMEOUT 240 LABELS unit)

  # ops-2 item 6: a bad --tls-cert/--tls-key is refused before the model load.
  # In a build without ARCINT_TLS the script reports SKIP (77), so it reads as
  # skipped rather than as a pass that asserted nothing; the ARCINT_TLS=ON gate
  # arm runs the assertion for real.
  add_test(NAME tls
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/tls.sh"
                   "$<TARGET_FILE:arcint>")
  set_tests_properties(tls PROPERTIES TIMEOUT 60 LABELS unit SKIP_RETURN_CODE 77)

  # ops item 3: POST /tokenize and POST /detokenize on the stub. The HTTP half
  # of the device-free gate: the llama-server shapes, the chat/Responses count
  # against a real generation's usage, the model-field lane routing and the
  # auth coverage.
  add_test(NAME tokenize
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/tokenize.sh"
                   "$<TARGET_FILE:arcint>")
  set_tests_properties(tokenize PROPERTIES TIMEOUT 120 LABELS unit)

  # Both device-free: the generator's own consistency check, and the
  # acceptance runner's self-test against a fixture manifest. Neither needs a
  # card or ARCINT_ACCEPTANCE, so both run in every build.
  add_test(NAME acceptance-enumeration
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tools/acceptance_manifest.py" --check)
  set_tests_properties(acceptance-enumeration PROPERTIES LABELS unit)

  add_test(NAME acceptance-runner
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/acceptance/selftest.py")
  set_tests_properties(acceptance-runner PROPERTIES LABELS unit)

  # The MTP window trace's replay (campaign research-mtp-width-policy, section 6
  # step 1): the trace format's round trip, the fixed-width arithmetic and the
  # port of Strata's cost-aware keep_draft. Standard library only (unittest), so
  # it runs on a build host with python3 and no packages; device-free.
  add_test(NAME mtp-width-replay
           COMMAND "${CMAKE_COMMAND}" -E env "PYTHONDONTWRITEBYTECODE=1"
                   "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_mtp_width_replay.py")
  set_tests_properties(mtp-width-replay PROPERTIES LABELS unit)

  # The MTP head-fit tools (campaign mtp-head-fit, E1): the capture format, the task
  # split, the round-start simulation, the replay's HTTP flow; with numpy, also eval
  # (its red case), the linear fit and the GGUF decoders. Device-free.
  add_test(NAME mtp-fit
           COMMAND "${CMAKE_COMMAND}" -E env "PYTHONDONTWRITEBYTECODE=1"
                   "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_mtp_fit.py")
  set_tests_properties(mtp-fit PROPERTIES LABELS unit)

  # tools/coder_mtp_gguf.py (campaign mtp-head-fit, E2): the head-only MTP GGUFs of the
  # no-training arms; a tensor it cannot place fails. Synthetic files, device-free.
  add_test(NAME coder-mtp-gguf
           COMMAND "${CMAKE_COMMAND}" -E env "PYTHONDONTWRITEBYTECODE=1"
                   "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_coder_mtp_gguf.py")
  set_tests_properties(coder-mtp-gguf PROPERTIES LABELS unit)

  # tools/llama_scoreboard.py: what makes a row of the community GPU tables comparable.
  add_test(NAME llama-scoreboard
           COMMAND "${CMAKE_COMMAND}" -E env "PYTHONDONTWRITEBYTECODE=1"
                   "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_llama_scoreboard.py")
  set_tests_properties(llama-scoreboard PROPERTIES LABELS unit)

  # tools/downgrade_check.py: what fails a unit at the release gate, device-free.
  add_test(NAME downgrade-check
           COMMAND "${CMAKE_COMMAND}" -E env "PYTHONDONTWRITEBYTECODE=1"
                   "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_downgrade_check.py")
  set_tests_properties(downgrade-check PROPERTIES LABELS unit)

  # tools/fa_port_dump_compare.py: the comparer for patch 0223's red of record, on synthetic dumps.
  add_test(NAME fa-port-dump-compare
           COMMAND "${CMAKE_COMMAND}" -E env "PYTHONDONTWRITEBYTECODE=1"
                   "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_fa_port_dump_compare.py")
  set_tests_properties(fa-port-dump-compare PROPERTIES LABELS unit)

  # contrib/llama.cpp/apply-series.sh's byte order (GitHub issue #6), device-free; its locale case
  # runs where en_US.UTF-8 is installed and is skipped (not failed) on a C-locale host
  add_test(NAME patch-series-order
           COMMAND "${CMAKE_COMMAND}" -E env "PYTHONDONTWRITEBYTECODE=1"
                   "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_patch_series_order.py")
  set_tests_properties(patch-series-order PROPERTIES LABELS unit)

  # tools/dspark_probe_read.py: the DSpark shortlist's probe log (patch 0243) read against a hand-counted fixture,
  # device-free; DSPARK_PROBE_READ_RED=1 / 2 / 3 / 4 are its reds (the kept count one too many, the inside divergences
  # folded into the prediction, the membership tripwire not counted, the validity limits not applied)
  add_test(NAME dspark-probe-read
           COMMAND "${CMAKE_COMMAND}" -E env "PYTHONDONTWRITEBYTECODE=1"
                   "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_dspark_probe_read.py")
  set_tests_properties(dspark-probe-read PROPERTIES LABELS unit)

  # tools/fn_sweep.py's flag tables and --no-mtp (the T1 leg), device-free.
  add_test(NAME fn-sweep-flags
           COMMAND "${CMAKE_COMMAND}" -E env "PYTHONDONTWRITEBYTECODE=1"
                   "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_fn_sweep_flags.py")
  set_tests_properties(fn-sweep-flags PROPERTIES LABELS unit)

  # tools/expert_layer_ahead_replay.py (campaign expert-foresight, the queued
  # layer-ahead replay): the arms on synthetic LAC1 captures, the hand-counted
  # metrics, the online order and the cache's residency view. Standard library
  # unittest (numpy needed for the arrays; gguf-py cells skip loudly without
  # it), so a build host without them still runs the cells that can.
  add_test(NAME expert-layer-ahead-replay
           COMMAND "${CMAKE_COMMAND}" -E env "PYTHONDONTWRITEBYTECODE=1"
                   "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_expert_layer_ahead_replay.py")
  set_tests_properties(expert-layer-ahead-replay PROPERTIES LABELS unit)

  # The release speed gate's decision function and parser (tests/acceptance/ab.py).
  # The Open Responses compliance gate (docs/campaigns/feyntalk.md Gate 3):
  # the official vendored CLI, the seven 1.3.0 ids against a --stub server. NOT
  # the unit label: it needs node >= 22 and the pinned npm packages, while the
  # offline unit gate does not. It skips loudly (exit 0, named) without node;
  # FEYNTALK_COMPLIANCE_REQUIRED=1 turns a skip into a failure.
  add_test(NAME feyntalk-compliance
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/feyntalk/run-compliance.sh"
                   "$<TARGET_FILE:arcint>")
  set_tests_properties(feyntalk-compliance PROPERTIES TIMEOUT 300 LABELS compliance)

  add_test(NAME acceptance-ab
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/acceptance/ab_selftest.py")
  set_tests_properties(acceptance-ab PROPERTIES LABELS unit)

  # patch 0070's device-free C++ cell, against the llama tree the build already
  # compiled against (EXPERT_PROFILE_SRC): the selftest's round trip,
  # truncation, version, tie-break, pending-swap, xe, heat-unit and heat-width
  # cells, plus the C++/Python byte equality. Run by python3 alone (the cell is
  # stdlib unittest), so the recipe is green on a build host without pytest;
  # the recipe's `ctest -L unit` still runs it, so a patch that does not compile
  # fails the package build instead of a skipped skip.
  if(ARCINT_LLAMA)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    # patch 0074's device-free terms cell: the fused q/k norm + IMRoPE kernel's
    # arithmetic and the dispatch's switch/order, read out of the patched tree
    # (LLAMA_SRC). The numerics stay a card arm (test-backend-ops RMS_NORM_MUL_ROPE).
    add_test(NAME fuse-qk-norm-rope
             COMMAND "${CMAKE_COMMAND}" -E env "LLAMA_SRC=${ARCINT_LLAMA_DIR}" "PYTHONDONTWRITEBYTECODE=1"
                     "${Python3_EXECUTABLE}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_fuse_qk_norm_rope.py")
    set_tests_properties(fuse-qk-norm-rope PROPERTIES LABELS unit)
    # patch 0075's device-free terms cell: the CPU-miss prefetch switch (default
    # off), a block-ahead hint in every row function, the volatile page touch and
    # its level-2 guard in both compute paths. The residency and parity arms are
    # in tests/test-opencl-miss-cpu.cpp (the patch's own C cell).
    # patch 0083's device-free cell: LLAMA_SYNC_TRIM's default-off, the trim
    # predicate, the device KV rotations and the one sampled-id read.
    add_test(NAME round-trip-trim
             COMMAND "${CMAKE_COMMAND}" -E env "LLAMA_SRC=${ARCINT_LLAMA_DIR}" "PYTHONDONTWRITEBYTECODE=1"
                     "${Python3_EXECUTABLE}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_round_trip_trim.py")
    set_tests_properties(round-trip-trim PROPERTIES LABELS unit)
    add_test(NAME cpu-miss-prefetch
             COMMAND "${CMAKE_COMMAND}" -E env "LLAMA_SRC=${ARCINT_LLAMA_DIR}" "PYTHONDONTWRITEBYTECODE=1"
                     "${Python3_EXECUTABLE}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_cpu_miss_prefetch.py")
    set_tests_properties(cpu-miss-prefetch PROPERTIES LABELS unit)
    # patch 0080's device-free cell: the MTP KV-only branch's store half equals the
    # full graph's, nothing follows it, and the switch defaults off. The K/V byte
    # equality is the card arm (README 0080).
    add_test(NAME mtp-kv-only
             COMMAND "${CMAKE_COMMAND}" -E env "LLAMA_SRC=${ARCINT_LLAMA_DIR}" "PYTHONDONTWRITEBYTECODE=1"
                     "${Python3_EXECUTABLE}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_mtp_kv_only.py")
    set_tests_properties(mtp-kv-only PROPERTIES LABELS unit)
    # patch 0081's device-free cell: the gated delta-net's rows form reads each
    # sequence's own cache row, and the packed output (attention scores plus the
    # K snapshots) is byte-identical to the gathered form's, on the CPU backend.
    add_executable(arcint-gdn-state-rows tests/test_gdn_state_rows.cpp)
    target_link_libraries(arcint-gdn-state-rows PRIVATE arcint_core)
    target_include_directories(arcint-gdn-state-rows PRIVATE "${ARCINT_LLAMA_DIR}/ggml/include")
    target_compile_options(arcint-gdn-state-rows PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME gdn-state-rows COMMAND arcint-gdn-state-rows)
    set_tests_properties(gdn-state-rows PROPERTIES LABELS unit)
    # patch 0119's device-free cell: a paged FLASH_ATTN_EXT over a shuffled pool,
    # a ring and a two-row page table is byte-equal to the unpaged op on the CPU
    # backend. Design: docs/campaigns/kv-streaming-design.md, increment I1.
    add_executable(arcint-fa-paged tests/test_fa_paged.cpp)
    target_link_libraries(arcint-fa-paged PRIVATE arcint_core)
    target_include_directories(arcint-fa-paged PRIVATE "${ARCINT_LLAMA_DIR}/ggml/include")
    target_compile_options(arcint-fa-paged PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME fa-paged COMMAND arcint-fa-paged)
    set_tests_properties(fa-paged PROPERTIES LABELS unit)
    # patch 0120's device-free cell: the KV stream's page map, host store,
    # eviction/write-back and table, and the pages through 0119's CPU paged
    # attention over four ubatches. Design: docs/campaigns/kv-streaming-design.md,
    # increment I2.
    add_executable(arcint-kv-stream-map tests/test_kv_stream_map.cpp)
    target_link_libraries(arcint-kv-stream-map PRIVATE arcint_core)
    target_include_directories(arcint-kv-stream-map PRIVATE
        "${ARCINT_LLAMA_DIR}/src" "${ARCINT_LLAMA_DIR}/ggml/include")
    target_compile_options(arcint-kv-stream-map PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME kv-stream-map COMMAND arcint-kv-stream-map)
    set_tests_properties(kv-stream-map PROPERTIES LABELS unit)
    # patch 0217's device-free cell: the recurrent-state rollback of inkling2 / inkling2_loop
    # (n_rs_seq 1..8, batches of 1..16 tokens, every rollback depth, one and two sequences) against
    # contexts that never saw the removed tokens, and the bound of the widened conv-window view.
    # It is the pinned tree's own tests/test-llama-archs.cpp, built against tests/llama-archs-shim
    # in place of llama.cpp's common library, which arcint does not build.
    # A tree without the mode stops the configure: the gate must not drop out of `ctest -L unit`
    # when the series is renumbered, rebased or applied in part.
    set(ARCINT_ARCHS_RS "")
    if(EXISTS "${ARCINT_LLAMA_DIR}/tests/test-llama-archs.cpp")
      file(STRINGS "${ARCINT_LLAMA_DIR}/tests/test-llama-archs.cpp" ARCINT_ARCHS_RS REGEX "--rs-rollback")
    endif()
    if(NOT ARCINT_ARCHS_RS)
      message(FATAL_ERROR "ARCINT_LLAMA_DIR lacks contrib/llama.cpp patch 0217 (the --rs-rollback mode of "
                          "tests/test-llama-archs.cpp): the ctest inkling-rs-rollback cannot be built; apply the series")
    endif()
    add_executable(arcint-llama-archs "${ARCINT_LLAMA_DIR}/tests/test-llama-archs.cpp")
    target_link_libraries(arcint-llama-archs PRIVATE arcint_core)
    target_include_directories(arcint-llama-archs BEFORE PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/llama-archs-shim"
        "${ARCINT_LLAMA_DIR}/include" "${ARCINT_LLAMA_DIR}/ggml/include" "${ARCINT_LLAMA_DIR}/src")
    add_test(NAME inkling-rs-rollback COMMAND arcint-llama-archs -s 1 --rs-rollback)
    set_tests_properties(inkling-rs-rollback PROPERTIES LABELS unit TIMEOUT 600
        ENVIRONMENT "GGML_NO_BACKTRACE=1")
    # patch 0123zzzzz3's device-free cell: the stream's page moves are per attention
    # layer. Three layers share one ring slot (the stream's slots[ikv % 2]); every
    # attention call copies its own host-store pages into the slot before it reads
    # them, and the CPU paged op does that copy too. Design: increment I5.
    add_executable(arcint-kv-stream-layers tests/test_kv_stream_layers.cpp)
    target_link_libraries(arcint-kv-stream-layers PRIVATE arcint_core)
    target_include_directories(arcint-kv-stream-layers PRIVATE
        "${ARCINT_LLAMA_DIR}/src" "${ARCINT_LLAMA_DIR}/ggml/include" "${ARCINT_LLAMA_DIR}/ggml/src")
    target_compile_options(arcint-kv-stream-layers PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME kv-stream-layers COMMAND arcint-kv-stream-layers)
    set_tests_properties(kv-stream-layers PROPERTIES LABELS unit)
    # 0129b: the stream beyond its pool has to read right -- two layers, four pages
    # streaming, q8_0:q4_0 and f16, compared byte for byte with the all-resident run
    add_executable(arcint-kv-stream-deep tests/test_kv_stream_deep.cpp)
    target_link_libraries(arcint-kv-stream-deep PRIVATE arcint_core)
    target_include_directories(arcint-kv-stream-deep PRIVATE
        "${ARCINT_LLAMA_DIR}/src" "${ARCINT_LLAMA_DIR}/ggml/include" "${ARCINT_LLAMA_DIR}/ggml/src")
    target_compile_options(arcint-kv-stream-deep PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME kv-stream-deep COMMAND arcint-kv-stream-deep)
    set_tests_properties(kv-stream-deep PROPERTIES LABELS unit)
    # patch 0121's device-free cell: the stream reaches llama_kv_cache and the graph and
    # stays off by default; reads the patched tree's source (a GGUF and a card are not
    # available here). Design: increment I3's prerequisite.
    add_executable(arcint-kv-stream-cache tests/test_kv_stream_cache.cpp)
    target_link_libraries(arcint-kv-stream-cache PRIVATE arcint_core)
    target_include_directories(arcint-kv-stream-cache PRIVATE
        "${ARCINT_LLAMA_DIR}/src" "${ARCINT_LLAMA_DIR}/ggml/include")
    target_compile_definitions(arcint-kv-stream-cache PRIVATE
        ARCINT_KV_STREAM_SRC_DIR="${ARCINT_LLAMA_DIR}/src"
        ARCINT_KV_STREAM_LLAMA_DIR="${ARCINT_LLAMA_DIR}")
    target_compile_options(arcint-kv-stream-cache PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME kv-stream-cache COMMAND arcint-kv-stream-cache)
    set_tests_properties(kv-stream-cache PROPERTIES LABELS unit)
    # 0123zz's device-free cell: the paged FLASH_ATTN_EXT predicate's logic, over the
    # agent's shapes with every clause flipped. The header lives in the pinned tree's
    # ggml-opencl, built only when the patch is present.
    if(EXISTS "${ARCINT_LLAMA_DIR}/ggml/src/ggml-opencl/ggml-opencl-fa-paged.h")
      add_executable(arcint-fa-paged-predicate tests/test_fa_paged_predicate.cpp)
      target_link_libraries(arcint-fa-paged-predicate PRIVATE arcint_core)
      target_include_directories(arcint-fa-paged-predicate PRIVATE
          "${ARCINT_LLAMA_DIR}/ggml/src/ggml-opencl" "${ARCINT_LLAMA_DIR}/ggml/src"
          "${ARCINT_LLAMA_DIR}/ggml/include")
      target_compile_options(arcint-fa-paged-predicate PRIVATE -Wall -Wextra -Wpedantic)
      add_test(NAME fa-paged-predicate COMMAND arcint-fa-paged-predicate)
      set_tests_properties(fa-paged-predicate PROPERTIES LABELS unit)
    endif()
    # patch 0212's device-free cell: the route of an indexed-bias decode (1.8.0's review item
    # S3), a failed GQA bias build falling back to the split variant instead of the abort. The
    # red twin compiles the rule before 0212 (FA_BIAS_ROUTE_RED) and must fail; the python
    # gate reads the dispatcher's wiring and the debug switch out of the patched tree.
    if(EXISTS "${ARCINT_LLAMA_DIR}/ggml/src/ggml-opencl/ggml-opencl-fa-bias-route.h")
      add_executable(arcint-fa-bias-route tests/test_fa_bias_route.cpp)
      target_include_directories(arcint-fa-bias-route PRIVATE "${ARCINT_LLAMA_DIR}/ggml/src/ggml-opencl")
      target_compile_options(arcint-fa-bias-route PRIVATE -Wall -Wextra -Wpedantic)
      add_test(NAME fa-bias-route COMMAND arcint-fa-bias-route)
      set_tests_properties(fa-bias-route PROPERTIES LABELS unit)
      add_executable(arcint-fa-bias-route-red tests/test_fa_bias_route.cpp)
      target_include_directories(arcint-fa-bias-route-red PRIVATE "${ARCINT_LLAMA_DIR}/ggml/src/ggml-opencl")
      target_compile_definitions(arcint-fa-bias-route-red PRIVATE FA_BIAS_ROUTE_RED=1)
      target_compile_options(arcint-fa-bias-route-red PRIVATE -Wall -Wextra -Wpedantic)
      add_test(NAME fa-bias-route-red COMMAND arcint-fa-bias-route-red)
      set_tests_properties(fa-bias-route-red PROPERTIES LABELS unit WILL_FAIL TRUE)
      add_test(NAME fa-bias-route-gate
               COMMAND "${CMAKE_COMMAND}" -E env "LLAMA_SRC=${ARCINT_LLAMA_DIR}" "PYTHONDONTWRITEBYTECODE=1"
                       "${Python3_EXECUTABLE}"
                       "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_fa_bias_route.py")
      set_tests_properties(fa-bias-route-gate PROPERTIES LABELS unit)
    endif()
    # I5, design section 4.3: the start refusal that stops the scheduler's silent CPU
    # placement of a paged FA. The rules are src/exec/kv_stream_refusal.h; the red
    # branch sets ARCINT_KV_STREAM_REFUSE_OFF=1 in the cell's main, which accepts an
    # ask the card cannot honour and fails the refusals.
    add_executable(arcint-kv-stream-refusal tests/test_kv_stream_refusal.cpp)
    target_link_libraries(arcint-kv-stream-refusal PRIVATE arcint_core)
    target_include_directories(arcint-kv-stream-refusal PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_options(arcint-kv-stream-refusal PRIVATE -Wall -Wextra -Wpedantic)
    # the unit's red switch; the served binary compiles the rule without it
    target_compile_definitions(arcint-kv-stream-refusal PRIVATE ARCINT_KV_STREAM_REFUSE_RED=1)
    add_test(NAME kv-stream-refusal COMMAND arcint-kv-stream-refusal)
    set_tests_properties(kv-stream-refusal PROPERTIES LABELS unit)
    # patch 0081's source-slot cell: the GDN family takes src[6] once (the rows
    # index; the replay's records in its own op), the mode word op_params[1]
    # selects the state path, and the raw gating projections of patch 0051 ride
    # src[8]/src[9]. The served graph's gate chain with those projections
    # attached computes the plain path's bytes on the CPU backend -- with rows on
    # and with replay on too. Before the fix the attached alpha sat on src[6] and
    # every unit crashed (the CPU op's I32 assert).
    add_executable(arcint-gdn-gate-slots tests/test_gdn_gate_slots.cpp)
    target_link_libraries(arcint-gdn-gate-slots PRIVATE arcint_core)
    target_include_directories(arcint-gdn-gate-slots PRIVATE
        "${ARCINT_LLAMA_DIR}/ggml/include" "${ARCINT_LLAMA_DIR}/src" "${ARCINT_LLAMA_DIR}/src/models")
    target_compile_options(arcint-gdn-gate-slots PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME gdn-gate-slots COMMAND arcint-gdn-gate-slots)
    set_tests_properties(gdn-gate-slots PROPERTIES LABELS unit)
    add_test(NAME gdn-gate-slots-gate
             COMMAND "${CMAKE_COMMAND}" -E env "LLAMA_SRC=${ARCINT_LLAMA_DIR}" "PYTHONDONTWRITEBYTECODE=1"
                     "${Python3_EXECUTABLE}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_gdn_gate_slots.py")
    set_tests_properties(gdn-gate-slots-gate PROPERTIES LABELS unit)
    # patch 0086/0087's device-free cell: a chain of prompt steps through
    # ggml_gated_delta_net_replay equals the snapshot chain's bytes on the CPU
    # backend (attention outputs and the final base state). The cards' arm is the
    # replay-chain case in test-backend-ops (README 0086/0087).
    add_executable(arcint-gdn-replay tests/test_gdn_replay.cpp)
    target_link_libraries(arcint-gdn-replay PRIVATE arcint_core)
    target_include_directories(arcint-gdn-replay PRIVATE "${ARCINT_LLAMA_DIR}/ggml/include")
    target_compile_options(arcint-gdn-replay PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME gdn-replay COMMAND arcint-gdn-replay)
    set_tests_properties(gdn-replay PROPERTIES LABELS unit)
    # patch 0082's device-free cell: one l2_norm over the joint q/k head groups
    # of the GDN conv output against the two separate norms, on the CPU backend,
    # byte equality; the gate cell reads the switch and the bf16 dispatch out of
    # the patched tree.
    add_executable(arcint-gdn-qk-l2-norm tests/test_gdn_qk_l2_norm.cpp)
    target_link_libraries(arcint-gdn-qk-l2-norm PRIVATE arcint_core)
    target_include_directories(arcint-gdn-qk-l2-norm PRIVATE "${ARCINT_LLAMA_DIR}/ggml/include")
    target_compile_options(arcint-gdn-qk-l2-norm PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME gdn-qk-l2-norm COMMAND arcint-gdn-qk-l2-norm)
    set_tests_properties(gdn-qk-l2-norm PROPERTIES LABELS unit)
    add_test(NAME gdn-qk-l2-norm-gate
             COMMAND "${CMAKE_COMMAND}" -E env "LLAMA_SRC=${ARCINT_LLAMA_DIR}" "PYTHONDONTWRITEBYTECODE=1"
                     "${Python3_EXECUTABLE}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_gdn_qk_l2_norm.py")
    set_tests_properties(gdn-qk-l2-norm-gate PROPERTIES LABELS unit)
    # patch 0081's build gate, read out of the patched tree (LLAMA_SRC): the
    # switch defaults off, one sequence only, the fused GDN the batch takes
    add_test(NAME gdn-state-rows-gate
             COMMAND "${CMAKE_COMMAND}" -E env "LLAMA_SRC=${ARCINT_LLAMA_DIR}" "PYTHONDONTWRITEBYTECODE=1"
                     "${Python3_EXECUTABLE}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_gdn_state_rows.py")
    set_tests_properties(gdn-state-rows-gate PROPERTIES LABELS unit)
    # patch 0086/0087's wiring, read out of the patched tree (LLAMA_SRC): the
    # switch defaults off, S gets two planes only when it is on, rollback drops
    # pending records, a save writes base + records, replay disables the rows
    # form, and the OpenCL kernel is built only under the switch.
    add_test(NAME gdn-replay-gate
             COMMAND "${CMAKE_COMMAND}" -E env "LLAMA_SRC=${ARCINT_LLAMA_DIR}" "PYTHONDONTWRITEBYTECODE=1"
                     "${Python3_EXECUTABLE}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_gdn_replay.py")
    set_tests_properties(gdn-replay-gate PROPERTIES LABELS unit)
    add_test(NAME gdn-rows-write-layout
             COMMAND "${CMAKE_COMMAND}" -E env "LLAMA_SRC=${ARCINT_LLAMA_DIR}" "PYTHONDONTWRITEBYTECODE=1"
                     "${Python3_EXECUTABLE}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_gdn_rows_write_layout.py")
    set_tests_properties(gdn-rows-write-layout PROPERTIES LABELS unit)
    add_test(NAME gdn-replay-hybrid-input
             COMMAND "${CMAKE_COMMAND}" -E env "LLAMA_SRC=${ARCINT_LLAMA_DIR}" "PYTHONDONTWRITEBYTECODE=1"
                     "${Python3_EXECUTABLE}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_gdn_replay_hybrid_input.py")
    set_tests_properties(gdn-replay-hybrid-input PROPERTIES LABELS unit)
    # 0086/0087's model list, read out of the patched tree (LLAMA_SRC): every
    # architecture the replay gate admits builds its GDN state with s_rpl_rows;
    # qwen35moe stayed on the snapshot gather and rs_s_rpl_rows was an unread
    # input with no buffer (the first decode's GGML_ASSERT(buffer)).
    add_test(NAME gdn-replay-model-branches
             COMMAND "${CMAKE_COMMAND}" -E env "LLAMA_SRC=${ARCINT_LLAMA_DIR}" "PYTHONDONTWRITEBYTECODE=1"
                     "${Python3_EXECUTABLE}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_gdn_replay_model_branches.py")
    set_tests_properties(gdn-replay-model-branches PROPERTIES LABELS unit)
    # 0131j: the KV window's copy-queue overlap keeps the slots' last readers across a
    # graph boundary, flushes the waited compute marker, and keys the prefetch reuse by the
    # layer's host store (the 1.7.0 release review's three overlap orderings).
    add_test(NAME fa-paged-overlap-order
             COMMAND "${CMAKE_COMMAND}" -E env "LLAMA_SRC=${ARCINT_LLAMA_DIR}" "PYTHONDONTWRITEBYTECODE=1"
                     "${Python3_EXECUTABLE}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_fa_paged_overlap_order.py")
    set_tests_properties(fa-paged-overlap-order PROPERTIES LABELS unit)
    # 0129x-0131d: the KV window's lead, verify, self-test, forced race, timeline and NOFIX
    # cells (29 source-text cases). Before 1.7.0 the file ran by hand only, and its
    # unittest.main() sat after the first class, so a script run took 3 of them.
    add_test(NAME fa-paged-lead
             COMMAND "${CMAKE_COMMAND}" -E env "LLAMA_SRC=${ARCINT_LLAMA_DIR}" "PYTHONDONTWRITEBYTECODE=1"
                     "${Python3_EXECUTABLE}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_fa_paged_lead.py")
    set_tests_properties(fa-paged-lead PROPERTIES LABELS unit)
    add_test(NAME expert-profile-save
             COMMAND "${CMAKE_COMMAND}" -E env "EXPERT_PROFILE_SRC=${ARCINT_LLAMA_DIR}" "PYTHONDONTWRITEBYTECODE=1"
                     "${Python3_EXECUTABLE}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_expert_profile_save.py")
    set_tests_properties(expert-profile-save PROPERTIES LABELS unit)
    # the LAC1 writer/reader round trip of the router-input capture (campaign
    # expert-foresight, the layer-ahead predictor's step 1): the Python mirror
    # writer, the C++ writer compiled alone, byte equality, the torn-record and
    # bad-header cells, and the two red mutants. No llama tree, no model: CXX is
    # the build's own compiler, and the cell skips if it cannot compile.
    add_test(NAME expert-router-capture
             COMMAND "${CMAKE_COMMAND}" -E env "CXX=${CMAKE_CXX_COMPILER}" "PYTHONDONTWRITEBYTECODE=1"
                     "${Python3_EXECUTABLE}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_expert_router_capture.py")
    set_tests_properties(expert-router-capture PROPERTIES LABELS unit)
    # patches 0088/0089's device-free cell: the int8 Q K^T's q8_0 block
    # offsets, its RNE Q quantizer and the S selftest against the f16 form, read
    # out of the kernel's own .inc (the text the backend prepends). The dispatch
    # and the switch, read out of the patched tree, are the python gate below.
    add_executable(arcint-fa-gqa-i8 tests/test_fa_gqa_i8.cpp)
    target_include_directories(arcint-fa-gqa-i8 PRIVATE
        "${ARCINT_LLAMA_DIR}/ggml/src/ggml-opencl/kernels")
    target_compile_options(arcint-fa-gqa-i8 PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME fa-gqa-i8 COMMAND arcint-fa-gqa-i8)
    set_tests_properties(fa-gqa-i8 PROPERTIES LABELS unit)
    add_test(NAME fa-gqa-i8-gate
             COMMAND "${CMAKE_COMMAND}" -E env "LLAMA_SRC=${ARCINT_LLAMA_DIR}" "PYTHONDONTWRITEBYTECODE=1"
                     "${Python3_EXECUTABLE}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_fa_gqa_i8.py")
    set_tests_properties(fa-gqa-i8-gate PROPERTIES LABELS unit)
    # patch 0187's eligibility: a host-side test of the pure predicate against the real strides of
    # Inkling's bias-table copy and the depth mix's transpose (cp39: the card never took the path)
    add_executable(arcint-opencl-cpy-t01 tests/test_opencl_cpy_t01.cpp)
    target_include_directories(arcint-opencl-cpy-t01 PRIVATE
        "${ARCINT_LLAMA_DIR}/ggml/src/ggml-opencl")
    target_compile_options(arcint-opencl-cpy-t01 PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME opencl-cpy-t01 COMMAND arcint-opencl-cpy-t01)
    set_tests_properties(opencl-cpy-t01 PROPERTIES LABELS unit)
    # patch 0220's device-free cell: the dense Q8_0 2D GEMM's gate on the exported graph's
    # MUL_MAT lines (red first), its launch shape, and a model of its arithmetic and red builds
    add_executable(arcint-opencl-q8-2d tests/test_opencl_q8_2d.cpp)
    target_include_directories(arcint-opencl-q8-2d PRIVATE
        "${ARCINT_LLAMA_DIR}/ggml/src/ggml-opencl")
    target_compile_options(arcint-opencl-q8-2d PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME opencl-q8-2d COMMAND arcint-opencl-q8-2d)
    set_tests_properties(opencl-q8-2d PROPERTIES LABELS unit)
    # patches 0204-0207's device-free cell: the depth mix's eligibility rule on the exported
    # graph's lines (red first), the launch plan, depth_mix.cl itself replayed on host threads
    # against the CPU op (and its red build), and the message read behind the card's page fault
    add_executable(arcint-depth-mix tests/test_depth_mix.cpp)
    target_link_libraries(arcint-depth-mix PRIVATE arcint_core)
    target_include_directories(arcint-depth-mix PRIVATE
        "${ARCINT_LLAMA_DIR}/ggml/include"
        "${ARCINT_LLAMA_DIR}/ggml/src/ggml-opencl/kernels")
    target_compile_options(arcint-depth-mix PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME depth-mix COMMAND arcint-depth-mix)
    set_tests_properties(depth-mix PROPERTIES LABELS unit)
    # patch 0211: the depth mix's TAKEN line at the FA port's level, once a process, and that
    # level one route_llama_log shows outside -v (read out of the patched tree and src/exec)
    add_test(NAME depth-mix-taken-line
             COMMAND "${CMAKE_COMMAND}" -E env "LLAMA_SRC=${ARCINT_LLAMA_DIR}" "PYTHONDONTWRITEBYTECODE=1"
                     "${Python3_EXECUTABLE}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_depth_mix_taken_line.py")
    set_tests_properties(depth-mix-taken-line PROPERTIES LABELS unit)
    # patch 0224: the 1.8.3 items' TAKEN lines at a level route_llama_log shows outside -v, once a process (read
    # out of the patched tree and src/exec)
    add_test(NAME taken-lines-183
             COMMAND "${CMAKE_COMMAND}" -E env "LLAMA_SRC=${ARCINT_LLAMA_DIR}" "PYTHONDONTWRITEBYTECODE=1"
                     "${Python3_EXECUTABLE}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_taken_lines_183.py")
    set_tests_properties(taken-lines-183 PROPERTIES LABELS unit)
    # patch 0221's device-free cell: the loop controller's rule on the exported graph's
    # LOOP_CONTROL lines (red first), the launch plan, the CPU op against the unfused chain, and
    # loop_control.cl itself replayed on host threads (f32 and bf16 builds, and the five reds);
    # patch 0226: loop_control_small.cl, the pair for 1 .. 8 rows, replayed the same way with its
    # six reds, its plan and the switch's row ranges
    add_executable(arcint-loop-control tests/test_loop_control.cpp)
    target_link_libraries(arcint-loop-control PRIVATE arcint_core)
    target_include_directories(arcint-loop-control PRIVATE
        "${ARCINT_LLAMA_DIR}/ggml/include"
        "${ARCINT_LLAMA_DIR}/ggml/src/ggml-opencl/kernels")
    target_compile_options(arcint-loop-control PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME loop-control COMMAND arcint-loop-control)
    set_tests_properties(loop-control PROPERTIES LABELS unit)
  endif()
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
