cmake_minimum_required(VERSION 3.14)
project(hapticsync LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization and RT warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O2 -Wall -Wextra -Wpedantic)
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/producer
    ${CMAKE_CURRENT_SOURCE_DIR}/postproc
)

# Look for ALSA (required on Linux target, optional on other platforms for dev)
find_package(ALSA QUIET)
find_package(Threads REQUIRED)

# ── 1. Producer Process Binary ───────────────────────────────────────────────
add_executable(producer_bin
    producer/main.cpp
    producer/alsa_source.cpp
    producer/assembler.cpp
)

target_link_libraries(producer_bin PRIVATE
    Threads::Threads
    $<$<PLATFORM_ID:Linux>:rt>
    $<$<BOOL:${ALSA_FOUND}>:${ALSA_LIBRARIES}>
)

if(ALSA_FOUND)
    target_include_directories(producer_bin PRIVATE ${ALSA_INCLUDE_DIRS})
    target_compile_definitions(producer_bin PRIVATE HAS_ALSA=1)
endif()

# ── 2. Post-Process Binary ───────────────────────────────────────────────────
add_executable(postproc_bin
    postproc/main.cpp
    postproc/vocab.cpp
    postproc/ctc_merge.cpp
)

target_link_libraries(postproc_bin PRIVATE
    Threads::Threads
    $<$<PLATFORM_ID:Linux>:rt>
)

# ── 3. Unit Tests ────────────────────────────────────────────────────────────
enable_testing()

add_executable(test_ring tests/cpp/test_ring.cpp)
target_link_libraries(test_ring PRIVATE Threads::Threads)
add_test(NAME test_ring COMMAND test_ring)

add_executable(test_tiling tests/cpp/test_tiling.cpp)
target_link_libraries(test_tiling PRIVATE Threads::Threads)
add_test(NAME test_tiling COMMAND test_tiling)

add_executable(test_ctc
    tests/cpp/test_ctc.cpp
    postproc/vocab.cpp
    postproc/ctc_merge.cpp
)
target_link_libraries(test_ctc PRIVATE Threads::Threads)
add_test(NAME test_ctc COMMAND test_ctc)

add_executable(test_events tests/cpp/test_events.cpp)
target_link_libraries(test_events PRIVATE Threads::Threads)
add_test(NAME test_events COMMAND test_events)

