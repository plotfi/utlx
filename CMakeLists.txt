cmake_minimum_required(VERSION 3.20)
project(TLXMemOpsPlugin LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Standalone build for the TLX plugin.
#
# Required variables (set via -D or environment):
#   TRITON_SOURCE_DIR  — path to the Triton source tree
#   TRITON_BUILD_DIR   — path to the Triton CMake build directory
#   LLVM_BUILD_DIR     — path to the LLVM/MLIR build directory
#
# Example:
#   export TRITON_EXT_ENABLED=1
#   cmake -B build \
#     -DTRITON_SOURCE_DIR=/path/to/triton \
#     -DTRITON_BUILD_DIR=/path/to/triton/build/cmake.linux-x86_64-cpython-3.11 \
#     -DLLVM_BUILD_DIR=/path/to/triton/llvm-project/build
#   cmake --build build
# ---------------------------------------------------------------------------

# Accept from environment if not set via -D
if(NOT TRITON_SOURCE_DIR AND DEFINED ENV{TRITON_SOURCE_DIR})
  set(TRITON_SOURCE_DIR $ENV{TRITON_SOURCE_DIR})
endif()
if(NOT TRITON_BUILD_DIR AND DEFINED ENV{TRITON_BUILD_DIR})
  set(TRITON_BUILD_DIR $ENV{TRITON_BUILD_DIR})
endif()
if(NOT LLVM_BUILD_DIR AND DEFINED ENV{LLVM_BUILD_DIR})
  set(LLVM_BUILD_DIR $ENV{LLVM_BUILD_DIR})
endif()

if(NOT TRITON_SOURCE_DIR)
  message(FATAL_ERROR "TRITON_SOURCE_DIR must be set (Triton source tree root)")
endif()
if(NOT TRITON_BUILD_DIR)
  message(FATAL_ERROR "TRITON_BUILD_DIR must be set (Triton CMake build directory)")
endif()
if(NOT LLVM_BUILD_DIR)
  message(FATAL_ERROR "LLVM_BUILD_DIR must be set (LLVM/MLIR build directory)")
endif()

# Find MLIR/LLVM CMake packages
list(APPEND CMAKE_PREFIX_PATH "${LLVM_BUILD_DIR}/lib/cmake/mlir")
list(APPEND CMAKE_PREFIX_PATH "${LLVM_BUILD_DIR}/lib/cmake/llvm")
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using MLIR from ${MLIR_DIR}")
message(STATUS "Using Triton source: ${TRITON_SOURCE_DIR}")
message(STATUS "Using Triton build: ${TRITON_BUILD_DIR}")

# Find libtriton.so
find_library(TRITON_LIBRARY
  NAMES triton
  PATHS "${TRITON_SOURCE_DIR}/python/triton/_C"
  NO_DEFAULT_PATH
)
if(NOT TRITON_LIBRARY)
  message(FATAL_ERROR "Could not find libtriton.so in ${TRITON_SOURCE_DIR}/python/triton/_C")
endif()
message(STATUS "Using libtriton: ${TRITON_LIBRARY}")

# ---------------------------------------------------------------------------
# Plugin target
# ---------------------------------------------------------------------------

set(TLX_PLUGIN_SOURCES
    TLXLocalAllocPlugin.cpp
    TLXConversionPatterns.cpp
)

add_library(TLXMemOpsPlugin SHARED ${TLX_PLUGIN_SOURCES})

target_include_directories(TLXMemOpsPlugin PRIVATE
    # Triton source headers (triton/Dialect/..., triton/Tools/...)
    ${TRITON_SOURCE_DIR}/include
    # python/src/ir.h is included as "python/src/ir.h" from PluginUtils.h
    ${TRITON_SOURCE_DIR}
    # Generated Triton headers (.h.inc files)
    ${TRITON_BUILD_DIR}/include
    # AMD dialect headers — includes use "amd/include/..." relative paths
    ${TRITON_SOURCE_DIR}/third_party/amd/include
    ${TRITON_SOURCE_DIR}/third_party
    ${TRITON_BUILD_DIR}/third_party
    # LLVM/MLIR headers
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(TLXMemOpsPlugin PRIVATE
    ${TRITON_LIBRARY}
    MLIRPass
)

set_target_properties(TLXMemOpsPlugin PROPERTIES
    CXX_STANDARD 17
)

target_compile_options(TLXMemOpsPlugin PRIVATE
    -fvisibility=hidden
    -fno-exceptions
    -fno-rtti
)
