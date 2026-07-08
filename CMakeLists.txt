cmake_minimum_required(VERSION 3.24)

# CUDA architectures must be chosen before project() enables the CUDA language, otherwise CMake
# substitutes the compiler default (sm_52). Ampere and newer only.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(DEFINED ENV{GPU_CUDA_ARCH} AND NOT "$ENV{GPU_CUDA_ARCH}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "$ENV{GPU_CUDA_ARCH}")
  else()
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
  endif()
endif()

project(gpu_inference_engine VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- options -----------------------------------------------------------------------------------
option(GIE_BUILD_TESTS "Build the correctness test suite" ON)
option(GIE_BUILD_BENCHMARKS "Build the benchmark / tuning tools" ON)
option(GIE_BUILD_EXAMPLES "Build the usage examples" ON)
option(GIE_ENABLE_CUDNN "Build the cuDNN baseline if cuDNN is found" ON)
option(GIE_ENABLE_ABLATIONS "Register diagnostic (numerically wrong) ablation kernels for time attribution" OFF)
option(GIE_ENABLE_CUTLASS "Build the CUTLASS baseline if CUTLASS is found" ON)
set(CUTLASS_DIR "" CACHE PATH "Root of a CUTLASS checkout (contains include/cutlass)")
set(CUDNN_ROOT "" CACHE PATH "Root containing include/cudnn.h and lib/libcudnn.so*")

# ---- CUDA architectures ---------------------------------------------------------------------------
# The Tensor Core path uses BF16/FP16 WMMA and the kernels assume sm_80+.
foreach(arch ${CMAKE_CUDA_ARCHITECTURES})
  string(REGEX REPLACE "[^0-9].*$" "" arch_num "${arch}")
  if(arch_num LESS 80)
    message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES contains ${arch}; this project supports sm_80 and newer only")
  endif()
endforeach()
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA toolkit ${CUDAToolkit_VERSION} at ${CUDAToolkit_BIN_DIR}")
find_package(Threads REQUIRED)
find_package(OpenMP COMPONENTS CXX)

# ---- warnings / flags ------------------------------------------------------------------------------
set(GIE_CXX_WARN -Wall -Wextra -Wshadow -Wno-unused-parameter)
set(GIE_CUDA_FLAGS
    "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>"
    "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wno-unused-parameter>")

# ---- build provenance (no git needed) -----------------------------------------------------------------
file(GLOB_RECURSE GIE_FP_FILES CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cuh ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.cu
     ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.h)
list(SORT GIE_FP_FILES)
set(GIE_FP_ACC "")
foreach(f ${GIE_FP_FILES})
  file(SHA256 ${f} fh)
  string(APPEND GIE_FP_ACC "${fh}")
endforeach()
string(SHA256 GIE_FP_FULL "${GIE_FP_ACC}")
string(SUBSTRING "${GIE_FP_FULL}" 0 16 GIE_SOURCE_FINGERPRINT)
string(REPLACE ";" "," GIE_CUDA_ARCHS_STR "${CMAKE_CUDA_ARCHITECTURES}")
string(TOUPPER "${CMAKE_BUILD_TYPE}" GIE_BT_UPPER)
set(GIE_CUDA_FLAGS_FULL "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${GIE_BT_UPPER}}")
set(GIE_CXX_FLAGS_FULL "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${GIE_BT_UPPER}}")
string(STRIP "${GIE_CUDA_FLAGS_FULL}" GIE_CUDA_FLAGS_FULL)
string(STRIP "${GIE_CXX_FLAGS_FULL}" GIE_CXX_FLAGS_FULL)
message(STATUS "source fingerprint: ${GIE_SOURCE_FINGERPRINT}")

# ---- library ------------------------------------------------------------------------------------
add_library(gie STATIC
  src/cpu/cpu_reference.cpp
  src/runtime/error.cpp
  src/runtime/tensor.cpp
  src/runtime/device_info.cpp
  src/runtime/conv2d.cpp
  src/runtime/registry.cpp
  src/runtime/execution_plan.cpp
  src/runtime/dispatcher.cpp
  src/runtime/perf_model.cpp
  src/runtime/timing.cpp
  src/runtime/json_lite.cpp
  src/runtime/build_info.cpp
  src/runtime/cuda_graph.cpp
  src/runtime/reference.cu
  src/runtime/testing.cu
  src/kernels/conv_common.cu
  src/kernels/direct_conv.cu
  src/kernels/elementwise.cu
  src/kernels/im2col.cu
  src/kernels/simt_tiled_conv.cu
  src/kernels/implicit_gemm.cu
  src/kernels/tensor_core_conv.cu
  src/kernels/tensor_core_mma.cu
  src/autotuner/tune_cache.cpp
  src/autotuner/autotuner.cpp
  src/profiling/ceilings.cu
  src/transformer/rmsnorm.cu
)
target_include_directories(gie PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/gie_build_info.h.in ${CMAKE_CURRENT_BINARY_DIR}/generated/gie_build_info.h @ONLY)
target_include_directories(gie PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
if(GIE_ENABLE_ABLATIONS)
  target_compile_definitions(gie PRIVATE GIE_ENABLE_ABLATIONS=1)
endif()
target_compile_options(gie PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${GIE_CXX_WARN}> ${GIE_CUDA_FLAGS})
target_link_libraries(gie PUBLIC CUDA::cudart CUDA::cublas Threads::Threads)
if(OpenMP_CXX_FOUND)
  target_link_libraries(gie PRIVATE OpenMP::OpenMP_CXX)
endif()
set_target_properties(gie PROPERTIES CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)

# NVTX range annotations (header-only NVTX3 ships with the toolkit) -- optional.
find_path(GIE_NVTX3_INCLUDE_DIR nvtx3/nvToolsExt.h HINTS ${CUDAToolkit_INCLUDE_DIRS} ${CUDAToolkit_TARGET_DIR}/include)
if(GIE_NVTX3_INCLUDE_DIR)
  message(STATUS "NVTX3 annotations: ENABLED (${GIE_NVTX3_INCLUDE_DIR})")
  target_include_directories(gie PUBLIC ${GIE_NVTX3_INCLUDE_DIR})
  target_compile_definitions(gie PUBLIC GIE_HAS_NVTX=1)
else()
  message(STATUS "NVTX3 annotations: disabled (nvtx3/nvToolsExt.h not found)")
endif()

# ---- tools / tests / examples -----------------------------------------------------------------------------
add_executable(gie_info tools/gie_info.cpp)
target_link_libraries(gie_info PRIVATE gie)
target_compile_options(gie_info PRIVATE ${GIE_CXX_WARN})

if(GIE_BUILD_EXAMPLES)
  add_executable(example_quickstart examples/quickstart.cpp)
  target_link_libraries(example_quickstart PRIVATE gie)
  target_compile_options(example_quickstart PRIVATE ${GIE_CXX_WARN})
endif()

if(GIE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(GIE_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
