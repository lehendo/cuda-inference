# Benchmark harness. cuDNN and CUTLASS are optional: when missing, the corresponding baseline is
# compiled out (the harness then records an explicit "unsupported: not found at configure time" row
# instead of silently dropping the baseline).

find_package(Git QUIET)
set(GIE_GIT_REV "unknown")
if(GIT_FOUND)
  execute_process(COMMAND ${GIT_EXECUTABLE} describe --always --dirty --tags
                  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} OUTPUT_VARIABLE rev
                  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE rev_rc)
  if(rev_rc EQUAL 0 AND rev)
    set(GIE_GIT_REV "${rev}")
  endif()
endif()

set(GIE_BENCH_SOURCES benchmark_main.cpp cudnn_baseline.cpp cutlass_baseline.cu)
add_executable(gie_bench ${GIE_BENCH_SOURCES})
target_link_libraries(gie_bench PRIVATE gie)
target_include_directories(gie_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(gie_bench PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${GIE_CXX_WARN}> ${GIE_CUDA_FLAGS})
target_compile_definitions(gie_bench PRIVATE GIE_GIT_REV="${GIE_GIT_REV}")

# ---- cuDNN ------------------------------------------------------------------------------------
set(GIE_HAVE_CUDNN OFF)
if(GIE_ENABLE_CUDNN)
  find_path(CUDNN_INCLUDE_DIR cudnn.h
            HINTS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDAToolkit_ROOT}
            PATH_SUFFIXES include include/12 include/cudnn)
  find_library(CUDNN_LIBRARY NAMES cudnn libcudnn.so.9 libcudnn.so.8
               HINTS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDAToolkit_ROOT}
               PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)
  if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
    set(GIE_HAVE_CUDNN ON)
    get_filename_component(CUDNN_LIB_DIR ${CUDNN_LIBRARY} DIRECTORY)
    message(STATUS "cuDNN baseline: ENABLED (${CUDNN_LIBRARY})")
    set_source_files_properties(cudnn_baseline.cpp PROPERTIES COMPILE_DEFINITIONS GIE_HAS_CUDNN=1)
    target_include_directories(gie_bench PRIVATE ${CUDNN_INCLUDE_DIR})
    target_link_libraries(gie_bench PRIVATE ${CUDNN_LIBRARY})
    set_target_properties(gie_bench PROPERTIES BUILD_RPATH ${CUDNN_LIB_DIR} INSTALL_RPATH ${CUDNN_LIB_DIR})
  else()
    message(STATUS "cuDNN baseline: DISABLED (cudnn.h / libcudnn not found; set -DCUDNN_ROOT=...)")
  endif()
else()
  message(STATUS "cuDNN baseline: DISABLED (GIE_ENABLE_CUDNN=OFF)")
endif()

# ---- CUTLASS (header only) ------------------------------------------------------------------------
set(GIE_HAVE_CUTLASS OFF)
if(GIE_ENABLE_CUTLASS)
  if(CUTLASS_DIR AND EXISTS "${CUTLASS_DIR}/include/cutlass/cutlass.h")
    set(GIE_HAVE_CUTLASS ON)
    message(STATUS "CUTLASS baseline: ENABLED (${CUTLASS_DIR})")
    set_source_files_properties(cutlass_baseline.cu PROPERTIES COMPILE_DEFINITIONS GIE_HAS_CUTLASS=1)
    target_include_directories(gie_bench PRIVATE ${CUTLASS_DIR}/include)
  else()
    message(STATUS "CUTLASS baseline: DISABLED (set -DCUTLASS_DIR=<checkout containing include/cutlass>)")
  endif()
else()
  message(STATUS "CUTLASS baseline: DISABLED (GIE_ENABLE_CUTLASS=OFF)")
endif()

# ---- autotuning driver / Experiment B ---------------------------------------------------------------
add_executable(gie_tune tune_main.cpp)
target_link_libraries(gie_tune PRIVATE gie)
target_compile_options(gie_tune PRIVATE ${GIE_CXX_WARN})

add_executable(gie_graph_bench graph_bench.cpp)
target_link_libraries(gie_graph_bench PRIVATE gie)
target_compile_options(gie_graph_bench PRIVATE ${GIE_CXX_WARN})

add_executable(gie_rmsnorm_bench rmsnorm_bench.cpp)
target_link_libraries(gie_rmsnorm_bench PRIVATE gie)
target_compile_options(gie_rmsnorm_bench PRIVATE ${GIE_CXX_WARN})
