cmake_minimum_required(VERSION 3.20)
project(velomocap_kernels LANGUAGES CXX)

# Standalone build of the VeloMocap native kernels.
#
# The Rust crate `velomocap-cpp` compiles exactly these sources through
# build.rs (cc), so this file exists for reproducible standalone builds,
# for inspecting generated assembly, and for benchmarking the kernels in
# isolation:
#
#   cmake -S cpp -B build/native -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/native --config Release
#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(velomocap_kernels STATIC src/kernels.cpp)
target_include_directories(velomocap_kernels PUBLIC include)

option(VELOMOCAP_AVX2 "Build with AVX2/FMA code generation" ON)

if(MSVC)
  target_compile_options(velomocap_kernels PRIVATE /O2 /fp:precise /W3 /EHsc)
  if(VELOMOCAP_AVX2)
    target_compile_options(velomocap_kernels PRIVATE /arch:AVX2)
  endif()
else()
  target_compile_options(velomocap_kernels PRIVATE -O3 -fPIC -Wall -Wextra)
  if(VELOMOCAP_AVX2)
    target_compile_options(velomocap_kernels PRIVATE -mavx2 -mfma)
  endif()
endif()

# Kernel self-test / benchmark driver (optional, C++ only).
add_executable(kernel_bench EXCLUDE_FROM_ALL tools/kernel_bench.cpp)
target_link_libraries(kernel_bench PRIVATE velomocap_kernels)
