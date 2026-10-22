cmake_minimum_required(VERSION 3.22.1)
project(llama_jni)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- llama.cpp build options (sirf library chahiye, tools/examples nahi) ----
set(BUILD_SHARED_LIBS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON    OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL            OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP           OFF CACHE BOOL "" FORCE)  # ggml ka apna threadpool use hoga
set(GGML_NATIVE           OFF CACHE BOOL "" FORCE)

# Modern phones (2019+) ke ARM cores dotprod + fp16 support karte hain -> fast inference.
# Agar kisi purane phone pe app "SIGILL" se crash ho, to ye line comment kar do.
set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+fp16" CACHE STRING "" FORCE)

# llama.cpp source yahan clone hona chahiye: app/src/main/cpp/llama.cpp
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp build-llama)

add_library(llama_jni SHARED llama_jni.cpp)

target_link_libraries(llama_jni
        llama
        android
        log)

# Android 15+ devices ke 16KB page size support ke liye
target_link_options(llama_jni PRIVATE "-Wl,-z,max-page-size=16384")
