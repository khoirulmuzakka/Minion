cmake_minimum_required(VERSION 3.12)
project(Minion)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Handle CMP0148 policy
if (POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()

# Compiler settings based on platform
if (APPLE)
    set(CMAKE_C_COMPILER /usr/bin/clang)
    set(CMAKE_CXX_COMPILER /usr/bin/clang++)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wno-unused-result")
elseif (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
elseif (UNIX AND CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_C_COMPILER /usr/bin/gcc)
    set(CMAKE_CXX_COMPILER /usr/bin/g++)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wno-unused-result")
else()
    message(WARNING "Unknown platform. Please check compiler settings.")
endif()


find_package(pybind11 REQUIRED CONFIG)

# Include directories
include_directories(
    minion/include
    minion/cec
)

# --------------------- dynamic minion library -----------------------------------------#
# Collect source files for the core library
file(GLOB MINION_SOURCE_FILES minion/src/*.cpp minion/cec/*.cpp)
list(REMOVE_ITEM MINION_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/minion/src/bindings.cpp)

# Define the shared library
add_library(minion SHARED ${MINION_SOURCE_FILES})
target_link_libraries(minion)
set_target_properties(minion PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

# --------------------- minionpy library ---------------------------------------#
# Collect source files for the Python bindings
file(GLOB minionpy_SOURCE_FILES minion/src/*.cpp minion/cec/*.cpp)

# Define the Python module
pybind11_add_module(minionpycpp ${minionpy_SOURCE_FILES})
set_target_properties(minionpycpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/minionpy/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/minionpy/lib
)

# Set up custom output directories for builds
message(STATUS "Configuration complete for Minion and MinionPy.")
