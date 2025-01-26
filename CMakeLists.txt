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
if (MSVC)
    set_target_properties(minion PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib/$<0:>
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib/$<0:>
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib/$<0:>
        WINDOWS_EXPORT_ALL_SYMBOLS TRUE
    )
else()
    set_target_properties(minion PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
    )
endif()

# --------------------- minionpy library ---------------------------------------#
# Collect source files for the Python bindings
file(GLOB minionpy_SOURCE_FILES minion/src/*.cpp minion/cec/*.cpp)

# Define the Python module
pybind11_add_module(minionpycpp ${minionpy_SOURCE_FILES})
if (MSVC)
    set_target_properties(minionpycpp PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/minionpy/lib/$<0:>
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/minionpy/lib/$<0:>
    )
else()
    set_target_properties(minionpycpp PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/minionpy/lib
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/minionpy/lib
    )
endif()

# Set up custom output directories for builds
message(STATUS "Configuration complete for Minion and MinionPy.")

####### Compile examples ##############
add_executable(main_cec examples/main_cec.cpp)
target_link_libraries(main_cec PRIVATE minion)
if (MSVC)
    set_target_properties(main_cec PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/$<0:>)
else()
    set_target_properties(main_cec PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
endif()


install(TARGETS minion
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install the Python module
install(TARGETS minionpycpp
    LIBRARY DESTINATION minionpy/lib
)