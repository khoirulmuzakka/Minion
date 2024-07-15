cmake_minimum_required(VERSION 3.12)
project(Minion)

set(CMAKE_CXX_STANDARD 11)

# Compiler optimizations
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# Specify pybind11 include directory
set(PYBIND11_INCLUDE_DIR "C:/Users/khoir/AppData/Roaming/Python/Python312/site-packages/pybind11/include")

# Find pybind11
set(pybind11_DIR "C:/Users/khoir/AppData/Roaming/Python/Python312/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

# Add include directories
include_directories(
    include
    ${PYBIND11_INCLUDE_DIR}
)

# Add source files
file(GLOB MINION_SOURCE_FILES src/*.cpp)
list(REMOVE_ITEM MINION_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp)

file(GLOB PYMINION_SOURCE_FILES src/*.cpp)

# Create the DLL (shared library)
add_library(minion SHARED ${MINION_SOURCE_FILES})
target_link_libraries(minion)

# Set output directory for shared library
set_target_properties(minion PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

# Create the Python extension module (.pyd)
pybind11_add_module(pyminioncpp ${PYMINION_SOURCE_FILES})

# Set output directory for Python module (.pyd)
set_target_properties(pyminioncpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

