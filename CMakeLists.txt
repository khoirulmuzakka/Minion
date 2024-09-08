cmake_minimum_required(VERSION 3.12)
project(Minion)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


if (POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()


if (APPLE)
    set(CMAKE_C_COMPILER /usr/bin/clang)
    set(CMAKE_CXX_COMPILER /usr/bin/clang++)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wno-unused-result")
    set(PYBIND11_INCLUDE_DIR "$ENV{CONDA_PREFIX}/lib/python3.10/site-packages/pybind11/include")
    set(pybind11_DIR "$ENV{CONDA_PREFIX}/lib/python3.10/site-packages/pybind11/share/cmake/pybind11")
    
elseif (WIN32)
    # Windows-specific settings
    set(PYBIND11_INCLUDE_DIR "C:/Users/khoir/AppData/Roaming/Python/Python312/site-packages/pybind11/include")
    set(pybind11_DIR "C:/Users/khoir/AppData/Roaming/Python/Python312/site-packages/pybind11/share/cmake/pybind11")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)

elseif (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_C_COMPILER "$ENV{CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc")
    set(CMAKE_CXX_COMPILER "$ENV{CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wno-unused-result")
    set(PYBIND11_INCLUDE_DIR "$ENV{CONDA_PREFIX}/lib/python3.11/site-packages/pybind11/include")
    set(pybind11_DIR "$ENV{CONDA_PREFIX}/lib/python3.11/site-packages/pybind11/share/cmake/pybind11")
endif()

find_package(pybind11 REQUIRED)


# Add include directories
include_directories(
    include
    ${PYBIND11_INCLUDE_DIR}
    external
)

# --------------------- dynamic minion library -----------------------------------------#
file(GLOB MINION_SOURCE_FILES src/*.cpp external/*.cpp)
list(REMOVE_ITEM MINION_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp)

add_library(minion SHARED ${MINION_SOURCE_FILES})
target_link_libraries(minion)
set_target_properties(minion PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

# --------------------- pyminion library ---------------------------------------#
file(GLOB PYMINION_SOURCE_FILES src/*.cpp external/*.cpp)
pybind11_add_module(pyminioncpp ${PYMINION_SOURCE_FILES})
set_target_properties(pyminioncpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

# --------------------- main files ---------------------------------------------#
add_executable(main_repeat examples/main_cec_repeat.cpp)
target_link_libraries(main_repeat PRIVATE minion)
set_target_properties(main_repeat PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_executable(main_repeat2 examples/main_cec_repeat2.cpp)
target_link_libraries(main_repeat2 PRIVATE minion)
set_target_properties(main_repeat2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

