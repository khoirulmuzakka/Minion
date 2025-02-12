cmake_minimum_required(VERSION 3.18)
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
    #set(CMAKE_C_COMPILER /usr/bin/clang)
    #set(CMAKE_CXX_COMPILER /usr/bin/clang++)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wno-unused-result")
elseif (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
elseif (UNIX AND CMAKE_COMPILER_IS_GNUCXX)
    #set(CMAKE_C_COMPILER /usr/bin/gcc)
    #set(CMAKE_CXX_COMPILER /usr/bin/g++)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wno-unused-result")
else()
    message(WARNING "Unknown platform. Please check compiler settings.")
endif()


# + ---------------------------------------------- +
# | Find pybind11 and Eigen|
# + ---------------------------------------------- +
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND python -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(pybind11_DIR ${PYBIND11_CMAKE_DIR})
find_package(pybind11 REQUIRED CONFIG)

if(DEFINED ENV{EIGEN3_INCLUDE_DIR})
    set(EIGEN3_INCLUDE_DIR $ENV{EIGEN3_INCLUDE_DIR})
else()
    find_package(Eigen3 3.0 REQUIRED)
endif()


# + --------------- +
# | LBFGSpp LIBRARY |
# + --------------- +
add_library(lbfgspp INTERFACE)
target_include_directories(lbfgspp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/external/lbfgspp/include)
target_link_libraries(lbfgspp INTERFACE Eigen3::Eigen)


# + -------------------------- +
# | Remove old compiled libs |
# + --------------------------- +
file(REMOVE_RECURSE ${CMAKE_CURRENT_SOURCE_DIR}/minionpy/lib)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/minionpy/lib)

include_directories(
    /usr/include/eigen3/
    external/lbfgspp/include
    minion/include
    minion/cec 
    ${EIGEN3_INCLUDE_DIR}
)

# + -------------------------------------------+
# | Compile minion library and some main files |
# + ------------------------------------------ +
#scikit-build is used to generate python wheels. During wheel generation using cibuildwheel, this c++ binaries are not needed.
if(NOT SKBUILD) 
    # --------------------- compile minion library -----------------------------------------#
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

    # --------------------- compile examples ---------------------------------------#

    # 1. example/main_cec.cpp
    #------------------------
    add_executable(main_cec examples/main_cec.cpp)
    target_link_libraries(main_cec PRIVATE minion)
    if (MSVC)
        set_target_properties(main_cec PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/$<0:>)
    else()
        set_target_properties(main_cec PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
    endif()

    # 2. example/main_minimizer.cpp
    #------------------------
    add_executable(main_mini examples/main_minimizer.cpp)
    target_link_libraries(main_mini PRIVATE minion)
    if (MSVC)
        set_target_properties(main_mini PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/$<0:>)
    else()
        set_target_properties(main_mini PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
    endif()

endif()

# + ---------------------------------------------- +
# | Compile Minionpycpp library for python binding|
# + ---------------------------------------------- +
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


# + ---------------------------------------------- +
# | Set install directory                          |
# + ---------------------------------------------- +
install(TARGETS minionpycpp
    LIBRARY DESTINATION minionpy/lib
)




