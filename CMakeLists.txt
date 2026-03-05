cmake_minimum_required(VERSION 3.18)
project(Minion VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()

if (APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wno-unused-result")
elseif (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
elseif (UNIX AND CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wno-unused-result")
else()
    message(WARNING "Unknown platform. Please check compiler settings.")
endif()

set(_default_python OFF)
if (DEFINED SKBUILD AND SKBUILD)
    set(_default_python ON)
endif()

option(MINION_BUILD_CEC "Build compiled CEC library" OFF)
option(MINION_BUILD_PYTHON "Build Python extension module" ${_default_python})
option(MINION_BUILD_EXAMPLES "Build C++ examples" OFF)

file(GLOB MINION_ALGO_IMPL_SOURCES minion/src/*.cpp)
list(REMOVE_ITEM MINION_ALGO_IMPL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/minion/src/bindings.cpp)

include(FetchContent)

# + -------------------------- +
# | Eigen (find or download)   |
# + -------------------------- +
find_package(Eigen3 QUIET CONFIG)
if(NOT Eigen3_FOUND)
    # Disable Eigen's test targets to avoid Qt AUTOGEN warnings from Eigen CMake files.
    set(BUILD_TESTING OFF CACHE BOOL "Disable testing (Eigen dependency)" FORCE)
    set(EIGEN_BUILD_BLAS OFF CACHE BOOL "Disable Eigen BLAS (avoids Fortran toolchain requirements)" FORCE)
    set(EIGEN_BUILD_LAPACK OFF CACHE BOOL "Disable Eigen LAPACK (avoids Fortran toolchain requirements)" FORCE)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "Disable Eigen docs" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "Disable Eigen tests" FORCE)
    FetchContent_MakeAvailable(eigen)
endif()

# + ----------------------------------------- +
# | Header-only algorithms public target      |
# + ----------------------------------------- +
add_library(minion INTERFACE)
target_include_directories(minion INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/minion/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/minion/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/lbfgspp/include>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/minion/src>
)
target_link_libraries(minion INTERFACE
    Eigen3::Eigen
)

# + -------------------------- +
# | Optional compiled CEC lib  |
# + -------------------------- +
if(MINION_BUILD_CEC OR MINION_BUILD_PYTHON OR MINION_BUILD_EXAMPLES)
    file(GLOB MINION_CEC_SOURCE_FILES minion/cec/*.cpp)
    add_library(Minion_cec STATIC ${MINION_CEC_SOURCE_FILES})
    target_include_directories(Minion_cec PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/minion/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/minion/cec>
        $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(Minion_cec PUBLIC Eigen3::Eigen)

    if (MSVC)
        set_target_properties(Minion_cec PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib/$<0:>
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib/$<0:>
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib/$<0:>
            WINDOWS_EXPORT_ALL_SYMBOLS TRUE
        )
    else()
        set_target_properties(Minion_cec PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
        )
    endif()
endif()

# + -------------------------- +
# | Optional C++ examples      |
# + -------------------------- +
if(MINION_BUILD_EXAMPLES)
    add_executable(main_mini examples/main_minimizer.cpp)
    target_link_libraries(main_mini PRIVATE minion)
    target_compile_definitions(main_mini PRIVATE MINION_ALGORITHMS_IMPLEMENTATION)

    add_executable(main_cec examples/main_cec.cpp)
    target_link_libraries(main_cec PRIVATE minion Minion_cec)
    target_compile_definitions(main_cec PRIVATE MINION_ALGORITHMS_IMPLEMENTATION)

    add_executable(main_cec_2 examples/main_cec_2.cpp)
    target_link_libraries(main_cec_2 PRIVATE minion Minion_cec)
    target_compile_definitions(main_cec_2 PRIVATE MINION_ALGORITHMS_IMPLEMENTATION)

    add_executable(main_cec_multi examples/main_cec_multi.cpp)
    target_link_libraries(main_cec_multi PRIVATE minion Minion_cec)
    target_compile_definitions(main_cec_multi PRIVATE MINION_ALGORITHMS_IMPLEMENTATION)

    add_executable(main_complexity examples/main_complexity.cpp)
    target_link_libraries(main_complexity PRIVATE minion Minion_cec)
    target_compile_definitions(main_complexity PRIVATE MINION_ALGORITHMS_IMPLEMENTATION)

    if (MSVC)
        set_target_properties(main_mini main_cec main_cec_2 main_cec_multi main_complexity
            PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/$<0:>
        )
    else()
        set_target_properties(main_mini main_cec main_cec_2 main_cec_multi main_complexity
            PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
        )
    endif()
endif()

# + -------------------------- +
# | Optional Python extension  |
# + -------------------------- +
if(MINION_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
    execute_process(
        COMMAND python -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(pybind11_DIR ${PYBIND11_CMAKE_DIR})
    find_package(pybind11 REQUIRED CONFIG)

    file(REMOVE_RECURSE ${CMAKE_CURRENT_SOURCE_DIR}/minionpy/lib)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/minionpy/lib)

    set(MINION_PYBIND_SOURCES minion/src/bindings.cpp)
    if(TARGET Minion_cec)
        list(APPEND MINION_PYBIND_SOURCES ${MINION_CEC_SOURCE_FILES})
    endif()

    pybind11_add_module(minionpycpp ${MINION_PYBIND_SOURCES})
    target_link_libraries(minionpycpp PRIVATE minion Eigen3::Eigen)
    if(TARGET Minion_cec)
        target_link_libraries(minionpycpp PRIVATE Minion_cec)
    endif()
    target_include_directories(minionpycpp PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/minion/include
        ${CMAKE_CURRENT_SOURCE_DIR}/minion/cec
    )

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
endif()

# + -------------------------- +
# | Install + CMake package    |
# + -------------------------- +
include(CMakePackageConfigHelpers)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/minion/include/ DESTINATION include/minion FILES_MATCHING PATTERN "*.h")
install(FILES ${MINION_ALGO_IMPL_SOURCES} DESTINATION include/minion/src)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/minion/cec/ DESTINATION include/minion FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external/lbfgspp/include/ DESTINATION include)

install(TARGETS minion
    EXPORT MinionTargets
)
if(TARGET Minion_cec)
    install(TARGETS Minion_cec
        EXPORT MinionTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
endif()
if(TARGET minionpycpp)
    install(TARGETS minionpycpp
        LIBRARY DESTINATION minionpy/lib
    )
endif()

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/MinionConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/MinionConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/MinionConfig.cmake
    INSTALL_DESTINATION lib/cmake/Minion
)

install(EXPORT MinionTargets
    FILE MinionTargets.cmake
    DESTINATION lib/cmake/Minion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/MinionConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/MinionConfigVersion.cmake
    DESTINATION lib/cmake/Minion
)
