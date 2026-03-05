cmake_minimum_required(VERSION 3.18)
project(Minion VERSION 1.2.0 LANGUAGES C CXX)

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
file(GLOB MINION_CEC_SOURCE_FILES minion/cec/*.cpp)

include(FetchContent)
option(MINION_FETCH_EIGEN "Fetch Eigen headers when Eigen3 package is not found" ON)

# + -------------------------- +
# | Eigen (find or download)   |
# + -------------------------- +
find_package(Eigen3 QUIET CONFIG)
if(NOT Eigen3_FOUND)
    if(NOT MINION_FETCH_EIGEN)
        message(FATAL_ERROR "Eigen3 not found. Install Eigen3 or set MINION_FETCH_EIGEN=ON.")
    endif()

    # Fetch Eigen headers only. Do not configure Eigen's CMake project:
    # this keeps Minion's build graph clean and avoids optional Fortran/BLAS paths.
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(eigen)
    if(NOT eigen_POPULATED)
        FetchContent_Populate(eigen)
    endif()

    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${eigen_SOURCE_DIR}"
    )
endif()

# + ----------------------------------------- +
# | Compiled algorithms core target           |
# + ----------------------------------------- +
set(MINION_SOURCE_FILES
    ${MINION_ALGO_IMPL_SOURCES}
    ${MINION_CEC_SOURCE_FILES}
)

add_library(minion SHARED ${MINION_SOURCE_FILES})

target_include_directories(minion PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/minion/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/minion/cec>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/lbfgspp/include>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/minion>
)
target_link_libraries(minion PUBLIC Eigen3::Eigen)
set_target_properties(minion PROPERTIES POSITION_INDEPENDENT_CODE ON)

# + -------------------------- +
# | Optional C++ examples      |
# + -------------------------- +
if(MINION_BUILD_EXAMPLES)
    add_executable(main_mini examples/main_minimizer.cpp)
    target_link_libraries(main_mini PRIVATE minion)

    add_executable(main_cec examples/main_cec.cpp)
    target_link_libraries(main_cec PRIVATE minion)

    add_executable(main_cec_2 examples/main_cec_2.cpp)
    target_link_libraries(main_cec_2 PRIVATE minion)

    add_executable(main_cec_multi examples/main_cec_multi.cpp)
    target_link_libraries(main_cec_multi PRIVATE minion)

    add_executable(main_complexity examples/main_complexity.cpp)
    target_link_libraries(main_complexity PRIVATE minion)

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

    pybind11_add_module(minionpycpp ${MINION_PYBIND_SOURCES})
    target_link_libraries(minionpycpp PRIVATE minion Eigen3::Eigen)
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
if(DEFINED SKBUILD AND SKBUILD)
    if(TARGET minionpycpp)
        install(TARGETS minionpycpp
            LIBRARY DESTINATION minionpy/lib
        )
    endif()
else()
    include(CMakePackageConfigHelpers)

    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/minion/include/ DESTINATION include/minion FILES_MATCHING PATTERN "*.h")
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/minion/cec/ DESTINATION include/minion FILES_MATCHING PATTERN "*.h")
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external/lbfgspp/include/ DESTINATION include)

    install(TARGETS minion
        EXPORT MinionTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
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
endif()

# + -------------------------- +
# | CPack (binary packaging)   |
# + -------------------------- +
set(CPACK_PACKAGE_NAME "minion")
set(CPACK_PACKAGE_VENDOR "Minion")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Minion optimization library")
if(UNIX AND NOT APPLE)
    set(CPACK_GENERATOR "TGZ;DEB")
    set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
    set(CPACK_DEBIAN_PACKAGE_SECTION "science")
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Khoirul Faiq Muzakka")
elseif(APPLE)
    set(CPACK_GENERATOR "TGZ")
endif()
include(CPack)
