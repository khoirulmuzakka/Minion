cmake_minimum_required(VERSION 3.18)
project(Minion VERSION 1.5.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
elseif (MINGW OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wno-unused-result")
elseif (APPLE)
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
option(MINION_BUILD_TESTS "Build repository test targets" OFF)

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

    # Fetch Eigen headers only. SOURCE_SUBDIR points to a non-project path so
    # MakeAvailable populates the source tree without adding Eigen's CMake project.
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
        SOURCE_SUBDIR _minion_no_cmake_project
    )
    FetchContent_MakeAvailable(eigen)

    if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${eigen_SOURCE_DIR}"
        )
    endif()
endif()

# + ----------------------------------------- +
# | Compiled algorithms core target           |
# + ----------------------------------------- +
set(MINION_CORE_SOURCE_FILES ${MINION_ALGO_IMPL_SOURCES})
add_library(minion SHARED ${MINION_CORE_SOURCE_FILES})

target_include_directories(minion PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/minion/include>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/minion>
)
if(WIN32)
    target_compile_definitions(minion PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
target_link_libraries(minion PUBLIC Eigen3::Eigen)
set_target_properties(minion PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(WIN32)
    # Export all symbols so MSVC generates minion.lib for downstream linking (e.g. pybind module).
    set_target_properties(minion PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

if(MINION_BUILD_CEC)
    add_library(minion_cec SHARED ${MINION_CEC_SOURCE_FILES})
    target_include_directories(minion_cec PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/minion/cec>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/minion/include>
        $<INSTALL_INTERFACE:include>
        $<INSTALL_INTERFACE:include/minion>
    )
    if(WIN32)
        target_compile_definitions(minion_cec PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
    endif()
    target_link_libraries(minion_cec PUBLIC Eigen3::Eigen)
    set_target_properties(minion_cec PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if(WIN32)
        set_target_properties(minion_cec PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()

    # Keep CEC input data available next to build-tree binaries/libraries.
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/cec_input_data DESTINATION ${CMAKE_BINARY_DIR})
endif()

# + -------------------------- +
# | Optional C++ examples      |
# + -------------------------- +
if(MINION_BUILD_EXAMPLES)
    function(minion_copy_runtime_deps target_name)
        if(WIN32)
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_RUNTIME_DLLS:${target_name}>
                    $<TARGET_FILE_DIR:${target_name}>
                COMMAND_EXPAND_LISTS
            )
        endif()
    endfunction()

    add_executable(main_mini examples/main_minimizer.cpp)
    target_link_libraries(main_mini PRIVATE minion)
    minion_copy_runtime_deps(main_mini)

    if(MINION_BUILD_CEC)
        add_executable(main_complexity examples/main_complexity.cpp)
        target_link_libraries(main_complexity PRIVATE minion minion_cec)
        minion_copy_runtime_deps(main_complexity)

        add_executable(main_cec examples/main_cec.cpp)
        target_link_libraries(main_cec PRIVATE minion minion_cec)
        minion_copy_runtime_deps(main_cec)

        add_executable(main_cec_2 examples/main_cec_2.cpp)
        target_link_libraries(main_cec_2 PRIVATE minion minion_cec)
        minion_copy_runtime_deps(main_cec_2)

        add_executable(main_cec_multi examples/main_cec_multi.cpp)
        target_link_libraries(main_cec_multi PRIVATE minion minion_cec)
        minion_copy_runtime_deps(main_cec_multi)
    endif()

    if (MSVC)
        set(_minion_example_targets main_mini)
        if(MINION_BUILD_CEC)
            list(APPEND _minion_example_targets main_complexity main_cec main_cec_2 main_cec_multi)
        endif()
        set_target_properties(${_minion_example_targets}
            PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/$<0:>
        )
        unset(_minion_example_targets)
    else()
        set(_minion_example_targets main_mini)
        if(MINION_BUILD_CEC)
            list(APPEND _minion_example_targets main_complexity main_cec main_cec_2 main_cec_multi)
        endif()
        set_target_properties(${_minion_example_targets}
            PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
        )
        unset(_minion_example_targets)
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

    set(MINION_PYBIND_SOURCES
        minion/src/bindings.cpp
        ${MINION_ALGO_IMPL_SOURCES}
        ${MINION_CEC_SOURCE_FILES}
    )

    pybind11_add_module(minionpycpp ${MINION_PYBIND_SOURCES})
    target_link_libraries(minionpycpp PRIVATE Eigen3::Eigen ${CMAKE_DL_LIBS})
    target_include_directories(minionpycpp PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/minion/include
        ${CMAKE_CURRENT_SOURCE_DIR}/minion/cec
    )
    if(WIN32)
        target_compile_definitions(minionpycpp PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    endif()

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
# | Optional repository tests  |
# + -------------------------- +
if(MINION_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
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

    if(MINION_BUILD_CEC)
        install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/minion/include/ DESTINATION include/minion FILES_MATCHING PATTERN "*.h")
    else()
        install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/minion/include/ DESTINATION include/minion
            FILES_MATCHING PATTERN "*.h" PATTERN "minion_cec.h" EXCLUDE)
    endif()
    if(MINION_BUILD_CEC)
        install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/minion/cec/ DESTINATION include/minion FILES_MATCHING PATTERN "*.h")
        install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cec_input_data/ DESTINATION cec_input_data)
    endif()

    install(TARGETS minion
        EXPORT MinionTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
    if(TARGET minion_cec)
        install(TARGETS minion_cec
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
endif()

# + -------------------------- +
# | CPack (binary packaging)   |
# + -------------------------- +
set(CPACK_PACKAGE_NAME "minion")
set(CPACK_PACKAGE_VENDOR "Minion")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Minion optimization library")
set(MINION_PACKAGE_SUFFIX "" CACHE STRING "Optional suffix appended to the CPack package file name")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}")
if(MINION_PACKAGE_SUFFIX)
    string(APPEND CPACK_PACKAGE_FILE_NAME "-${MINION_PACKAGE_SUFFIX}")
endif()
if(UNIX AND NOT APPLE)
    set(CPACK_GENERATOR "TGZ;DEB")
    set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
    set(CPACK_DEBIAN_PACKAGE_SECTION "science")
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Khoirul Faiq Muzakka")
elseif(APPLE)
    set(CPACK_GENERATOR "TGZ")
endif()
include(CPack)
