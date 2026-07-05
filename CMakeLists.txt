cmake_minimum_required(VERSION 3.18)
project(Minion VERSION 1.6.1 LANGUAGES C CXX)

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
option(MINION_BUILD_LIBCMAES "Build libcmaes comparison example" ON)
option(MINION_FETCH_LIBCMAES "Fetch libcmaes when building the libcmaes example" ON)
option(MINION_BUILD_BBOB2009 "Build bbob2009 comparison example" ON)
option(MINION_FETCH_BBOB2009 "Fetch COCO when building the bbob2009 example" ON)

file(GLOB MINION_ALGO_IMPL_SOURCES CONFIGURE_DEPENDS minion/src/*.cpp)
list(REMOVE_ITEM MINION_ALGO_IMPL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/minion/src/bindings.cpp)
file(GLOB MINION_CEC_SOURCE_FILES CONFIGURE_DEPENDS minion/cec/*.cpp)

include(FetchContent)
option(MINION_FETCH_EIGEN "Fetch Eigen headers when Eigen3 package is not found" ON)

# + -------------------------- +
# | Eigen (find or download)   |
# + -------------------------- +
find_package(Eigen3 QUIET CONFIG)
set(_MINION_EIGEN3_PACKAGE_USABLE FALSE)
if(Eigen3_FOUND AND TARGET Eigen3::Eigen)
    get_target_property(_MINION_EIGEN3_INCLUDE_DIRS Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    if(_MINION_EIGEN3_INCLUDE_DIRS)
        foreach(_MINION_EIGEN3_INCLUDE_DIR IN LISTS _MINION_EIGEN3_INCLUDE_DIRS)
            if(EXISTS "${_MINION_EIGEN3_INCLUDE_DIR}")
                set(_MINION_EIGEN3_PACKAGE_USABLE TRUE)
                break()
            endif()
        endforeach()
    endif()
endif()

if(NOT _MINION_EIGEN3_PACKAGE_USABLE)
    set(Eigen3_FOUND FALSE)
endif()

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
    endif()
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${eigen_SOURCE_DIR}"
    )
endif()

if(MINION_BUILD_LIBCMAES AND NOT Eigen3_FOUND)
    # libcmaes expects a real Eigen3 package, so provide a tiny config shim
    # when we only have a fetched source tree.
    set(MINION_EIGEN3_PACKAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_minion_eigen3_package")
    file(MAKE_DIRECTORY "${MINION_EIGEN3_PACKAGE_DIR}")
    file(WRITE "${MINION_EIGEN3_PACKAGE_DIR}/Eigen3Config.cmake" "
if(NOT TARGET Eigen3::Eigen)\n
    add_library(Eigen3::Eigen INTERFACE IMPORTED)\n
    set_target_properties(Eigen3::Eigen PROPERTIES\n
        INTERFACE_INCLUDE_DIRECTORIES \"${eigen_SOURCE_DIR}\")\n
endif()\n
set(Eigen3_FOUND TRUE)\n
set(Eigen3_VERSION 3.4.0)\n")
    file(WRITE "${MINION_EIGEN3_PACKAGE_DIR}/Eigen3ConfigVersion.cmake" "
set(PACKAGE_VERSION \"3.4.0\")\n
if(PACKAGE_VERSION VERSION_LESS PACKAGE_FIND_VERSION)\n
    set(PACKAGE_VERSION_COMPATIBLE FALSE)\n
else()\n
    set(PACKAGE_VERSION_COMPATIBLE TRUE)\n
    if(PACKAGE_VERSION VERSION_EQUAL PACKAGE_FIND_VERSION)\n
        set(PACKAGE_VERSION_EXACT TRUE)\n
    endif()\n
endif()\n")
    set(Eigen3_DIR "${MINION_EIGEN3_PACKAGE_DIR}" CACHE PATH "Synthetic Eigen3 package dir" FORCE)
endif()

if(MINION_BUILD_LIBCMAES)
    set(LIBCMAES_BUILD_SHARED_LIBS ON CACHE BOOL "Build libcmaes shared library" FORCE)
    set(LIBCMAES_BUILD_PYTHON OFF CACHE BOOL "Build libcmaes Python bindings" FORCE)
    set(LIBCMAES_BUILD_TESTS OFF CACHE BOOL "Build libcmaes tests" FORCE)
    set(LIBCMAES_BUILD_EXAMPLES OFF CACHE BOOL "Build libcmaes examples" FORCE)
    set(LIBCMAES_USE_OPENMP ON CACHE BOOL "Use OpenMP in libcmaes" FORCE)
    set(LIBCMAES_ENABLE_SURROG ON CACHE BOOL "Enable libcmaes surrogates" FORCE)

    if(MINION_FETCH_LIBCMAES)
        FetchContent_Declare(
            libcmaes
            GIT_REPOSITORY https://github.com/CMA-ES/libcmaes.git
            # Pin to a real upstream release tag so FetchContent can resolve it reliably.
            GIT_TAG v0.10.2
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(libcmaes)
    else()
        message(FATAL_ERROR "MINION_BUILD_LIBCMAES requires MINION_FETCH_LIBCMAES=ON when no external libcmaes package is provided.")
    endif()
endif()

if(MINION_BUILD_BBOB2009)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    if(MINION_FETCH_BBOB2009)
        FetchContent_Declare(
            coco
            GIT_REPOSITORY https://github.com/numbbo/coco.git
            GIT_TAG master
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(coco)
    else()
        message(FATAL_ERROR "MINION_BUILD_BBOB2009 requires MINION_FETCH_BBOB2009=ON when no external COCO package is provided.")
    endif()

    set(MINION_COCO_COLORAMA_SHIM_DIR "${CMAKE_CURRENT_BINARY_DIR}/_minion_coco_python_shim")
    file(MAKE_DIRECTORY "${MINION_COCO_COLORAMA_SHIM_DIR}")
    file(WRITE "${MINION_COCO_COLORAMA_SHIM_DIR}/colorama.py" "
def init(*args, **kwargs):
    return None

class _Ansi:
    def __getattr__(self, name):
        return ''

Fore = _Ansi()
Style = _Ansi()
")

    set(MINION_COCO_C_DIR "${coco_SOURCE_DIR}/code-experiments/build/c")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${MINION_COCO_COLORAMA_SHIM_DIR}
            ${Python3_EXECUTABLE}
            scripts/fabricate
            --skip-java
            --skip-matlab
            --skip-octave
            --skip-python
            --skip-postprocessing
            --skip-rust
            --skip-unit-tests
            --skip-integration-tests
        WORKING_DIRECTORY ${coco_SOURCE_DIR}
        RESULT_VARIABLE _minion_coco_fabricate_result
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(NOT _minion_coco_fabricate_result EQUAL 0)
        message(FATAL_ERROR "Failed to run COCO fabricate step in ${coco_SOURCE_DIR}")
    endif()

    if(NOT EXISTS "${MINION_COCO_C_DIR}/coco.c")
        message(FATAL_ERROR "COCO fabricate did not generate ${MINION_COCO_C_DIR}/coco.c")
    endif()

    add_library(coco STATIC "${MINION_COCO_C_DIR}/coco.c")
    target_include_directories(coco PUBLIC "${MINION_COCO_C_DIR}")
    target_compile_features(coco PUBLIC c_std_99)
    if(UNIX AND NOT APPLE)
        target_link_libraries(coco PUBLIC m)
    endif()
    set_target_properties(coco PROPERTIES POSITION_INDEPENDENT_CODE ON)
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

    if(MINION_BUILD_LIBCMAES)
        add_executable(main_cec_multi_libcmaes examples/main_cec_multi_libcmaes.cpp)
        target_link_libraries(main_cec_multi_libcmaes PRIVATE minion minion_cec libcmaes::cmaes)
        minion_copy_runtime_deps(main_cec_multi_libcmaes)
    endif()

    if(MINION_BUILD_BBOB2009)
        add_library(bbob2009 STATIC external/bbob2009/bbob2009.cpp)
        target_include_directories(bbob2009 PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/bbob2009>
            $<INSTALL_INTERFACE:include>
        )
        target_link_libraries(bbob2009 PUBLIC coco)
        set_target_properties(bbob2009 PROPERTIES POSITION_INDEPENDENT_CODE ON)

        add_executable(main_bbob_multi examples/main_bbob_multi.cpp)
        target_link_libraries(main_bbob_multi PRIVATE minion bbob2009)
        minion_copy_runtime_deps(main_bbob_multi)

        if(MINION_BUILD_LIBCMAES)
            add_executable(main_bbob_multi_libcmaes examples/main_bbob_multi_libcmaes.cpp)
            target_link_libraries(main_bbob_multi_libcmaes PRIVATE minion bbob2009 libcmaes::cmaes)
            minion_copy_runtime_deps(main_bbob_multi_libcmaes)
        endif()
    endif()

    if (MSVC)
        set(_minion_example_targets main_mini)
        if(MINION_BUILD_CEC)
            list(APPEND _minion_example_targets main_complexity main_cec main_cec_2 main_cec_multi)
            if(MINION_BUILD_LIBCMAES)
                list(APPEND _minion_example_targets main_cec_multi_libcmaes)
            endif()
        endif()
        if(MINION_BUILD_BBOB2009)
            list(APPEND _minion_example_targets main_bbob_multi)
            if(MINION_BUILD_LIBCMAES)
                list(APPEND _minion_example_targets main_bbob_multi_libcmaes)
            endif()
        endif()
        set_target_properties(${_minion_example_targets}
            PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/$<0:>
        )
        unset(_minion_example_targets)
    else()
        set(_minion_example_targets main_mini)
        if(MINION_BUILD_CEC)
            list(APPEND _minion_example_targets main_complexity main_cec main_cec_2 main_cec_multi)
            if(MINION_BUILD_LIBCMAES)
                list(APPEND _minion_example_targets main_cec_multi_libcmaes)
            endif()
        endif()
        if(MINION_BUILD_BBOB2009)
            list(APPEND _minion_example_targets main_bbob_multi)
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

    file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/minionpy)

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
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/minionpy/$<0:>
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/minionpy/$<0:>
        )
    else()
        set_target_properties(minionpycpp PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/minionpy
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/minionpy
        )
    endif()

    if(MINION_BUILD_LIBCMAES)
        pybind11_add_module(libcmaes_bridge external/libcmaes/python/libcmaes_bridge.cpp)
        target_link_libraries(libcmaes_bridge PRIVATE libcmaes::cmaes)
        if(WIN32)
            target_compile_definitions(libcmaes_bridge PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
        endif()

        if (MSVC)
            set_target_properties(libcmaes_bridge PROPERTIES
                LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/minionpy/$<0:>
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/minionpy/$<0:>
            )
        else()
            set_target_properties(libcmaes_bridge PROPERTIES
                LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/minionpy
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/minionpy
            )
        endif()
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
            LIBRARY DESTINATION minionpy
            RUNTIME DESTINATION minionpy
        )
    endif()
    if(MINION_BUILD_CEC)
        install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cec_input_data/
            DESTINATION minionpy/cec_input_data
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
            LIBRARY DESTINATION minionpy
            RUNTIME DESTINATION minionpy
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
