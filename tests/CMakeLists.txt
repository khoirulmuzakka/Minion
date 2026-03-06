cmake_minimum_required(VERSION 3.18)
project(minion_fetch_usage_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Keep everything local to this test build folder.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(Minion QUIET CONFIG)
#find_package(Minion REQUIRED CONFIG)
if(NOT Minion_FOUND)
    FetchContent_Declare(
        minion
        GIT_REPOSITORY https://github.com/khoirulmuzakka/Minion.git
        GIT_TAG main
        GIT_SHALLOW TRUE
    )
    # Dependency mode: algorithms only.
    set(MINION_BUILD_CEC ON CACHE BOOL "Disable CEC" FORCE)
    set(MINION_BUILD_PYTHON OFF CACHE BOOL "Disable Python extension" FORCE)
    set(MINION_BUILD_EXAMPLES OFF CACHE BOOL "Disable examples" FORCE)
    FetchContent_MakeAvailable(minion)
endif()

add_executable(minion_test main.cpp)
target_link_libraries(minion_test PRIVATE minion minion_cec)

enable_testing()
add_test(NAME minion_integration COMMAND minion_test)
