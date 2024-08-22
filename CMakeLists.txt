cmake_minimum_required(VERSION 3.12)
project(Minion)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Compiler optimizations
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wno-unused-result")
endif()

if (POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()

# Specify pybind11 include directory
if(WIN32)
    set(PYBIND11_INCLUDE_DIR "C:/Users/khoir/AppData/Roaming/Python/Python312/site-packages/pybind11/include")
    set(pybind11_DIR "C:/Users/khoir/AppData/Roaming/Python/Python312/site-packages/pybind11/share/cmake/pybind11")
else()
    #find_package(pybind11 REQUIRED)
    set(PYBIND11_INCLUDE_DIR "/home/muzakka/anaconda3/lib/python3.11/site-packages/pybind11/include")
    set(pybind11_DIR "/home/muzakka/anaconda3/lib/python3.11/site-packages/pybind11/share/cmake/pybind11")
endif()
find_package(pybind11 REQUIRED)


# Add include directories
include_directories(
    include
    ${PYBIND11_INCLUDE_DIR}
    external
)

# --------------------- dynamic minion library -----------------------------------------#
#file(GLOB MINION_SOURCE_FILES src/*.cpp external/*.cpp)
#list(REMOVE_ITEM MINION_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp)

#add_library(minion SHARED ${MINION_SOURCE_FILES})
#target_link_libraries(minion)
#set_target_properties(minion PROPERTIES
#    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
#    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
#)

# --------------------- static minion library -----------------------------------------#
file(GLOB MINION_SOURCE_FILES src/*.cpp external/*.cpp)
list(REMOVE_ITEM MINION_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp)

add_library(minion_static STATIC ${MINION_SOURCE_FILES})
target_link_libraries(minion_static)
set_target_properties(minion_static PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

# --------------------- pyminion library ---------------------------------------#
file(GLOB PYMINION_SOURCE_FILES src/*.cpp external/*.cpp)
pybind11_add_module(pyminioncpp ${PYMINION_SOURCE_FILES})
set_target_properties(pyminioncpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

# --------------------- main files ---------------------------------------------#
add_executable(main_cec examples/main_cec.cpp)
target_link_libraries(main_cec PRIVATE minion_static)
set_target_properties(main_cec PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_executable(main_repeat examples/main_cec_repeat.cpp)
target_link_libraries(main_repeat PRIVATE minion_static)
set_target_properties(main_repeat PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_executable(main_repeat2 examples/main_cec_repeat2.cpp)
target_link_libraries(main_repeat2 PRIVATE minion_static)
set_target_properties(main_repeat2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

