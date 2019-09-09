cmake_minimum_required(VERSION 3.3)
project(Minion)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -Wno-comment -Wno-return-type -Wno-unused-value -fpermissive")
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_BUILD_TYPE DEBUG)

OPTION (USE_OpenMP "Use OpenMP" ON)
IF(USE_OpenMP)
  FIND_PACKAGE(OpenMP)
  IF(OPENMP_FOUND)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  ENDIF()
ENDIF()

include_directories( ./include)
file(GLOB_RECURSE LIB_SOURCE_FILES "src/*.cpp")

#compile the Minion lib
add_library(minion STATIC ${LIB_SOURCE_FILES})
target_link_libraries(minion armadillo)
set_target_properties(minion PROPERTIES ARCHIVE_OUTPUT_DIRECTORY  ./lib)

#compile MainPSOTest.cpp
add_executable(PSOTest  examples/MainPSOTest.cpp)
target_link_libraries(PSOTest minion)
set_target_properties(PSOTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
