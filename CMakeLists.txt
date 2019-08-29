cmake_minimum_required(VERSION 3.3)
project(Minion)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -Wno-comment -Wno-return-type -Wno-unused-value -fpermissive")
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_BUILD_TYPE DEBUG)

include_directories( ./include)
