Compilation and Installation
=============================

MinionPy (Python)
-----------------

Install from PyPI:

.. code-block:: shell

   pip install --upgrade minionpy

Quick check:

.. code-block:: python

   import minionpy as mpy


Build Minion From Source
------------------------

Build requirements:

- CMake >= 3.18
- C++17 compiler
- pybind11 (only when building the Python extension)

On Windows, use Visual C++ Build Tools.

Build with helper scripts:

- Linux/macOS: ``./compile.sh``
- Windows: ``compile.bat``

Or build manually with CMake.

Minimal C++ setup (compiled library):

.. code-block:: shell

   cmake -S . -B build \
     -DMINION_BUILD_CEC=OFF \
     -DMINION_BUILD_PYTHON=OFF \
     -DMINION_BUILD_EXAMPLES=OFF
   cmake --build build --config Release

Build all optional components:

.. code-block:: shell

   cmake -S . -B build \
     -DMINION_BUILD_CEC=ON \
     -DMINION_BUILD_PYTHON=ON \
     -DMINION_BUILD_EXAMPLES=ON
   cmake --build build --config Release


Install as a CMake Package
--------------------------

Install Minion so downstream projects can use ``find_package(Minion CONFIG REQUIRED)``:

.. code-block:: shell

   cmake -S . -B build \
     -DMINION_BUILD_CEC=ON \
     -DMINION_BUILD_PYTHON=OFF \
     -DMINION_BUILD_EXAMPLES=OFF
   cmake --build build --config Release
   cmake --install build --prefix /path/to/minion-install

This installs:

- headers under ``include/``
- compiled Minion library under ``lib/`` (for example ``libminion.so`` on Linux)
- CMake package files under ``lib/cmake/Minion``
- optional compiled CEC library ``libminion_cec`` when ``MINION_BUILD_CEC=ON``
- CEC input data under ``cec_input_data/`` when ``MINION_BUILD_CEC=ON``


Use Minion in Your CMake Project
--------------------------------

1. Point CMake to your Minion install prefix:

.. code-block:: shell

   cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/minion-install

2. In your project ``CMakeLists.txt``:

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.18)
   project(MyApp LANGUAGES CXX)

   find_package(Minion CONFIG REQUIRED)

   add_executable(my_app src/main.cpp src/solver.cpp)
   target_link_libraries(my_app PRIVATE minion)
   # Optional: only if you use CEC benchmarks
   # target_link_libraries(my_app PRIVATE minion_cec)

   target_compile_features(my_app PRIVATE cxx_std_17)

3. Include Minion headers in your source:

.. code-block:: cpp

   #include <minion.h>


Alternative: FetchContent (No System Install)
---------------------------------------------

If you do not want to pre-install Minion:

.. code-block:: cmake

   include(FetchContent)

   find_package(Minion QUIET CONFIG)
   if(NOT Minion_FOUND)
     FetchContent_Declare(
       minion
       GIT_REPOSITORY https://github.com/khoirulmuzakka/Minion.git
       GIT_TAG main
       GIT_SHALLOW TRUE
     )
     set(MINION_BUILD_CEC OFF CACHE BOOL "" FORCE)
     set(MINION_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
     set(MINION_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
     FetchContent_MakeAvailable(minion)
   endif()

   add_executable(my_app src/main.cpp)
   target_link_libraries(my_app PRIVATE minion)


Use Local MinionPy Build
------------------------

If you built locally (without ``pip install``), import from the repository checkout:

.. code-block:: python

   import sys
   sys.path.append("/path/to/Minion")
   import minionpy as mpy
