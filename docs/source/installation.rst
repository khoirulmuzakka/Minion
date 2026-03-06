Compilation and Installation
============================

Python (MinionPy)
-----------------

Install from PyPI:

.. code-block:: shell

   pip install --upgrade minionpy

Quick check:

.. code-block:: python

   import minionpy as mpy


Native Installation by Platform
-------------------------------

- **Windows**: compile Minion from source with CMake.
- **Linux**: install the ``.deb`` package from GitHub Release assets.
- **macOS**: install from release archive (``.tgz`` / ``tgz.zip`` asset).

Linux ``.deb`` install:

.. code-block:: shell

   sudo dpkg -i minion_<version>_<arch>.deb
   sudo apt-get install -f

macOS ``.tgz`` / ``tgz.zip`` install:

.. code-block:: shell

   unzip minion-<version>-macos.tgz.zip
   tar -xzf minion-<version>-macos.tgz -C /tmp/minion_pkg
   sudo rsync -a /tmp/minion_pkg/ /usr/local/


Compilation From Scratch
------------------------

Build requirements:

- CMake >= 3.18
- C++17 compiler (GCC/Clang/MSVC)
- Eigen3 (or automatic fetch through CMake)
- Python 3 + pybind11 (only when building Python bindings)

Build with helper scripts:

- Linux/macOS: ``./compile.sh``
- Windows: ``compile.bat``

Manual build with CMake:

.. code-block:: shell

   cmake -S . -B build \
     -DMINION_BUILD_CEC=ON \
     -DMINION_BUILD_PYTHON=OFF \
     -DMINION_BUILD_EXAMPLES=ON
   cmake --build build --config Release

Install:

.. code-block:: shell

   cmake --install build --prefix /path/to/minion-install


Using Minion in a C++ Project
-----------------------------

Include header:

.. code-block:: cpp

   #include <minion/minion.h>

CMake setup with ``find_package`` first and ``FetchContent`` fallback:

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.18)
   project(MyApp LANGUAGES CXX)
   set(CMAKE_CXX_STANDARD 17)
   set(CMAKE_CXX_STANDARD_REQUIRED ON)

   include(FetchContent)

   find_package(Minion QUIET CONFIG)
   if(NOT Minion_FOUND)
     FetchContent_Declare(
       minion
       GIT_REPOSITORY https://github.com/khoirulmuzakka/Minion.git
       GIT_TAG main
       GIT_SHALLOW TRUE
     )
     set(MINION_BUILD_CEC ON CACHE BOOL "Build CEC library" FORCE)
     set(MINION_BUILD_PYTHON OFF CACHE BOOL "Disable Python extension" FORCE)
     set(MINION_BUILD_EXAMPLES OFF CACHE BOOL "Disable examples" FORCE)
     FetchContent_MakeAvailable(minion)
   endif()

   add_executable(my_app src/main.cpp)
   target_link_libraries(my_app PRIVATE minion)
   # Optional: only if you use CEC benchmark suite
   # target_link_libraries(my_app PRIVATE minion_cec)
