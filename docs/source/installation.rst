Compilation and Installation
============================

Minion can be used in two ways:

- as the Python package ``minionpy``;
- as a native C++ library linked from another C++ project.

For Python users, start with the PyPI package. If a wheel is not available for
your platform or Python version, build the Python interface from source. For
C++ users, use a release package when available, or build the native library
with CMake.


Python Package
--------------

Install MinionPy from PyPI:

.. code-block:: shell

   python -m pip install --upgrade minionpy

Check the installation:

.. code-block:: shell

   python -c "import minionpy; print(minionpy.__version__)"

PyPI wheels are not available for every operating system, Python version, or
CPU architecture. If ``pip`` cannot find a compatible wheel, use
`Build MinionPy From Source`_.


Released Native Packages
------------------------

Linux users can install the ``.deb`` package from the GitHub release assets:

.. code-block:: shell

   sudo dpkg -i minion_<version>_<arch>.deb
   sudo apt-get install -f

macOS users can install from the release archive. If the downloaded asset is a
``tgz.zip`` file, unzip it first:

.. code-block:: shell

   unzip minion-<version>-macos.tgz.zip
   mkdir -p /tmp/minion_pkg
   tar -xzf minion-<version>-macos.tgz -C /tmp/minion_pkg
   sudo rsync -a /tmp/minion_pkg/ /usr/local/

Windows users should build from source with CMake.


Source Build Requirements
-------------------------

Required for native C++ builds:

- Git, when cloning the repository;
- CMake >= 3.18;
- a C++17 compiler, such as GCC, Clang, or MSVC;
- Eigen3, or network access so CMake can fetch Eigen automatically.

Required for Python builds, including the default helper-script builds:

- Python 3.9 or newer;
- Python development headers;
- ``pybind11``.

Optional for documentation builds:

- Doxygen;
- Pandoc;
- Sphinx;
- sphinx-rtd-theme;
- nbsphinx;
- Breathe.

On Ubuntu 24.04, install the native build tools with:

.. code-block:: shell

   sudo apt update
   sudo apt install git cmake build-essential

For Python builds on Ubuntu 24.04, also install:

.. code-block:: shell

   sudo apt install python3-dev python3-pip python3-pybind11

For documentation builds on Ubuntu 24.04, also install:

.. code-block:: shell

   sudo apt install doxygen pandoc python3-sphinx python3-sphinx-rtd-theme \
     python3-nbsphinx python3-breathe

Clone the repository:

.. code-block:: shell

   git clone https://github.com/khoirulmuzakka/Minion.git
   cd Minion


Build From Source With CMake
----------------------------

Helper Scripts
~~~~~~~~~~~~~~

The helper scripts build the C++ libraries, C++ examples, and Python extension
module:

- Linux/macOS: ``./compile.sh``
- Windows: ``compile.bat``

They configure CMake with:

- ``MINION_BUILD_CEC=ON``
- ``MINION_BUILD_EXAMPLES=ON``
- ``MINION_BUILD_PYTHON=ON``

Because Python bindings are enabled by default in the helper scripts, Python
development headers and ``pybind11`` must be available. To build only the
native C++ library, use the manual CMake build below with
``MINION_BUILD_PYTHON=OFF``.

Alternatively, edit ``compile.sh`` or ``compile.bat`` before running them and
set the CMake options to match the build you want.

On Linux/macOS:

.. code-block:: shell

   ./compile.sh

For a debug build:

.. code-block:: shell

   ./compile.sh --debug

On Windows, run from a Developer Command Prompt or another shell where CMake
and Visual Studio 2022 are available:

.. code-block:: bat

   compile.bat

The helper scripts also try to build the documentation after compiling the
project. If Doxygen, Pandoc, or one of the required Python documentation
packages is missing, documentation generation is skipped with a warning. If all
documentation tools are installed, the Sphinx build imports ``minionpy`` from
the build tree.


Manual Native-Only CMake Build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a native C++ build without Python bindings:

.. code-block:: shell

   cmake -S . -B build \
     -DCMAKE_BUILD_TYPE=Release \
     -DMINION_BUILD_CEC=ON \
     -DMINION_BUILD_PYTHON=OFF \
     -DMINION_BUILD_EXAMPLES=ON
   cmake --build build --config Release

Install to a chosen prefix:

.. code-block:: shell

   cmake --install build --prefix /path/to/minion-install

If Eigen3 is not installed, CMake fetches Eigen automatically by default. To
require a system Eigen installation instead, add:

.. code-block:: shell

   -DMINION_FETCH_EIGEN=OFF


Build MinionPy From Source
--------------------------

Use this route when no PyPI wheel is available for your platform, or when you
want to use the Python interface from a local checkout.

Building MinionPy from source is the same CMake build as the native library,
but with ``MINION_BUILD_PYTHON=ON``:

.. code-block:: shell

   cmake -S . -B build \
     -DCMAKE_BUILD_TYPE=Release \
     -DMINION_BUILD_CEC=ON \
     -DMINION_BUILD_PYTHON=ON \
     -DMINION_BUILD_EXAMPLES=OFF
   cmake --build build --config Release

The Python package code is in the ``minionpy`` directory. The compiled Python
extension is written to ``minionpy/lib``. After the build, import MinionPy from
the repository root:

.. code-block:: python

   import minionpy
   print(minionpy.__version__)

If you run Python from another directory, add the repository root
(the parent directory of ``minionpy``) to ``sys.path`` before importing
``minionpy``:

.. code-block:: python

   import sys
   sys.path.insert(0, "/path/to/Minion")

   import minionpy
   print(minionpy.__version__)

To select a specific Python interpreter, pass it to CMake:

.. code-block:: shell

   cmake -S . -B build \
     -DMINION_BUILD_PYTHON=ON \
     -DPython3_EXECUTABLE=/path/to/python3

Make sure the ``python`` command on ``PATH`` belongs to the same environment
and can import ``pybind11``.


Build the Documentation
-----------------------

The documentation uses Doxygen for C++ API extraction and Sphinx for HTML
pages. The Sphinx configuration imports ``minionpy``, so build MinionPy first:

.. code-block:: shell

   cmake -S . -B build \
     -DCMAKE_BUILD_TYPE=Release \
     -DMINION_BUILD_CEC=ON \
     -DMINION_BUILD_PYTHON=ON
   cmake --build build --config Release
   doxygen Doxyfile
   cd docs
   make html

The generated HTML documentation is written to ``docs/build/html``. On
Windows, use ``docs\\make.bat html`` if GNU Make is not available.


Use Minion in a C++ Project
---------------------------

Include the main header:

.. code-block:: cpp

   #include <minion/minion.h>

Use ``find_package`` for an installed Minion package, with ``FetchContent`` as
a fallback:

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
   # Optional: only if you use the CEC benchmark suite
   # target_link_libraries(my_app PRIVATE minion_cec)
