Compilation and Installation
=============================

MinionPy Installation
----------------------

MinionPy can be installed directly via PIP:

.. code-block:: shell

   pip install minionpy

To verify the installation, run:

.. code-block:: python

   import minionpy as mpy

If MinionPy is not installed via PIP and you want to use the locally compiled version, ensure that the correct path is added to `sys.path`:

.. code-block:: python

   import sys
   sys.path.append("/path/to/minionpy/")  # Adjust this to the correct directory
   import minionpy as mpy


Minion (C++) Compilation
------------------------

To use Minion in a C++ project, you need to compile the Minion dynamic library from source and link it to your project. The compilation process also generates the necessary library for the MinionPy Python wrapper.

1. **Install Dependencies**

   Before compiling Minion, install the required dependencies:

   - **CMake**: A tool for managing the build process.
   - **pybind11**: A header-only library for creating Python bindings in C++.

   *Note for Windows users:*  
   To compile the source code, you need Microsoft C++ Build Tools. Download them from:

   - [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

2. **Compile Minion and MinionPy**

   Follow these steps to compile Minion:

   1. Modify the `CMakeLists.txt` if necessary (e.g., specify the location of `pybind11`).
   2. Run the appropriate compilation script:
      - On Linux/macOS:  

        .. code-block:: shell

           ./compile.sh

      - On Windows:  

        .. code-block:: shell

           compile.bat

   This will automatically build the required libraries and place them in the appropriate directories.

3. **Upon Successful Compilation**

   After compilation, the following files will be generated in the `./lib` directory:

   - **For C++ development:**
     - `minion.dll` (Windows) or `minion.so` (Linux/macOS)
   
   - **For Python development:**
     - `minionpy*.so` (Python wrapper for Minion)

   The C++ dynamic library (`minion.dll` or `minion.so`) is used in C++ applications, while `minionpy*.so` allows Python integration.  
   The Python wrapper code can be found in the `./minionpy` directory.


4. **Using the Minion Library in C++**

   After compiling the library, you can use Minion in your C++ projects by including the necessary headers and linking against the compiled library.

   Example:

   .. code-block:: cpp

      #include "minion/minion.h"

   To properly link against Minion, modify your `CMakeLists.txt` as follows:

   .. code-block:: cmake

      add_executable(main_mini examples/main_minimizer.cpp)
      target_link_libraries(main_mini PRIVATE minion)

      if (MSVC)
          set_target_properties(main_mini PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/$<0:>)
      else()
          set_target_properties(main_mini PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
      endif()


5. **Using MinionPy in Python**

   If MinionPy is not installed via PIP but compiled locally, manually add the `minionpy` directory to `sys.path` before importing:

   .. code-block:: python

      import sys
      sys.path.append("/path/to/minionpy/")  # Adjust to the correct directory
      import minionpy as mpy

   This ensures Python can find and import MinionPy.

