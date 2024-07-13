# Minion: Derivative-Free Optimization Library

Minion is a library for derivative-free optimization algorithms currently implemented in Python and Cython. It offers a collection of state-of-the-art optimization techniques designed to tackle complex optimization problems efficiently, especially when gradients are unavailable or costly to compute.

## Key Features

- **Optimization Algorithms:**
  - Includes Generalized Artificial Bee Colony (GABC), Adaptive Differential Evolution (M_LJADE_AMR and M_LSHADE_AMR).
- **Customizable:**
  - Define your objective functions, constraints, and termination criteria for flexible optimization scenarios.
- **Python and Cython Implementation:**
  - Combines the ease of Python with the performance of Cython for critical sections, ensuring both readability and speed.
- **Parallelizable:**
  - Assumes vectorized function evaluations, enabling easy integration with multithreading or multiprocessing for enhanced computational efficiency.

## Algorithms Included

- **Global Best Artificial Bee Colony (GABC):**
  - Enhances exploration and exploitation capabilities using generalized search mechanisms inspired by honey bee foraging behavior.
- **Modified JADE with Linear Population Size Reduction and Adaptive Mutation Rate (M_LJADE_AMR):**
  - State-of-the-art variant of DE.
- **Modified SHADE with Linear Population Size Reduction and Adaptive Mutation Rate (M_LSHADE_AMR):**
  - State-of-the-art variant of DE.

## Future Additions

Planned expansions include more evolutionary strategies, swarm intelligence methods, and metaheuristic algorithms based on community feedback and research advancements.

## In Progress

- **C++ Version:**
  - Currently in development to extend Minion's capabilities and performance.

## How to Compile and Use Minion Library

1. **Install Dependencies**
   - Run `install_dependencies.bat` file to install the required dependencies.

2. **Compile Cython Library**
   - Run `cython_compile.bat` file to compile the Cython library.
   - *Note:* To compile the source code, you need Microsoft C++ Build Tools. Download from [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

3. **Upon Successful Compilation**
   - The library file (*.pyd) should be in `./lib`.
   - You can import the library as:
     ```python
     import sys
     custom_path = 'path/to/lib/'
     sys.path.append(custom_path)
     from minion import *
     ```
