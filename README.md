# Minion: Derivative-Free Optimization Library

Minion is a library for derivative-free optimization algorithms, implemented in both C++ and Python. It offers a suite of state-of-the-art optimization techniques, specifically designed to efficiently solve complex optimization problems where gradients are unavailable.

## Key Features

- **Optimization Algorithms:**
  - Includes state-of-the arts variants of differential evolution, hybrid Grey Wolf - Differential Evolution (GWO-DE), and more. 
- **Parallelizable:**
  - Always assumes vectorized function evaluations, enabling easy integration with multithreading or multiprocessing for enhanced computational efficiency.

## Algorithms Included
- **Nelder-Mead:**
  - Widely-used optimization algorithm.
- **JADE:**
  - State-of-the-art variant of Differential Evolution (DE).
- **SHADE:**
  - State-of-the-art variant of Differential Evolution (DE).
- **LSHADE:**
  - State-of-the-art variant of Differential Evolution (DE).
- **NLSHADE-RSP:**
  - State-of-the-art variant of Differential Evolution (DE).
- **GWO-DE:**
  - Hybrid Grey Wolf-Deifferential Dvolution.
- **ARRDE: Adaptive restart-refine DE**
  - State-of-the-art variant of Differential Evolution (DE).

## How to Compile and Use Minion Library

1. **Install Dependencies**
   - Install CMake, pybind11.
   - *Note for Windows users:* To compile the source code, you need Microsoft C++ Build Tools. Download from [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

2. **Compile Minion and Pyminion Library**
   - Modify `CMakeLists.txt` to reflect the location of pybind11.
   - Run `compile.bat` file to compile the library.

3. **Upon Successful Compilation**
   - The dynamic library (`minion.dll` or `minion.so` and `pyminioncpp*.pyd`) should be in `./lib/Release`. `minion.dll` (Windows) or `minion.so` (Unix) is the dynamic library to be used in C++ development, while `pyminioncpp*.pyd` is used for Python import. The Python wrapper code can be found in `./pyminion`.

   - In Python, you can import the library as:
     ```python
     import sys
     custom_path = 'path/to/folderthatcontainpyminionfolder'
     sys.path.append(custom_path)
     import pyminion
     ```

## Example: Minimizing the Rosenbrock Function

Below is an example demonstrating how to use the `MFADE` class from the Minion library to minimize the Rosenbrock function.

### Rosenbrock Function Minimization

Create a file named `example.cpp` with the following content:

```cpp
// example.cpp

#include <iostream>
#include <vector>
#include <utility>
#include <cmath>
#include <functional>
#include "minion.h"

// Rosenbrock function definition
std::vector<double> rosenbrock(const std::vector<std::vector<double>>& x, void* data) {
    std::vector<double> results;
    for (const auto& xi : x) {
        double a = 1.0;
        double b = 100.0;
        double sum = 0.0;
        for (size_t i = 0; i < xi.size() - 1; ++i) {
            sum += b * std::pow((xi[i + 1] - xi[i] * xi[i]), 2) + std::pow((a - xi[i]), 2);
        }
        results.push_back(sum);
    }
    return results;
}

int main() {
    // Define the bounds for the decision variables
    std::vector<std::pair<double, double>> bounds = { {-5.0, 5.0}, {-5.0, 5.0} };

    // Initial guess
    std::vector<double> x0 = { 0.0, 0.0 };

    // Population size
    int population_size = 30;

    // Maximum number of function evaluations
    int maxevals = 100000;

    // Create an instance of M_LJADE_AMR
    minion::FADE(rosenbrock, bounds, nullptr, x0, population_size, maxevals);

    // Run the optimizer
    minion::MinionResult result = optimizer.optimize();

    // Output the results
    std::cout << "Best solution found: ";
    for (const auto& val : result.x) {
        std::cout << val << " ";
    }
    std::cout << "\nBest objective value: " << result.fun << std::endl;

    return 0;
}
