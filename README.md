# Minion: Derivative-Free Optimization Library

![CI](https://github.com/khoirulmuzakka/Minion/actions/workflows/ci.yml/badge.svg)



The Minion library is a toolkit for derivative-free optimization, designed to solve complex optimization problems where gradients are unavailable or unreliable. It includes a collection of state-of-the-art optimization algorithms that have won top positions in the IEEE Congress on Evolutionary Computation (CEC) competitions, which are not commonly available in standard optimization libraries such as SciPy, NLopt, OptimLib, pyGMO, and pagmo2.

Minion also serves as a testing ground for researchers to develop and evaluate new optimization algorithms. It incorporates recent benchmark functions from the CEC competitions held in 2011, 2014, 2017, 2019, 2020, and 2022, offering a robust environment for algorithm testing and comparison. 

Currently, Minion implements several leading algorithms, including: JADE, L-SHADE (1st place in CEC2014), jSO (1st place in CEC2017),  j2020 (3rd place in CEC2020), NL-SHADE-RSP (1st place in CEC2021), LSRTDE (1st place in CEC2024), ARRDE (Adaptive Restart-Refine Differential Evolution, our own algorithm). These algorithms generally offer superior robustness and faster convergence compared to basic differential evolution algorithms. Additionally, basic optimization methods, such as Nelder-Mead and the original Differential Evolution, are also included in the library.

Most of the algorithms  implemented in Minion are population-based, which makes them inherently parallelizable and well-suited for fast, efficient processing. To further enhance performance, Minion is optimized for vectorized functions, enabling seamless integration with multithreading and multiprocessing capabilities.

Minion is implemented in C++ with a Python wrapper, making it accessible and functional in both languages. The library has been tested on Windows 11, Linux Ubuntu 24.04, and macOS Sequoia 15.

## Key Features

- **Optimization Algorithms:**
  - Includes state-of-the arts variants of differential evolution, hybrid Grey Wolf - Differential Evolution (GWO-DE), and more. 
- **Parallelizable:**
  - Always assumes vectorized function evaluations, enabling easy integration with multithreading or multiprocessing for enhanced computational efficiency.

## Algorithms Included
- Nelder-Mead
- **State-of-the-art variants of differential evolution** : JADE, SHADE, LSHADE (1st in CEC2014), NLSHADE-RSP (1st in CEC2021), j2020 (3rd in CEC2020), jSO (1st in CEC2017), and LSRTDE ((1st in CEC2024)
- **ARRDE: Adaptive restart-refine DE** : A new state-of-the-art variant of Differential Evolution (DE).

## CEC Benchmark Problems 
- CEC2011, CEC2014, CEC2017, CEC2019, CEC2020 and CEC2022

## How to Compile and Use Minion Library

1. **Install Dependencies**
   - Install CMake, pybind11.
   - *Note for Windows users:* To compile the source code, you need Microsoft C++ Build Tools. Download from [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

2. **Compile Minion and Minionpy Library**
   - Modify `CMakeLists.txt` to reflect the location of pybind11.
   - Run `compile.sh` file to compile the library.

3. **Upon Successful Compilation**
   - The dynamic library (`minion.dll` or `minion.so` and `minionpycpp*.so`) should be in `./lib`. `minion.dll` (Windows) or `minion.so` (Unix) is the dynamic library to be used in C++ development, while `minionpy*.so` is used for Python import. The Python wrapper code can be found in `./minionpy`.