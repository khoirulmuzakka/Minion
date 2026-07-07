# Changelog
## [1.7.0] - 2026-07-07
### Added
- Added benchmark functionality for CEC and BBOB2009 suites.
- Added BBOB2009 benchmark support to the Python and C++ APIs.

### Fixed
- Fixed memory leaks in the CEC benchmark implementations.

## [1.6.1] - 2026-06-13
### Changed
- Removed `tol` / `relTol` from the Minion API.
- Made algorithm name handling more robust across common naming variants.
- Updated `CEC2011` to more closely mirror the original MATLAB implementation.
- Streamlined the CEC API.
- Added `convergence_tol` to the options of supported algorithms.

## [1.5.0] - 2026-03-07
### Changed
- Reverted back to compiled library style.
- Fixed compilation issue on Windows MSVC due to min/max macros.

## [1.2.0] - 2026-03-05
### Added
- Added RCMAES
- Streamlined CMake usage.
- Header only style library

## [1.1.0] - 2025-11-23
### Added
- Added IMODE and AGSK
- CEC2011 is now completely rewritten in C++. No MATLAB!

## [1.0.1] - 2025-11-07
### Changed
- Fixed memory leaks.
- Version 1.0.1 now instead of 0.2.9

## [0.2.8] - 2025-11-06
### Changed
- More correct implementations of LSHADE_cnepsin and ARRDE.
## [0.2.7] - 2025-10-10
### Changed
- More correct implementations of LSHADE, jSO, JADE, DE, and ARRDE.

## [0.2.6] - 2025-10-06
### Added
- Added BIPOP_aCMAES

## [0.2.5] - 2025-10-05
### Added
- Added PSO, SPSO-2011, DMSPSO, LSHADE_cnepsin, and CMAES

## [0.2.4] - 2025-10-02
### Changed
- GIL handling during python calls. 
- An improved mutation implementation.

## [0.2.3] - 2025-08-21
### Changed
- Classical algorithms such as L-BFGS pick the best guess from the given guesses during initialization. 
- Stability improvements in L-BFGS and L-BFGS-B.

## [0.2.2] - 2025-07-31
### Changed
- Initial guess x0 is now a list of guesses. Thus, it is possible now to have more than one initial guesses. 

## [0.2.1] - 2025-03-01
### Added
- Fixed "none" unrecognized as bound_strategy

## [0.2.0] - 2025-02-24
### Added
- Added "minimum_population_size" option for ARRDE

## [0.1.9] - 2025-02-19
### Fixed
- Fixed L-BFGS-B violate bounds when calculating derivatives.

## [0.1.8] - 2025-02-19
### Added
- Implemented L-BFGS
- L-BFGS and L-BFGS-B use noise-robust Lanczos derivative 

### Changed
- Step size during derivative calculation in L-BFGS-B and L-BFGS is now adaptive.

### Fixed
- Fixed some typos in the docs.


## [0.1.7] - 2025-02-13
### Added
- Implemented `Process_Parallel` and `Thread_Parallel` for multiprocessing and multithreading support.
- Implemented `MinimizerBase::getBestFromHistory`.

### Changed
- `MinimizerBase::getBestFromHistory` is used for most algorithms now

### Fixed
- Resolved wrong function value when calling `optimize` function of `LSRTDE` and `NLDSHADE_RSP`

## [0.1.6] - 2025-02-11
### Added
- Implemented `L_BFGS_B` 
- Improved `DA` with `L_BFGS_B` as the local search.

### Changed
- Fine-tuned `DA` hyperparameters.

### Fixed
- Fixed problem with GitHub CI for wheel generation.
