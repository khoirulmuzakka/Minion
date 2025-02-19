# Changelog

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
- Improved `DA`with `L_BFGS_B`as the local search. 

### Changed
- Fine tune `DA`hyperparameters

### Fixed
- Fixed problem with github CI for wheel generations. 

