"""Python convenience wrapper for the C++ benchmark runner.

This module only prepares a :class:`~minionpy.minionpycpp.BenchmarkConfig`
and forwards execution to the compiled C++ implementation.

The heavy benchmark work runs inside the pybind11 binding with the GIL
released, so Python threads are not blocked during optimization.
"""

from __future__ import annotations

from pathlib import Path
from .minionpycpp import Benchmark, BenchmarkConfig, BenchmarkMode, run_benchmark as _run_benchmark

__all__ = ["benchmark", "run_benchmark", "Benchmark", "BenchmarkConfig", "BenchmarkMode"]


def _parse_mode(mode: str | BenchmarkMode) -> BenchmarkMode:
    """Normalize a benchmark mode value.

    Args:
        mode: Either a :class:`BenchmarkMode` enum value or a string such as
            ``"cec"`` or ``"bbob"``.

    Returns:
        The corresponding :class:`BenchmarkMode`.

    Raises:
        ValueError: If ``mode`` is not recognized.
    """
    if isinstance(mode, BenchmarkMode):
        return mode
    normalized = str(mode).strip().lower()
    if normalized == "cec":
        return BenchmarkMode.Cec
    if normalized == "bbob":
        return BenchmarkMode.Bbob
    raise ValueError("mode must be 'cec' or 'bbob'")


def benchmark(
    mode: str | BenchmarkMode,
    num_runs: int = 1,
    dimension: int = 10,
    algo: str = "ARRDE",
    popsize: int = 0,
    year: int = 2017,
    max_evals: int = -1,
    nthreads: int = 1,
    acc: int = 8,
    dump_results: bool = True,
    results_folder: str | Path = ".",
    log_min_ev: bool = False,
):
    """Run the benchmark driver from Python.

    This is a thin wrapper over the C++ benchmark engine. It builds a
    :class:`BenchmarkConfig` and passes it to the compiled
    :func:`run_benchmark` binding.

    The actual optimization loop, progress reporting, and file dumping all
    happen in C++ with the GIL released.

    Args:
        mode: Benchmark family, either ``"cec"``, ``"bbob"``, or a
            :class:`BenchmarkMode` value.
        num_runs: Number of independent runs to execute.
        dimension: Problem dimension passed to the benchmark runner.
        algo: Optimizer name, for example ``"ARRDE"`` or ``"jSO"``.
        popsize: Population size used by the optimizer.
        year: Benchmark suite year.
        max_evals: Maximum number of evaluations. ``-1`` uses the default
            derived from the dimension.
        nthreads: Number of worker threads.
        acc: Decimal precision used for console output and dumps.
        dump_results: Whether to write the benchmark result table to disk.
        results_folder: Output folder for result files.
        log_min_ev: Whether to record min-EV traces.

    Returns:
        A :class:`BenchmarkResult` object containing the raw matrix of results
        and the results filename when dumping is enabled.

    Raises:
        ValueError: If ``mode`` is invalid.
        RuntimeError: If the C++ benchmark validation fails.
    """

    config = BenchmarkConfig()
    config.mode = _parse_mode(mode)
    config.num_runs = int(num_runs)
    config.dimension = int(dimension)
    config.algo = str(algo)
    config.population_size = int(popsize)
    config.year = int(year)
    config.max_evals = int(max_evals)
    config.nthreads = int(nthreads)
    config.acc = int(acc)
    config.dump_results = bool(dump_results)
    config.results_folder = str(Path(results_folder))
    config.log_min_ev = bool(log_min_ev)
    return _run_benchmark(config)


run_benchmark = benchmark
run_benchmark.__doc__ = benchmark.__doc__
