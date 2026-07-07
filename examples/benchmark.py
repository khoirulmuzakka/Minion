from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import minionpy as mpy


def _rosenbrock_batch(X):
    values = []
    for x in X:
        total = 0.0
        for i in range(len(x) - 1):
            a = x[i + 1] - x[i] * x[i]
            b = 1.0 - x[i]
            total += 100.0 * a * a + b * b
        values.append(total)
    return values


def demo_cec() -> None:
    print("\n=== CEC example ===")
    cec = mpy.CEC2017Functions(function_number=1, dimension=10)
    bounds = [(-100.0, 100.0)] * 10
    x0 = [[0.0] * 10]

    optimizer = mpy.Minimizer(
        func=cec,
        x0=x0,
        bounds=bounds,
        algo="ARRDE",
        maxevals=2000,
        callback=None,
        seed=42,
        options=None,
    )

    result = optimizer.optimize()
    print(f"CEC result: best f = {result.fun}, f_opt = {cec.f_opt}, nfev = {result.nfev}")


def demo_bbob() -> None:
    print("\n=== BBOB2009 example ===")
    bbob = mpy.BBOB2009Problem(function_number=1, dimension=10)
    bounds = bbob.bounds
    x0 = [bbob.initial_solution]

    optimizer = mpy.Minimizer(
        func=bbob,
        x0=x0,
        bounds=bounds,
        algo="ARRDE",
        maxevals=2000,
        callback=None,
        seed=42,
        options=None,
    )

    result = optimizer.optimize()
    print(f"BBOB result: best f = {result.fun}, f_opt = {bbob.f_opt}, nfev = {result.nfev}")


def demo_benchmark_runner() -> None:
    print("\n=== Benchmark helper example ===")
    result = mpy.run_benchmark(
        mode="cec", #bbob
        num_runs=5,
        dimension=10,
        algo="ARRDE",
        popsize=0,
        year=2017,
        max_evals=20000,
        nthreads=8,
        acc=8,
        dump_results=True,
        results_folder="./results",
        log_min_ev=False,
    )

    print(f"Benchmark helper finished with {len(result.results)} run(s)")
    print(f"Benchmark helper results file: {result.results_file or '(not dumped)'}")
    if result.results:
        print(f"Benchmark helper first value: {result.results[0][0]}")


def main() -> int:
    print("Minion benchmark examples")
    print("Running three demos: CEC, BBOB2009, and the benchmark helper")

    demo_cec()
    demo_bbob()
    demo_benchmark_runner()

    print("\nAll benchmark demos completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
