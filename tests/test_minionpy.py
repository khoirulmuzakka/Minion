from __future__ import annotations

import math
import os
import sys
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import minionpy


ALGORITHMS = (
    "DE",
    "LSHADE",
    "AGSK",
    "JADE",
    "j2020",
    "NLSHADE_RSP",
    "LSRTDE",
    "jSO",
    "IMODE",
    "ARRDE",
    "GWO_DE",
    "NelderMead",
    "ABC",
    "PSO",
    "SPSO2011",
    "DMSPSO",
    "LSHADE_cnEpSin",
    "CMAES",
    "RCMAES",
    "BIPOP_aCMAES",
    "DA",
    "L_BFGS_B",
    "L_BFGS",
)

SPHERE_UPPER = {algorithm: 1e-3 for algorithm in ALGORITHMS}
ROSENBROCK_UPPER = {algorithm: 10.0 for algorithm in ALGORITHMS}
EVAL_SLACK = 100


def sphere_batch(x_batch):
    return minionpy.sphere(np.asarray(x_batch, dtype=float)).tolist()


def rosenbrock_batch(x_batch):
    return minionpy.rosenbrock(np.asarray(x_batch, dtype=float)).tolist()


def make_cec2017_batch(function_number: int, dimension: int):
    cec2017 = minionpy.CEC2017Functions(function_number, dimension)

    def objective(x_batch):
        return list(cec2017(x_batch))

    return objective


def _run_case(
    func,
    bounds,
    x0,
    algorithm: str,
    maxevals: int,
    seed: int,
    quality_upper: float | None = None,
):
    result = minionpy.Minimizer(
        func=func,
        bounds=bounds,
        x0=x0,
        algo=algorithm,
        maxevals=maxevals,
        seed=seed,
        options={"convergence_tol": 1e-8},
    ).optimize()

    finite_ok = math.isfinite(result.fun)
    eval_ok = result.nfev <= maxevals + EVAL_SLACK
    quality_ok = True if quality_upper is None else result.fun <= quality_upper
    passed = finite_ok and eval_ok and quality_ok

    return {
        "algorithm": algorithm,
        "fun": float(result.fun),
        "nfev": int(result.nfev),
        "passed": passed,
        "message": (
            f"finite={finite_ok} nfev={result.nfev} "
            f"limit={maxevals + EVAL_SLACK} best_f={result.fun}"
            + ("" if quality_upper is None else f" quality_limit={quality_upper}")
        ),
    }


def run_sphere_suite(
    dimension: int = 5,
    maxevals: int = 4000,
    seed: int = 42,
):
    bounds = [(-5.0, 5.0)] * dimension
    x0 = [[0.5] * dimension]
    records = []

    for algorithm in ALGORITHMS:
        record = _run_case(
            func=sphere_batch,
            bounds=bounds,
            x0=x0,
            algorithm=algorithm,
            maxevals=maxevals,
            seed=seed,
            quality_upper=SPHERE_UPPER[algorithm],
        )
        records.append(record)

    return records


def run_rosenbrock_suite(
    dimension: int = 5,
    maxevals: int = 4000,
    seed: int = 42,
):
    bounds = [(-5.0, 5.0)] * dimension
    x0 = [[0.5] * dimension]
    records = []

    for algorithm in ALGORITHMS:
        record = _run_case(
            func=rosenbrock_batch,
            bounds=bounds,
            x0=x0,
            algorithm=algorithm,
            maxevals=maxevals,
            seed=seed,
            quality_upper=ROSENBROCK_UPPER[algorithm],
        )
        records.append(record)

    return records


def run_cec2017_suite(
    function_numbers,
    algorithms=ALGORITHMS,
    dimension: int = 10,
    maxevals: int = 10000,
    seed: int = 20250306,
):
    bounds = [(-100.0, 100.0)] * dimension
    x0 = [[0.0] * dimension]
    records = []

    for function_number in function_numbers:
        objective = make_cec2017_batch(function_number, dimension)
        for algorithm in algorithms:
            record = _run_case(
                func=objective,
                bounds=bounds,
                x0=x0,
                algorithm=algorithm,
                maxevals=maxevals,
                seed=seed,
            )
            record["function_number"] = function_number
            records.append(record)

    return records


def _print_function_results(function_name: str, records, dimension: int, maxevals: int):
    print(f"{function_name} minimization using all Minion algorithms")
    print(f"dimension={dimension}, maxevals={maxevals}\n")
    print(f"{'Algorithm':<18}{'best_f':>16}{'nfev':>12}")
    print("-" * 46)
    for record in records:
        print(f"{record['algorithm']:<18}{record['fun']:>16.8e}{record['nfev']:>12}")


def _print_cec2017_results(records, function_numbers, dimension: int, maxevals: int, seed: int):
    print(f"\nCEC2017 minimization (F{min(function_numbers)}-F{max(function_numbers)}, dimension={dimension})")
    print(f"functions=F{min(function_numbers)}-F{max(function_numbers)}, maxevals={maxevals}, seed={seed}\n")
    current_function = None
    for index, record in enumerate(records):
        function_number = record["function_number"]
        if function_number != current_function:
            current_function = function_number
            print(f"Function F{function_number}")
            print(f"{'Algorithm':<18}{'best_f':>16}{'nfev':>12}")
            print("-" * 46)
        print(f"{record['algorithm']:<18}{record['fun']:>16.8e}{record['nfev']:>12}")
        next_is_new_function = index + 1 < len(records) and records[index + 1]["function_number"] != function_number
        if next_is_new_function:
            print()
    if records and records[-1]["function_number"] == current_function:
        print()


def _check_vectorized_test_functions():
    x_batch = np.zeros((3, 5), dtype=float)

    sphere_values = np.asarray(minionpy.sphere(x_batch), dtype=float)
    rosenbrock_values = np.asarray(minionpy.rosenbrock(x_batch), dtype=float)

    if sphere_values.shape != (3,) or rosenbrock_values.shape != (3,):
        return {
            "algorithm": "test_functions",
            "passed": False,
            "message": f"unexpected output shapes sphere={sphere_values.shape} rosenbrock={rosenbrock_values.shape}",
        }
    if not np.all(np.isfinite(sphere_values)) or not np.all(np.isfinite(rosenbrock_values)):
        return {
            "algorithm": "test_functions",
            "passed": False,
            "message": "non-finite values from vectorized test functions",
        }
    return {"algorithm": "test_functions", "passed": True, "message": "ok"}


def _check_cec2017_wrapper():
    cec2017 = minionpy.CEC2017Functions(1, 10)
    x_batch = np.zeros((2, 10), dtype=float).tolist()

    values = np.asarray(cec2017(x_batch), dtype=float)

    if values.shape != (2,):
        return {"algorithm": "CEC2017_wrapper", "passed": False, "message": f"unexpected output shape {values.shape}"}
    if not np.all(np.isfinite(values)):
        return {"algorithm": "CEC2017_wrapper", "passed": False, "message": "non-finite values from CEC2017 wrapper"}
    return {"algorithm": "CEC2017_wrapper", "passed": True, "message": "ok"}


def main():
    test_dimension = 5
    test_maxevals = 4000
    test_seed = 42

    cec_dimension = 10
    cec_maxevals = 10000
    cec_seed = 20250306
    cec_function_numbers = list(range(1, 31))

    failed_checks = 0
    total_checks = 0

    x_batch_check = _check_vectorized_test_functions()
    total_checks += 1
    if not x_batch_check["passed"]:
        failed_checks += 1
        print(f"[FAIL][Vectorized] {x_batch_check['algorithm']} {x_batch_check['message']}", file=sys.stderr)

    cec_wrapper_check = _check_cec2017_wrapper()
    total_checks += 1
    if not cec_wrapper_check["passed"]:
        failed_checks += 1
        print(f"[FAIL][Wrapper] {cec_wrapper_check['algorithm']} {cec_wrapper_check['message']}", file=sys.stderr)

    sphere_records = run_sphere_suite(
        dimension=test_dimension,
        maxevals=test_maxevals,
        seed=test_seed,
    )
    _print_function_results("Sphere", sphere_records, test_dimension, test_maxevals)
    total_checks += len(sphere_records)
    for record in sphere_records:
        if not record["passed"]:
            failed_checks += 1
            print(f"[FAIL][Sphere] {record['algorithm']} {record['message']}", file=sys.stderr)

    print()
    rosenbrock_records = run_rosenbrock_suite(
        dimension=test_dimension,
        maxevals=test_maxevals,
        seed=test_seed,
    )
    _print_function_results("Rosenbrock", rosenbrock_records, test_dimension, test_maxevals)
    total_checks += len(rosenbrock_records)
    for record in rosenbrock_records:
        if not record["passed"]:
            failed_checks += 1
            print(f"[FAIL][Rosenbrock] {record['algorithm']} {record['message']}", file=sys.stderr)

    cec_records = run_cec2017_suite(
        function_numbers=cec_function_numbers,
        algorithms=ALGORITHMS,
        dimension=cec_dimension,
        maxevals=cec_maxevals,
        seed=cec_seed,
    )
    _print_cec2017_results(cec_records, cec_function_numbers, cec_dimension, cec_maxevals, cec_seed)
    total_checks += len(cec_records)
    for record in cec_records:
        if not record["passed"]:
            failed_checks += 1
            print(f"[FAIL][CEC2017 F{record['function_number']}] {record['algorithm']} {record['message']}", file=sys.stderr)

    passed_checks = total_checks - failed_checks
    print(f"\nTest summary: {passed_checks}/{total_checks} checks passed.")
    return 0 if failed_checks == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
