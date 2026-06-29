from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def find_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    while True:
        if (current / "minion").exists() and (current / "minionpy").exists():
            return current
        if current.parent == current:
            raise RuntimeError("Could not locate the repository root")
        current = current.parent


REPO_ROOT = find_repo_root()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "minionpy"))

import cocoex
import minionpy as mpy
import libcmaes_bridge


def recommended_lambda(dimension: int) -> int:
    return max(5, 4 + int(math.floor(3.0 * math.log(dimension))))


class ProgressTracker:
    def __init__(self, fopt: float, progress_step: int):
        self.fopt = float(fopt)
        self.progress_step = max(1, int(progress_step))
        self.evals = 0
        self.best = math.inf
        self.next_progress = self.progress_step
        self.eval_history: list[int] = []
        self.err_history: list[float] = []

    def _record_progress(self) -> None:
        while self.evals >= self.next_progress:
            self.eval_history.append(self.next_progress)
            self.err_history.append(max(abs(self.best - self.fopt), 1e-300))
            self.next_progress += self.progress_step

    def update(self, values) -> None:
        for value in np.asarray(values, dtype=float).ravel():
            self.evals += 1
            value = float(value)
            if value < self.best:
                self.best = value
            self._record_progress()

    def batch_objective(self, scalar_objective):
        def wrapped(x_batch):
            x_batch = np.asarray(x_batch, dtype=float)
            if x_batch.ndim == 1:
                x_batch = x_batch[None, :]
            values = [float(scalar_objective(x)) for x in x_batch]
            self.update(values)
            return values

        return wrapped

    def scalar_objective(self, scalar_objective):
        def wrapped(x):
            value = float(scalar_objective(np.asarray(x, dtype=float)))
            self.update([value])
            return value

        return wrapped

    @property
    def final_error(self) -> float:
        return max(abs(self.best - self.fopt), 1e-300)


MINION_ALGOS = [
    ("Minion CMAES", "CMAES", 0.3),
    ("Minion BIPOP_aCMAES", "BIPOP_aCMAES", 0.3),
    ("Minion RCMAES", "RCMAES", 0.2),
]

LIBCMAES_ALGOS = [
    ("libcmaes cmaes", "cmaes", 0.3),
    ("libcmaes acmaes", "acmaes", 0.3),
    ("libcmaes bipop", "bipop", 0.3),
    ("libcmaes abipop", "abipop", 0.3),
]


def make_cec2017_problem(function_number: int, dimension: int):
    cec = mpy.CEC2017Functions(function_number, dimension)
    fopt = 100 * function_number
    bounds = [(-100.0, 100.0)] * dimension
    x0 = [0.0] * dimension

    def scalar(x):
        return float(cec([np.asarray(x, dtype=float).tolist()])[0])

    return scalar, fopt, bounds, x0


def make_bbob_problem(function_number: int, dimension: int):
    problem = cocoex.BareProblem(
        suite_name="bbob",
        function=function_number,
        dimension=dimension,
        instance=1,
    )
    fopt = float(problem.best_value())
    bounds = [(-5.0, 5.0)] * dimension
    x0 = [0.0] * dimension

    def scalar(x):
        return float(problem(np.asarray(x, dtype=float)))

    return scalar, fopt, bounds, x0, problem.id


def minion_options(algo_name: str, dimension: int, lambda_: int, sigma0: float):
    common = {
        "population_size": 0,
        "bound_strategy": "reflect-random",
        "convergence_tol": 0.0,
    }
    if algo_name == "CMAES":
        common.update(
            {
                "mu": max(1, lambda_ // 2),
                "cc": 0.0,
                "cs": 0.0,
                "c1": 0.0,
                "cmu": 0.0,
                "damps": 0.0,
            }
        )
    elif algo_name == "BIPOP_aCMAES":
        common.update({"max_iterations": 5000})
    elif algo_name == "RCMAES":
        pass
    else:
        raise ValueError(f"Unsupported Minion algorithm: {algo_name}")
    return common


def run_minion(algo_label: str, algo_name: str, scalar_objective, bounds, x0, budget: int, seed: int, sigma0: float, progress_step: int, fopt: float):
    tracker = ProgressTracker(fopt, progress_step)
    objective = tracker.batch_objective(scalar_objective)
    dim = len(bounds)
    lambda_ = recommended_lambda(dim)
    options = minion_options(algo_name, dim, lambda_, sigma0)
    result = mpy.Minimizer(
        func=objective,
        bounds=bounds,
        x0=[x0],
        algo=algo_name,
        maxevals=budget,
        seed=seed,
        options=options,
    ).optimize()
    return {
        "family": "minion",
        "label": algo_label,
        "algorithm": algo_name,
        "best_f": float(result.fun),
        "nfev": int(result.nfev),
        "error": tracker.final_error,
        "tracker": tracker,
        "result": result,
        "status": "ok",
    }


def run_libcmaes(algo_label: str, algo_name: str, scalar_objective, bounds, x0, budget: int, seed: int, sigma0: float, progress_step: int, fopt: float):
    tracker = ProgressTracker(fopt, progress_step)
    dim = len(bounds)
    lambda_ = recommended_lambda(dim)
    objective = tracker.scalar_objective(scalar_objective)
    try:
        result = libcmaes_bridge.optimize(
            objective=objective,
            x0=[float(v) for v in np.asarray(x0, dtype=float).ravel().tolist()],
            sigma0=float(sigma0),
            bounds=[(float(lo), float(hi)) for lo, hi in bounds],
            algo=str(algo_name),
            **{"lambda": int(lambda_)},
            seed=int(seed),
            maxevals=int(budget),
        )
        return {
            "family": "libcmaes",
            "label": algo_label,
            "algorithm": algo_name,
            "best_f": float(result["best_f"]),
            "nfev": int(result["nevals"]),
            "error": tracker.final_error,
            "tracker": tracker,
            "result": result,
            "status": "ok",
        }
    except Exception as exc:
        return {
            "family": "libcmaes",
            "label": algo_label,
            "algorithm": algo_name,
            "best_f": math.inf,
            "nfev": 0,
            "error": tracker.final_error,
            "tracker": tracker,
            "result": None,
            "status": f"failed: {exc}",
        }


def summarize_records(records):
    print(f"{'Label':<24}{'best_f':>16}{'nfev':>12}{'error':>16}  status")
    print("-" * 92)
    for record in records:
        print(
            f"{record['label']:<24}{record['best_f']:>16.8e}{record['nfev']:>12}{record['error']:>16.8e}  {record.get('status', 'ok')}"
        )


def plot_records(records, title: str, dimension: int):
    plt.figure(figsize=(10, 6))
    for record in records:
        tracker = record["tracker"]
        if not tracker.eval_history:
            continue
        xvals = np.asarray(tracker.eval_history, dtype=float) / float(dimension)
        yvals = np.asarray(tracker.err_history, dtype=float)
        plt.semilogy(xvals, yvals, label=record["label"])
    plt.xlabel(r"$\#\mathrm{evals} / D$")
    plt.ylabel(r"$|f - f_{opt}|$")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()


def run_cec2017():
    print("\n# CEC 2017 comparison.")
    cec_dimension = 10
    cec_budget = 10_000
    cec_seed = 20250306
    cec_progress_step = 5_000
    run_full_cec = False
    cec_functions = list(range(1, 31)) if run_full_cec else [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 25, 29]

    cec_records = []
    for function_number in cec_functions:
        scalar, fopt, bounds, x0 = make_cec2017_problem(function_number, cec_dimension)
        for label, algo_name, sigma0 in MINION_ALGOS:
            cec_records.append(
                {
                    **run_minion(label, algo_name, scalar, bounds, x0, cec_budget, cec_seed, sigma0, cec_progress_step, fopt),
                    "function_number": function_number,
                    "problem": f"CEC2017 F{function_number}",
                }
            )
        for label, algo_name, sigma0 in LIBCMAES_ALGOS:
            cec_records.append(
                {
                    **run_libcmaes(label, algo_name, scalar, bounds, x0, cec_budget, cec_seed, sigma0, cec_progress_step, fopt),
                    "function_number": function_number,
                    "problem": f"CEC2017 F{function_number}",
                }
            )

    print(f"CEC2017 functions: {cec_functions}")
    for function_number in cec_functions:
        print(f"CEC2017 F{function_number}")
        summarize_records([r for r in cec_records if r["function_number"] == function_number])

    selected_cec = 18 if 18 in cec_functions else cec_functions[0]
    plot_records(
        [r for r in cec_records if r["function_number"] == selected_cec],
        f"CEC2017 F{selected_cec} comparison",
        cec_dimension,
    )


def run_bbob():
    print("\n# BBOB comparison.")
    bbob_function = 7
    bbob_dimension = 10
    bbob_budget = 100_000
    bbob_seed = 20250306
    bbob_progress_step = 100_000

    scalar, fopt, bounds, x0, bbob_problem_id = make_bbob_problem(bbob_function, bbob_dimension)
    bbob_records = []
    for label, algo_name, sigma0 in MINION_ALGOS:
        bbob_records.append(
            run_minion(label, algo_name, scalar, bounds, x0, bbob_budget, bbob_seed, sigma0, bbob_progress_step, fopt)
        )
    for label, algo_name, sigma0 in LIBCMAES_ALGOS:
        bbob_records.append(
            run_libcmaes(label, algo_name, scalar, bounds, x0, bbob_budget, bbob_seed, sigma0, bbob_progress_step, fopt)
        )

    print("BBOB problem:", bbob_problem_id)
    summarize_records(bbob_records)
    plot_records(bbob_records, f"BBOB F{bbob_function} comparison", bbob_dimension)


def main():
    print("Repository root:", REPO_ROOT)
    print("Minion Python:", mpy.__file__)
    print("libcmaes bridge:", getattr(libcmaes_bridge, "__file__", "<built-in>"))

    run_cec2017()
    run_bbob()
    plt.show()


if __name__ == "__main__":
    main()
