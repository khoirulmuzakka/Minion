from __future__ import annotations

import math
import os
import sys
from typing import Callable

import numpy as np


IMPORT_MODE = os.environ.get("MINIONPY_IMPORT_MODE", "local")
if IMPORT_MODE not in {"local", "installed"}:
    raise RuntimeError("MINIONPY_IMPORT_MODE must be either 'local' or 'installed'.")

if IMPORT_MODE == "local":
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
    "NLSHADE_LBC",
    "LSRTDE",
    "RDEX",
    "jSO",
    "IMODE",
    "ARRDE",
    "NelderMead",
    "ABC",
    "PSO",
    "SPSO2011",
    "DMSPSO",
    "LSHADE_cnEpSin",
    "CMAES",
    "ACMAES",
    "RCMAES",
    "BIPOP_aCMAES",
    "DA",
    "L_BFGS_B",
    "L_BFGS",
)

MAXEVALS = 180
EVAL_SLACK = 40
TOLERANCE_STOP_ALGORITHMS = (
    "DE",
    "LSHADE",
    "AGSK",
    "JADE",
    "NLSHADE_RSP",
    "NLSHADE_LBC",
    "LSRTDE",
    "RDEX",
    "jSO",
    "IMODE",
    "NelderMead",
    "PSO",
    "SPSO2011",
    "DMSPSO",
    "LSHADE_cnEpSin",
    "CMAES",
    "ACMAES",
    "DA",
)
RESTART_TOLERANCE_ALGORITHMS = ("RCMAES", "BIPOP_aCMAES")
RESTART_STRATEGY_ALGORITHMS = ("ARRDE", "j2020", "RCMAES", "BIPOP_aCMAES")
MAXITERS_ALGORITHMS = tuple(algo for algo in ALGORITHMS if algo not in RESTART_STRATEGY_ALGORITHMS)


def sphere_batch(x_batch):
    return [float(np.sum(np.asarray(x, dtype=float) ** 2)) for x in x_batch]


def zero_batch(x_batch):
    return [0.0 for _ in x_batch]


def make_minimizer(algo: str, callback=None, maxevals: int = MAXEVALS, options=None):
    merged_options = {"x_tol": 1e-8, "f_tol": 1e-12}
    if options:
        merged_options.update(options)
    return minionpy.Minimizer(
        func=sphere_batch,
        bounds=[(-1.0, 1.0), (-1.0, 1.0)],
        x0=[[0.3, -0.2]],
        algo=algo,
        maxevals=maxevals,
        callback=callback,
        seed=3,
        options=merged_options,
    )


def make_zero_minimizer(algo: str, maxevals: int = MAXEVALS, options=None):
    merged_options = {"x_tol": 0.0, "f_tol": 1e-12}
    if options:
        merged_options.update(options)
    return minionpy.Minimizer(
        func=zero_batch,
        bounds=[(-1.0, 1.0), (-1.0, 1.0)],
        x0=[[0.3, -0.2]],
        algo=algo,
        maxevals=maxevals,
        seed=7,
        options=merged_options,
    )


def run_checks(checks: list[tuple[str, Callable[[], None]]]) -> int:
    total = len(checks)
    for index, (name, check) in enumerate(checks, 1):
        check()
        print(f"Test {index}/{total}: {name}: passed")
    print(f"{total}/{total} Passed.")
    return 0


def check_status_string_representation():
    assert str(minionpy.TerminationStatus.Converged) == "converged"
    assert repr(minionpy.TerminationStatus.MaxEvaluationsReached) == "max_evaluations_reached"


def check_result_object_semantics():
    converged = make_minimizer("NelderMead").optimize()
    exhausted = make_minimizer("ARRDE", options={"minimum_population_size": 4}).optimize()

    assert converged.succeeded()
    assert not exhausted.succeeded()
    assert converged.status == minionpy.TerminationStatus.Converged
    assert "status=converged" in repr(converged)


def check_vectorized_test_functions():
    x_batch = np.zeros((3, 5), dtype=float)
    sphere_values = np.asarray(minionpy.sphere(x_batch), dtype=float)
    rosenbrock_values = np.asarray(minionpy.rosenbrock(x_batch), dtype=float)

    assert sphere_values.shape == (3,)
    assert rosenbrock_values.shape == (3,)
    assert np.all(np.isfinite(sphere_values))
    assert np.all(np.isfinite(rosenbrock_values))


def check_cec2017_wrapper():
    cec2017 = minionpy.CEC2017Functions(1, 10)
    values = np.asarray(cec2017(np.zeros((2, 10), dtype=float).tolist()), dtype=float)

    assert values.shape == (2,)
    assert np.all(np.isfinite(values))


def check_callback_false_or_none_continues():
    for callback in (lambda result: False, lambda result: None):
        calls = []

        def wrapped(result):
            calls.append(result)
            return callback(result)

        result = make_minimizer("ARRDE", callback=wrapped, options={"minimum_population_size": 4}).optimize()
        assert calls
        assert result.status != minionpy.TerminationStatus.CallbackStopped
        assert result.status == minionpy.TerminationStatus.MaxEvaluationsReached


def check_callback_receives_minion_result():
    seen = []

    def callback(result):
        seen.append(result)
        return True

    result = make_minimizer("DE", callback=callback).optimize()

    assert seen
    assert isinstance(seen[0], minionpy.MinionResult)
    assert math.isfinite(seen[0].fun)
    assert seen[0].nfev > 0
    assert result.status == minionpy.TerminationStatus.CallbackStopped
    assert result.message == "Callback requested optimization stop."


def check_all_algorithms_have_terminal_status():
    bad = []
    for algo in ALGORITHMS:
        options = {"minimum_population_size": 4} if algo == "ARRDE" else None
        result = make_minimizer(algo, options=options).optimize()
        if result.status in {
            minionpy.TerminationStatus.Running,
            minionpy.TerminationStatus.RuntimeError,
        }:
            bad.append((algo, str(result.status), result.message))
        if not math.isfinite(result.fun):
            bad.append((algo, "non_finite", result.fun))
        if result.nfev > MAXEVALS + EVAL_SLACK:
            bad.append((algo, "nfev", result.nfev))

    assert not bad, bad


def check_all_algorithms_callback_stop():
    bad = []

    def stop_now(result):
        return True

    for algo in ALGORITHMS:
        options = {"minimum_population_size": 4} if algo == "ARRDE" else None
        result = make_minimizer(algo, callback=stop_now, options=options).optimize()
        if result.status != minionpy.TerminationStatus.CallbackStopped:
            bad.append((algo, str(result.status), result.message))

    assert not bad, bad


def check_budget_exhaustion_algorithms_do_not_fake_convergence():
    for algo in ("ARRDE", "j2020", "RCMAES"):
        options = {"minimum_population_size": 4} if algo == "ARRDE" else None
        result = make_minimizer(algo, options=options).optimize()
        assert result.status == minionpy.TerminationStatus.MaxEvaluationsReached, (
            algo,
            result.status,
            result.message,
        )


def check_x_tol_converges_population_algorithm():
    result = make_minimizer(
        "DE",
        maxevals=80,
        options={"x_tol": 10.0, "f_tol": 0.0, "population_size": 8},
    ).optimize()

    assert result.status == minionpy.TerminationStatus.Converged


def check_f_tol_handles_zero_objective_value_spread():
    result = make_zero_minimizer(
        "PSO",
        maxevals=80,
        options={"x_tol": 0.0, "f_tol": 1e-12, "population_size": 10},
    ).optimize()

    assert result.status == minionpy.TerminationStatus.Converged
    assert result.fun == 0.0


def check_default_f_tol_is_disabled():
    result = minionpy.Minimizer(
        func=zero_batch,
        bounds=[(-1.0, 1.0), (-1.0, 1.0)],
        x0=[[0.3, -0.2]],
        algo="DE",
        maxevals=80,
        seed=7,
        options={"x_tol": -1.0, "population_size": 8},
    ).optimize()

    assert result.status == minionpy.TerminationStatus.MaxEvaluationsReached


def check_negative_tolerances_disable_convergence():
    result = make_zero_minimizer(
        "DE",
        maxevals=80,
        options={"x_tol": -1.0, "f_tol": -1.0, "population_size": 8},
    ).optimize()

    assert result.status == minionpy.TerminationStatus.MaxEvaluationsReached


def check_all_algorithms_stop_at_maxiters():
    for algo in MAXITERS_ALGORITHMS:
        options = {
            "maxiters": 1,
            "x_tol": -1.0,
            "f_tol": -1.0,
            "population_size": 12,
            "g_epsilon": 0.0,
            "g_epsilon_rel": 0.0,
            "f_reltol": 0.0,
        }
        if algo == "DA":
            options["use_local_search"] = False
        result = make_minimizer(
            algo,
            maxevals=500,
            options=options,
        ).optimize()
        assert result.status == minionpy.TerminationStatus.MaxIterationsReached, (
            algo,
            result.status,
            result.message,
        )


def check_all_algorithms_stop_at_zero_maxiters():
    for algo in MAXITERS_ALGORITHMS:
        options = {
            "maxiters": 0,
            "x_tol": -1.0,
            "f_tol": -1.0,
            "population_size": 12,
            "g_epsilon": 0.0,
            "g_epsilon_rel": 0.0,
            "f_reltol": 0.0,
        }
        if algo == "DA":
            options["use_local_search"] = False
        result = make_minimizer(
            algo,
            maxevals=500,
            options=options,
        ).optimize()
        assert result.status == minionpy.TerminationStatus.MaxIterationsReached, (
            algo,
            result.status,
            result.message,
        )
        assert result.nit == 0, (algo, result.nit)


def check_restart_algorithms_do_not_use_maxiters():
    for algo in RESTART_STRATEGY_ALGORITHMS:
        options = {
            "maxiters": 1,
            "x_tol": -1.0,
            "f_tol": -1.0,
            "population_size": 12,
        }
        if algo == "ARRDE":
            options["minimum_population_size"] = 4
        result = make_minimizer(
            algo,
            maxevals=120,
            options=options,
        ).optimize()
        assert result.status != minionpy.TerminationStatus.MaxIterationsReached, (
            algo,
            result.status,
            result.message,
        )


def check_legacy_max_iterations_is_ignored():
    for algo in ("L_BFGS", "L_BFGS_B"):
        result = make_minimizer(
            algo,
            maxevals=80,
            options={
                "maxiters": -1,
                "max_iterations": 1,
                "g_epsilon": 0.0,
                "g_epsilon_rel": 0.0,
                "f_reltol": 0.0,
            },
        ).optimize()
        assert result.status != minionpy.TerminationStatus.MaxIterationsReached, (
            algo,
            result.status,
            result.message,
        )
        assert result.nit > 1, (algo, result.nit)


def check_all_tolerance_stop_algorithms_converge():
    for algo in TOLERANCE_STOP_ALGORITHMS:
        options = {"x_tol": 0.0, "f_tol": 1e-12, "population_size": 12}
        if algo == "DA":
            options["use_local_search"] = False
        result = make_zero_minimizer(
            algo,
            maxevals=120,
            options=options,
        ).optimize()
        assert result.status == minionpy.TerminationStatus.Converged, (
            algo,
            result.status,
            result.message,
        )


def check_j2020_keeps_reset_strategy_without_tolerance_stop():
    result = make_zero_minimizer(
        "j2020",
        maxevals=70,
        options={"x_tol": 10.0, "f_tol": 1e-12, "population_size": 32},
    ).optimize()

    assert result.status == minionpy.TerminationStatus.MaxEvaluationsReached


def check_cma_restart_cap_option():
    for algo in RESTART_TOLERANCE_ALGORITHMS:
        result = make_zero_minimizer(
            algo,
            maxevals=120,
            options={"x_tol": 0.0, "f_tol": 1e-12, "max_restarts": 0, "population_size": 8},
        ).optimize()
        assert result.status == minionpy.TerminationStatus.Stagnated, (
            algo,
            result.status,
            result.message,
        )
        assert result.message == "Maximum number of restarts reached."


def check_dual_annealing_callback_without_local_search():
    result = make_minimizer(
        "DA",
        callback=lambda result: True,
        options={"use_local_search": False},
    ).optimize()

    assert result.status == minionpy.TerminationStatus.CallbackStopped


def check_lbfgs_callbacks_stop():
    for algo in ("L_BFGS", "L_BFGS_B"):
        result = make_minimizer(algo, callback=lambda result: True).optimize()
        assert result.status == minionpy.TerminationStatus.CallbackStopped


def check_rdex_direct_wrapper():
    direct = minionpy.RDEX(
        func=sphere_batch,
        bounds=[(-1.0, 1.0), (-1.0, 1.0)],
        x0=[[0.3, -0.2]],
        maxevals=MAXEVALS,
        seed=3,
    ).optimize()
    generic = make_minimizer("RDEX").optimize()

    assert math.isfinite(direct.fun)
    assert math.isfinite(generic.fun)
    assert direct.status != minionpy.TerminationStatus.Running
    assert generic.status != minionpy.TerminationStatus.Running


def main() -> int:
    checks = [
        ("readable termination status strings", check_status_string_representation),
        ("MinionResult semantics", check_result_object_semantics),
        ("vectorized test functions", check_vectorized_test_functions),
        ("CEC2017 wrapper returns finite vector output", check_cec2017_wrapper),
        ("callbacks returning False or None continue", check_callback_false_or_none_continues),
        ("callback receives MinionResult and can stop", check_callback_receives_minion_result),
        ("all algorithms return terminal finite results", check_all_algorithms_have_terminal_status),
        ("all algorithms stop when callback returns True", check_all_algorithms_callback_stop),
        ("budget-exhaustion algorithms do not fake convergence", check_budget_exhaustion_algorithms_do_not_fake_convergence),
        ("x_tol converges population algorithm", check_x_tol_converges_population_algorithm),
        ("f_tol handles zero objective-value spread", check_f_tol_handles_zero_objective_value_spread),
        ("default f_tol is disabled", check_default_f_tol_is_disabled),
        ("negative tolerances disable convergence", check_negative_tolerances_disable_convergence),
        ("supported algorithms stop at maxiters", check_all_algorithms_stop_at_maxiters),
        ("supported algorithms stop at zero maxiters", check_all_algorithms_stop_at_zero_maxiters),
        ("restart algorithms do not use maxiters", check_restart_algorithms_do_not_use_maxiters),
        ("legacy max_iterations is ignored", check_legacy_max_iterations_is_ignored),
        ("all tolerance-stop algorithms converge", check_all_tolerance_stop_algorithms_converge),
        ("j2020 keeps reset strategy without tolerance stop", check_j2020_keeps_reset_strategy_without_tolerance_stop),
        ("CMA restart cap option works", check_cma_restart_cap_option),
        ("Dual Annealing callback works without local search", check_dual_annealing_callback_without_local_search),
        ("L-BFGS callbacks stop", check_lbfgs_callbacks_stop),
        ("RDEX direct and generic wrappers run", check_rdex_direct_wrapper),
    ]
    return run_checks(checks)


if __name__ == "__main__":
    raise SystemExit(main())
