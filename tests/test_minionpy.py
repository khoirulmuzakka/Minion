from __future__ import annotations

import math
import os
import sys
import unittest
from statistics import NormalDist
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
    "GradientDescent",
    "GD",
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
    "GradientDescent",
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

    assert converged.status == minionpy.TerminationStatus.Converged
    assert exhausted.status != minionpy.TerminationStatus.Converged
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


def check_lbfgs_line_search_fallback_stagnates():
    cases = {
        "L_BFGS": {"max_linesearch": 1, "c_1": 0.9, "c_2": 0.1},
        "L_BFGS_B": {"max_linesearch": 0},
    }
    for algo, options in cases.items():
        result = make_minimizer(
            algo,
            maxevals=80,
            options={
                **options,
                "g_epsilon": 0.0,
                "g_epsilon_rel": 0.0,
                "f_reltol": 0.0,
            },
        ).optimize()
        assert result.status == minionpy.TerminationStatus.Stagnated, (
            algo,
            result.status,
            result.message,
        )
        assert "line search" in result.message, (algo, result.message)


def check_gradient_descent_backtracking_improves_large_step():
    common_options = {
        "update_rule": "gd",
        "gradient_estimator": "coordinate_fd",
        "base_learning_rate": 10.0,
        "maxiters": 1,
        "g_tol": -1.0,
        "x_tol": -1.0,
        "f_tol": -1.0,
    }
    without_ls = make_minimizer(
        "GradientDescent",
        maxevals=120,
        options=common_options,
    ).optimize()
    with_ls = make_minimizer(
        "GradientDescent",
        maxevals=120,
        options={**common_options, "use_line_search": True, "max_linesearch": 8},
    ).optimize()

    assert math.isfinite(with_ls.fun)
    assert with_ls.fun < without_ls.fun, (without_ls.fun, with_ls.fun)


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


_UNCERTAINTY_ASSERTIONS = unittest.TestCase()


def check_hessian_polynomial_fit_with_known_observation_errors():
    rng = np.random.default_rng(182)
    t = np.linspace(-0.8, 1.7, 90)
    design = np.column_stack([np.ones(t.size), t, t**2])
    errors = 0.12 + 0.07 * (t + 0.8)
    observed = design @ np.array([0.7, -1.2, 0.4]) + rng.normal(0, errors)
    weighted = design / errors[:, None]
    expected_x = np.linalg.lstsq(weighted, observed / errors, rcond=None)[0]
    expected_cov = np.linalg.inv(weighted.T @ weighted)

    def chi_squared(points):
        residuals = (np.asarray(points) @ design.T - observed) / errors
        return np.sum(residuals**2, axis=1)

    fit = minionpy.Minimizer(
        chi_squared, [(-3, 3)] * 3, x0=[[0, 0, 0]],
        algo="NelderMead", maxevals=5000, seed=17,
        options={"x_tol": 1e-9, "f_tol": 1e-12}).optimize()
    np.testing.assert_allclose(fit.x, expected_x, atol=2e-5)
    result = minionpy.Uncertainty(chi_squared, fit.x, [(-3, 3)] * 3).hessian(scale=2)
    assert result.covariance_valid
    assert result.covariance_available
    assert result.uncertainty_complete
    assert result.retained_rank == 3
    assert result.discarded_modes == []
    np.testing.assert_allclose(result.hessian, 2 * weighted.T @ weighted, rtol=2e-7)
    np.testing.assert_allclose(result.covariance, expected_cov, rtol=2e-7)
    expected_std = np.sqrt(np.diag(expected_cov))
    np.testing.assert_allclose(result.std, expected_std, rtol=2e-7)
    np.testing.assert_allclose(result.correlation,
                               expected_cov / np.outer(expected_std, expected_std), rtol=2e-7)
    z = NormalDist().inv_cdf(0.975)
    np.testing.assert_allclose(result.ci_lower, np.asarray(fit.x) - z * expected_std, rtol=2e-7)
    np.testing.assert_allclose(result.ci_upper, np.asarray(fit.x) + z * expected_std, rtol=2e-7)
    assert result.differentiation_error < 1e-6
    half = minionpy.Uncertainty(lambda p: chi_squared(p) / 2, fit.x).hessian(scale=1)
    np.testing.assert_allclose(half.covariance, result.covariance, rtol=2e-7)


def check_hessian_bounds_and_one_sided_mixed_derivatives():
    seen = []
    matrix = np.array([[4., 1.], [1., 3.]])

    def objective(points):
        p = np.asarray(points)
        assert np.all((p >= [0, -1]) & (p <= [1, 0]))
        seen.extend(points)
        return np.einsum("bi,ij,bj->b", p, matrix, p) / 2

    result = minionpy.Uncertainty(objective, [0, 0], [(0, 1), (-1, 0)]).hessian(1)
    assert result.stencil_types == ["forward", "backward"]
    assert result.warnings
    np.testing.assert_allclose(result.hessian, matrix, atol=1e-10)
    np.testing.assert_allclose(result.covariance, np.linalg.inv(matrix), atol=1e-10)
    assert result.nfev == len(seen)


def check_hessian_unidentified_or_negative_curvature_has_no_covariance():
    for objective,status in (
        [
            (lambda p: np.asarray(p)[:, 0]**2, "singular"),
            (lambda p: np.asarray(p)[:, 0]**2 - np.asarray(p)[:, 1]**2, "indefinite"),
            (lambda p: np.zeros(len(p)), "singular"),
        ]
    ):
        r = minionpy.Uncertainty(objective, [0, 0]).hessian(1, discard_nonpositive=False)
        assert r.status == status
        assert not r.covariance_valid
        assert not r.covariance_available
        assert r.covariance.size == 0
        assert not r.std
        assert r.eigenvalues.shape == (2,)


def check_hessian_partial_uncertainty_discards_unresolved_modes():
    for curvature in (
        [0., -2., 1e-12]
    ):
        # Rotate modes so the retained contribution reaches both parameters.
        rotation = np.array([[.8, -.6], [.6, .8]])
        matrix = rotation @ np.diag([4., curvature]) @ rotation.T

        def objective(points):
            p = np.asarray(points)
            return np.einsum("bi,ij,bj->b", p, matrix, p) / 2

        r = minionpy.Uncertainty(objective, [0, 0]).hessian(2)
        expected = .5 * np.outer(rotation[:, 0], rotation[:, 0])
        assert r.status == "partial"
        assert r.covariance_available
        assert not r.covariance_valid
        assert not r.uncertainty_complete
        assert r.retained_rank == 1
        assert r.discarded_modes == [0]
        np.testing.assert_allclose(r.covariance, expected, atol=1e-12)
        np.testing.assert_allclose(r.std, np.sqrt(np.diag(expected)), atol=1e-12)
        np.testing.assert_allclose(r.correlation, np.ones((2, 2)), atol=1e-12)
        assert any("incomplete" in warning for warning in r.warnings)
        if curvature < 0:
            assert any("too-small steps" in warning for warning in r.warnings)


def check_hessian_discarded_parameter_zero_is_not_reported_as_a_valid_uncertainty():
    r = minionpy.Uncertainty(lambda p: np.asarray(p)[:, 0]**2, [0, 0]).hessian(1)
    assert r.status == "partial"
    np.testing.assert_allclose(r.std, [np.sqrt(.5), 0])
    assert np.isnan(r.correlation[1]).all()
    assert np.isnan(r.correlation[:, 1]).all()
    assert r.ci_lower[1] == r.ci_upper[1] == 0
    assert not r.uncertainty_complete
    assert any("do not mean zero uncertainty" in w for w in r.warnings)


def check_hessian_no_positive_modes_leaves_uncertainty_unavailable():
    for sign in (
        [0, -1]
    ):
        r = minionpy.Uncertainty(lambda p: sign*np.sum(np.asarray(p)**2, axis=1), [0, 0]).hessian(1)
        assert not r.covariance_available
        assert not r.covariance_valid
        assert r.retained_rank == 0
        assert len(r.discarded_modes) == 2
        assert r.covariance.size == 0
        assert not r.std
        assert any("No resolved positive" in w for w in r.warnings)


def check_hessian_partial_covariance_uses_reported_scaled_modes():
    scales = np.array([.01, 100.])
    matrix = np.array([[1., 2.], [2., 1.]])
    def objective(points):
        p = np.asarray(points) / scales
        return np.einsum("bi,ij,bj->b", p, matrix, p)/2
    r = minionpy.Uncertainty(objective, [0, 0]).hessian(1, parameter_scales=scales)
    np.testing.assert_allclose(r.scaled_eigenvalues, [-1, 3], atol=1e-12)
    v = scales*r.scaled_eigenvectors[:, 1]
    np.testing.assert_allclose(r.covariance, np.outer(v,v)/3, atol=1e-12)


def check_hessian_parameter_units_and_batch_accounting():
    calls = []
    scales = np.array([1e-4, 1e4])
    matrix = np.array([[3., .5], [.5, 2.]])

    def objective(points):
        calls.append(len(points))
        z = np.asarray(points) / scales
        return np.einsum("bi,ij,bj->b", z, matrix, z) / 2

    r = minionpy.Uncertainty(objective, [0, 0]).hessian(
        1, parameter_scales=scales, batch_size=3, check_stability=False,
        confidence_level=.9)
    assert r.covariance_valid
    assert r.nfev == 9 == sum(calls)
    assert r.objective_calls == len(calls) == 3
    np.testing.assert_allclose(r.covariance, np.diag(scales) @ np.linalg.inv(matrix) @ np.diag(scales))
    assert math.isclose(r.condition_number, np.linalg.cond(matrix), rel_tol=1e-6)
    np.testing.assert_allclose(np.asarray(r.ci_upper) / r.std, NormalDist().inv_cdf(.95))


def check_hessian_step_sensitivity_and_nonstationarity_are_reported():
    quartic = minionpy.Uncertainty(lambda p: np.asarray(p)[:, 0]**4, [0]).hessian(1, step_sizes=[.1])
    assert quartic.differentiation_error > .5
    assert any("sensitive" in w for w in quartic.warnings)
    off_center = minionpy.Uncertainty(lambda p: np.asarray(p)[:, 0]**2, [1]).hessian(1)
    assert any("stationary" in w for w in off_center.warnings)


def check_hessian_all_points_in_one_call():
    for batch_size in (
        [None, -1]
    ):
        calls = []

        def objective(points):
            calls.append(len(points))
            return np.sum(np.asarray(points)**2, axis=1)

        # Twelve parameters give 289 points, exceeding the old default of 256.
        analysis = minionpy.Uncertainty(objective, [0] * 12)
        kwargs = {} if batch_size is None else {"batch_size": batch_size}
        result = analysis.hessian(1, check_stability=False, **kwargs)
        assert calls == [289]
        assert result.objective_calls == 1
        assert result.nfev == 289
        calls.clear()
        result = analysis.hessian(1, **kwargs)
        assert calls == [289, 289]
        assert result.objective_calls == 2
        assert result.nfev == 578
        np.testing.assert_allclose(result.hessian, 2 * np.eye(12), atol=1e-12)


def check_hessian_invalid_options():
    for kwargs in (
        [
            {"scale": 0}, {"scale": float("nan")}, {"scale": 1, "confidence_level": 1},
            {"scale": 1, "batch_size": 0}, {"scale": 1, "step_sizes": [0]},
            {"scale": 1, "batch_size": -2},
            {"scale": 1, "step_sizes": [1, 2]}, {"scale": 1, "parameter_scales": [-1]},
        ]
    ):
        with _UNCERTAINTY_ASSERTIONS.assertRaises((ValueError, RuntimeError, TypeError)):
            minionpy.Uncertainty(lambda p: np.asarray(p)[:, 0]**2, [0]).hessian(**kwargs)


def check_hessian_bad_objective_results():
    for objective in (
        [lambda p: [], lambda p: [float("nan")] * len(p)]
    ):
        with _UNCERTAINTY_ASSERTIONS.assertRaises(RuntimeError):
            minionpy.Uncertainty(objective, [0]).hessian(1)


def check_hessian_callback_exception_propagates():
    def objective(points):
        raise ValueError("user objective failed")
    with _UNCERTAINTY_ASSERTIONS.assertRaisesRegex(ValueError, "user objective failed"):
        minionpy.Uncertainty(objective, [0]).hessian(1)


def check_hessian_invalid_analysis_inputs():
    for point,bounds in (
        [([2], [(0, 1)]), ([0], [(0, 0)]),
                                                 ([0, 0], [(0, 1)]), ([float("nan")], None)]
    ):
        with _UNCERTAINTY_ASSERTIONS.assertRaises((ValueError, RuntimeError)):
            minionpy.Uncertainty(lambda p: [0] * len(p), point, bounds)


_BOOTSTRAP_FIT_OPTIONS = {"x_tol": 1e-9, "f_tol": -1.0}


def _bootstrap_quadratic(center):
    center = np.asarray(center).copy()
    return lambda points: np.sum((np.asarray(points) - center)**2, axis=1)


def _bootstrap_for_test(analysis, sampler, **kwargs):
    settings = dict(n_samples=6, optimizer="NelderMead", maxevals=3000,
                    seed=41, options=_BOOTSTRAP_FIT_OPTIONS, require_convergence=True)
    settings.update(kwargs)
    return analysis.bootstrap(sampler, **settings)


def check_bootstrap_gaussian_polynomial_bootstrap_matches_reference_statistics():
    t = np.linspace(-.8, 1.7, 60)
    design = np.column_stack([np.ones_like(t), t, t**2])
    errors = .12 + .07*(t+.8)
    weights = design/errors[:, None]
    observations = design @ [.7, -1.2, .4] + np.random.default_rng(182).normal(0, errors)

    def objective(data):
        fixed = data.copy()
        return lambda p: np.sum(((np.asarray(p) @ design.T-fixed)/errors)**2, axis=1)

    fit = minionpy.Minimizer(objective(observations), [(-5, 5)]*3,
                            x0=[[0, 0, 0]], algo="NelderMead", maxevals=6000,
                            seed=17, options=_BOOTSTRAP_FIT_OPTIONS).optimize()
    analysis = minionpy.Uncertainty(objective(observations), fit.x, [(-5, 5)]*3)
    reference = []
    calls = []

    def sampler(index, sample_seed):
        calls.append(index)
        data = design @ fit.x + np.random.default_rng(sample_seed).normal(0, errors)
        reference.append(np.linalg.lstsq(weights, data/errors, rcond=None)[0])
        return objective(data)

    r = _bootstrap_for_test(analysis, sampler, n_samples=100, maxevals=6000, confidence_level=.9)
    assert calls == list(range(100))
    assert r.n_requested == r.n_valid == r.n_converged == r.n_used == 100
    assert r.covariance_available and r.covariance_valid
    assert r.status == "ok"
    assert len(r.realization_results) == 100
    assert r.used_indices == list(range(100))
    np.testing.assert_allclose(r.samples, reference, atol=2e-6)
    np.testing.assert_allclose(r.mean, r.samples.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(r.covariance, np.cov(r.samples, rowvar=False, ddof=1), atol=1e-12)
    np.testing.assert_allclose(r.std, np.std(r.samples, axis=0, ddof=1), atol=1e-12)
    np.testing.assert_allclose(r.correlation, np.corrcoef(r.samples, rowvar=False), atol=1e-12)
    np.testing.assert_allclose([r.ci_lower, r.ci_upper], np.percentile(r.samples, [5, 95], axis=0), atol=1e-12)
    h = analysis.hessian(2)
    assert np.all((np.asarray(r.std)/h.std > .7) & (np.asarray(r.std)/h.std < 1.4))
    assert r.nfev == sum(f.nfev for f in r.realization_results)


def check_bootstrap_reproducible_sampling_and_optimizer_seeds():
    analysis = minionpy.Uncertainty(_bootstrap_quadratic([0, 0]), [0, 0], [(-5, 5)]*2)
    def sampler(index, seed):
        return _bootstrap_quadratic(np.random.default_rng(seed).normal(size=2))
    a = _bootstrap_for_test(analysis, sampler)
    b = _bootstrap_for_test(analysis, sampler)
    np.testing.assert_array_equal(a.samples, b.samples)
    assert a.sample_seeds == b.sample_seeds
    assert a.optimizer_seeds == b.optimizer_seeds
    assert len(set(a.sample_seeds)) == 6
    c = _bootstrap_for_test(analysis, sampler, seed=42)
    assert a.sample_seeds != c.sample_seeds
    assert not np.array_equal(a.samples, c.samples)
    # Optimizer draws and budgets must not affect the statistical seed stream.
    d = _bootstrap_for_test(analysis, sampler, optimizer="DE", maxevals=60,
                  require_convergence=False, options={"population_size": 10})
    assert a.sample_seeds == d.sample_seeds
    assert a.optimizer_seeds == d.optimizer_seeds


def check_bootstrap_sampler_called_once_and_realizations_remain_fixed():
    calls, evaluations = [], [0]*6
    def sampler(index, seed):
        calls.append(index)
        center = index/10
        def objective(points):
            evaluations[index] += len(points)
            return (np.asarray(points)[:, 0]-center)**2
        return objective
    analysis = minionpy.Uncertainty(_bootstrap_quadratic([0]), [0], [(-5, 5)])
    r = _bootstrap_for_test(analysis, sampler)
    assert calls == list(range(6))
    assert min(evaluations) > 1
    assert sum(evaluations) == r.nfev
    np.testing.assert_allclose(r.samples[:, 0], np.arange(6)/10, atol=1e-7)


def check_bootstrap_automatically_generated_master_seed_can_be_reused():
    analysis = minionpy.Uncertainty(_bootstrap_quadratic([0]), [0], [(-5, 5)])
    sampler = lambda index, seed: _bootstrap_quadratic([np.random.default_rng(seed).uniform(-1, 1)])
    a = _bootstrap_for_test(analysis, sampler, seed=None)
    b = _bootstrap_for_test(analysis, sampler, seed=a.seed)
    np.testing.assert_array_equal(a.samples, b.samples)
    assert a.sample_seeds == b.sample_seeds


def check_bootstrap_failures_preserved_and_exclusions_reported():
    def sampler(index, seed):
        if index == 1:
            raise ValueError("invalid simulated data")
        if index == 2:
            return None
        if index == 3:
            return lambda points: []
        if index == 4:
            return lambda points: [float("nan")]*len(points)
        return _bootstrap_quadratic([index/10])
    r = _bootstrap_for_test(minionpy.Uncertainty(_bootstrap_quadratic([0]), [0], [(-5, 5)]), sampler)
    assert r.n_valid == r.n_converged == r.n_used == 2
    assert r.used_indices == [0, 5]
    assert r.covariance_available and not r.covariance_valid
    assert r.status == "warning"
    assert np.isnan(r.samples[1:5]).all()
    for i in range(1, 5):
        assert r.realization_results[i].status == minionpy.TerminationStatus.RuntimeError
        assert r.realization_results[i].message
    assert "invalid simulated data" in r.realization_results[1].message
    assert any("bias" in warning for warning in r.warnings)
    np.testing.assert_allclose(r.std, np.std(r.samples[[0, 5]], axis=0, ddof=1))


def check_bootstrap_convergence_policy_and_insufficient_samples():
    analysis = minionpy.Uncertainty(_bootstrap_quadratic([0]), [0], [(-5, 5)])
    sampler = lambda index, seed: _bootstrap_quadratic([1+index/10])
    strict = _bootstrap_for_test(analysis, sampler, options={"maxiters": 0})
    assert strict.n_valid == 6 and strict.n_converged == strict.n_used == 0
    assert strict.status == "insufficient_samples"
    assert strict.covariance.size == 0
    assert not strict.covariance_available
    assert np.isfinite(strict.samples).all()
    permissive = _bootstrap_for_test(analysis, sampler, options={"maxiters": 0}, require_convergence=False)
    assert permissive.n_used == 6 and permissive.covariance_available
    assert not permissive.covariance_valid
    assert any("optimization error" in w for w in permissive.warnings)


def check_bootstrap_zero_spread_has_undefined_correlation():
    analysis = minionpy.Uncertainty(_bootstrap_quadratic([0]), [0], [(-5, 5)])
    # A constant objective with fixed initial simplex also provides zero spread.
    r = _bootstrap_for_test(analysis, lambda index, seed: lambda p: np.zeros(len(p)),
                  options={"maxiters": 0}, require_convergence=False)
    assert r.std == [0.0]
    assert np.isnan(r.correlation).all()
    assert any("zero observed" in w for w in r.warnings)


def check_bootstrap_invalid_configuration_is_rejected_before_sampling():
    for kwargs in (
        [
            {"n_samples": 1}, {"maxevals": 0}, {"seed": -2},
            {"confidence_level": 0}, {"confidence_level": 1},
            {"optimizer": "invalid"},
        ]
    ):
        calls = []
        with _UNCERTAINTY_ASSERTIONS.assertRaises((ValueError, RuntimeError, TypeError)):
            _bootstrap_for_test(minionpy.Uncertainty(_bootstrap_quadratic([0]), [0], [(-5, 5)]),
                      lambda index, seed: calls.append(index), **kwargs)
        assert not calls


def check_bootstrap_no_accepted_fits():
    r = _bootstrap_for_test(minionpy.Uncertainty(_bootstrap_quadratic([0]), [0], [(-5, 5)]), lambda i, s: None)
    assert r.status == "insufficient_samples"
    assert r.n_valid == r.n_used == 0
    assert not r.covariance_available
    assert np.isnan(r.samples).all()


def check_bootstrap_default_arrde_keeps_finite_budget_limited_fits_with_warning():
    analysis = minionpy.Uncertainty(_bootstrap_quadratic([0]), [0], [(-5, 5)])
    r = analysis.bootstrap(lambda i, s: _bootstrap_quadratic([i/10]), n_samples=3,
                           maxevals=100, seed=19)
    assert r.n_valid == r.n_used == 3
    assert r.covariance_available
    assert not r.covariance_valid
    assert any("optimization error" in w for w in r.warnings)


def check_bootstrap_bounds_required_for_bounded_optimizer():
    with _UNCERTAINTY_ASSERTIONS.assertRaisesRegex(RuntimeError, "bounds"):
        _bootstrap_for_test(minionpy.Uncertainty(_bootstrap_quadratic([0]), [0]), lambda i, s: _bootstrap_quadratic([1]))


def check_bootstrap_unconstrained_lbfgs_bootstrap():
    r = _bootstrap_for_test(minionpy.Uncertainty(_bootstrap_quadratic([0]), [0]),
                  lambda i, s: _bootstrap_quadratic([i/10]), optimizer="L_BFGS")
    assert r.n_used == 6
    np.testing.assert_allclose(r.samples[:, 0], np.arange(6)/10, atol=1e-5)


def check_bootstrap_bounded_bootstrap_without_warm_start():
    analysis = minionpy.Uncertainty(_bootstrap_quadratic([0]), [0], [(-5, 5)])
    r = _bootstrap_for_test(analysis, lambda i, s: _bootstrap_quadratic([i/10]), warm_start=False)
    assert r.n_used == 6
    np.testing.assert_allclose(r.samples[:, 0], np.arange(6)/10, atol=1e-7)


def check_bootstrap_single_accepted_fit_cannot_produce_covariance():
    r = _bootstrap_for_test(minionpy.Uncertainty(_bootstrap_quadratic([0]), [0], [(-5, 5)]),
                  lambda i, s: _bootstrap_quadratic([1]) if i == 0 else None)
    assert r.n_used == 1
    assert r.status == "insufficient_samples"
    assert not r.covariance_available
    assert r.covariance.size == 0
    assert not r.ci_lower


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
        ("L-BFGS line-search fallback stagnates", check_lbfgs_line_search_fallback_stagnates),
        ("GradientDescent backtracking rescues an oversized step", check_gradient_descent_backtracking_improves_large_step),
        ("RDEX direct and generic wrappers run", check_rdex_direct_wrapper),
    ]
    checks.extend([
        ('hessian: polynomial fit with known observation errors', check_hessian_polynomial_fit_with_known_observation_errors),
        ('hessian: bounds and one sided mixed derivatives', check_hessian_bounds_and_one_sided_mixed_derivatives),
        ('hessian: unidentified or negative curvature has no covariance', check_hessian_unidentified_or_negative_curvature_has_no_covariance),
        ('hessian: partial uncertainty discards unresolved modes', check_hessian_partial_uncertainty_discards_unresolved_modes),
        ('hessian: discarded parameter zero is not reported as a valid uncertainty', check_hessian_discarded_parameter_zero_is_not_reported_as_a_valid_uncertainty),
        ('hessian: no positive modes leaves uncertainty unavailable', check_hessian_no_positive_modes_leaves_uncertainty_unavailable),
        ('hessian: partial covariance uses reported scaled modes', check_hessian_partial_covariance_uses_reported_scaled_modes),
        ('hessian: parameter units and batch accounting', check_hessian_parameter_units_and_batch_accounting),
        ('hessian: step sensitivity and nonstationarity are reported', check_hessian_step_sensitivity_and_nonstationarity_are_reported),
        ('hessian: all points in one call', check_hessian_all_points_in_one_call),
        ('hessian: invalid options', check_hessian_invalid_options),
        ('hessian: bad objective results', check_hessian_bad_objective_results),
        ('hessian: callback exception propagates', check_hessian_callback_exception_propagates),
        ('hessian: invalid analysis inputs', check_hessian_invalid_analysis_inputs),
        ('bootstrap: gaussian polynomial bootstrap matches reference statistics', check_bootstrap_gaussian_polynomial_bootstrap_matches_reference_statistics),
        ('bootstrap: reproducible sampling and optimizer seeds', check_bootstrap_reproducible_sampling_and_optimizer_seeds),
        ('bootstrap: sampler called once and realizations remain fixed', check_bootstrap_sampler_called_once_and_realizations_remain_fixed),
        ('bootstrap: automatically generated master seed can be reused', check_bootstrap_automatically_generated_master_seed_can_be_reused),
        ('bootstrap: failures preserved and exclusions reported', check_bootstrap_failures_preserved_and_exclusions_reported),
        ('bootstrap: convergence policy and insufficient samples', check_bootstrap_convergence_policy_and_insufficient_samples),
        ('bootstrap: zero spread has undefined correlation', check_bootstrap_zero_spread_has_undefined_correlation),
        ('bootstrap: invalid configuration is rejected before sampling', check_bootstrap_invalid_configuration_is_rejected_before_sampling),
        ('bootstrap: no accepted fits', check_bootstrap_no_accepted_fits),
        ('bootstrap: default arrde keeps finite budget limited fits with warning', check_bootstrap_default_arrde_keeps_finite_budget_limited_fits_with_warning),
        ('bootstrap: bounds required for bounded optimizer', check_bootstrap_bounds_required_for_bounded_optimizer),
        ('bootstrap: unconstrained lbfgs bootstrap', check_bootstrap_unconstrained_lbfgs_bootstrap),
        ('bootstrap: bounded bootstrap without warm start', check_bootstrap_bounded_bootstrap_without_warm_start),
        ('bootstrap: single accepted fit cannot produce covariance', check_bootstrap_single_accepted_fit_cannot_produce_covariance),
    ])
    return run_checks(checks)


if __name__ == "__main__":
    raise SystemExit(main())
