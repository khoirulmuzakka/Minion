import importlib.util
import os
import sys
from pathlib import Path

import iminuit
import numpy as np
from scipy.optimize import minimize

sys.path.append("../")
import minionpy as mpy
import minionpy.test_functions as mpytest


def load_ntqn():
    candidates = []

    env_path = os.environ.get("NTQN_PATH")
    if env_path:
        candidates.append(Path(env_path))

    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    candidates.extend(
        [
            repo_root / "external" / "noise-tolerant-bfgs" / "ntqn.py",
            repo_root / "noise-tolerant-bfgs" / "ntqn.py",
            here.parent / "noise-tolerant-bfgs" / "ntqn.py",
        ]
    )

    for candidate in candidates:
        candidate = candidate.resolve()
        if not candidate.is_file():
            continue
        spec = importlib.util.spec_from_file_location("ntqn", candidate)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module.np, "float"):
            module.np.float = float
        patch_ntqn(module)
        return module, candidate

    return None, None


def patch_ntqn(module):
    original = module._line_search_nt_wolfe

    def wrapped(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except UnboundLocalError:
            x_k = np.asarray(args[2], dtype=float)
            grad = args[1]
            func = args[0]

            f_k = kwargs.get("f_k")
            if f_k is None:
                f_k = func(x_k)

            g_k = kwargs.get("g_k")
            if g_k is None:
                g_k = grad(x_k)

            eps_f = kwargs.get("eps_f", 0.0)
            eps_g = kwargs.get("eps_g", 0.0)

            if callable(eps_f):
                eps_fk = eps_f(x_k)
            else:
                eps_fk = np.copy(eps_f)

            if callable(eps_g):
                eps_gk = eps_g(x_k)
            else:
                eps_gk = np.copy(eps_g)

            return 0.0, 0.0, np.inf, f_k, g_k, eps_fk, eps_gk, 0, 0, False, False, False

    module._line_search_nt_wolfe = wrapped


N = 0
noise_ratio = 1e-4
N_dict = {}


class MaxEvalExceeded(RuntimeError):
    pass


TEST_FUNCTIONS = {
    "sphere": mpytest.sphere,
    "sum_squares": mpytest.sum_squares,
    "zakharov": mpytest.zakharov,
    "bent_cigar": mpytest.bent_cigar,
    "discus": mpytest.discus,
    "dixon_price": mpytest.dixon_price,
    "rosenbrock": mpytest.rosenbrock,
    "happy_cat": mpytest.happy_cat,
    "hgbat": mpytest.hgbat,
    "step": mpytest.step,
}

CEC2017_FUNCTION_NUMBERS = range(1, 31)

def make_rotation_matrix(dimension, rng):
    q, r = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    diag = np.sign(np.diag(r))
    diag[diag == 0.0] = 1.0
    return q * diag


def make_transformed_function(test_func, bounds, dimension, func_name):
    lo, hi = bounds
    transform_seed = 1000 + sum(ord(ch) for ch in func_name)
    transform_rng = np.random.default_rng(transform_seed)
    rotation = make_rotation_matrix(dimension, transform_rng)
    max_shift = min(2.0, 0.1 * (hi - lo))
    shift = transform_rng.uniform(-max_shift, max_shift, size=dimension)
    value_shift = 0.0# float(transform_rng.uniform(10.0, 100.0))

    def transformed(X):
        X = np.asarray(X, dtype=float)
        Z = (X - shift) @ rotation
        return test_func(Z) + value_shift

    def transformed_scalar(x):
        x = np.asarray(x, dtype=float)
        z = (x - shift) @ rotation
        return float(test_func([z])[0] + value_shift)

    return transformed, transformed_scalar


def test_optimization_noise(test_func, bounds, dimension, func_name, maxevals, seed, ntqn_module=None):
    global N, N_dict, noise_ratio

    use_transform = not func_name.startswith("CEC2017 ")
    result = {}
    result["Dimensions"] = dimension
    result["Function"] = f"{func_name} (shift+rotate)" if use_transform else func_name
    bounds_list = [bounds] * dimension
    x0 = [[0.0 for _ in range(dimension)]]
    if use_transform:
        transformed_func, transformed_scalar = make_transformed_function(test_func, bounds, dimension, func_name)
    else:
        def transformed_func(X):
            return np.asarray(test_func(X), dtype=float)

        def transformed_scalar(x):
            return float(test_func([np.asarray(x, dtype=float).tolist()])[0])

    rng = np.random.default_rng(seed)
    state = {"best": np.inf}

    def func_wrapper(X):
        global N
        if N >= maxevals:
            raise MaxEvalExceeded()
        ret = np.array(transformed_func(X))
        noisy = ret + noise_ratio * rng.normal(size=len(X)) * np.abs(ret)
        state["best"] = min(state["best"], float(np.min(noisy)))
        N += len(X)
        return noisy

    def func_scipy(par):
        return func_wrapper([par])[0]

    def func_minuit(par):
        global N
        if N >= maxevals:
            raise MaxEvalExceeded()
        value = transformed_scalar(par)
        noisy = value + noise_ratio * rng.normal() * abs(value)
        state["best"] = min(state["best"], float(noisy))
        N += 1
        return float(noisy)

    def func_clean(par):
        return np.array(transformed_func([np.asarray(par, dtype=float)]))[0]

    def ntqn_grad(par):
        x = np.asarray(par, dtype=float)
        grad = np.zeros_like(x)
        h_base = 1e-5
        for i in range(len(x)):
            h = h_base * max(1.0, abs(x[i]))
            xp = x.copy()
            xm = x.copy()
            xp[i] += h
            xm[i] -= h
            grad[i] = (func_scipy(xp) - func_scipy(xm)) / (2.0 * h)
        return grad

    def ntqn_eps_f(par):
        return noise_ratio * abs(func_clean(par))

    def ntqn_eps_g(par):
        x = np.asarray(par, dtype=float)
        comp = np.zeros_like(x)
        h_base = 1e-5
        for i in range(len(x)):
            h = h_base * max(1.0, abs(x[i]))
            xp = x.copy()
            xm = x.copy()
            xp[i] += h
            xm[i] -= h
            comp[i] = (ntqn_eps_f(xp) + ntqn_eps_f(xm)) / (2.0 * h)
        return float(np.linalg.norm(comp))

    def run_minion_bounded(algo, label, n_points, extra_options=None):
        global N
        N = 0
        state["best"] = np.inf
        options = {
            "population_size": 0,
            "N_points_derivative": n_points,
            "func_noise_ratio": noise_ratio,
            "convergence_tol" : 0.0
        }
        if extra_options:
            options.update(extra_options)
        try:
            res = mpy.Minimizer(
                func_wrapper,
                bounds_list,
                x0=x0,
                algo=algo,
                maxevals=maxevals,
                callback=None,
                seed=seed,
                options=options,
            ).optimize()
            result[label] = res.fun
        except MaxEvalExceeded:
            result[label] = state["best"]
        N_dict[label] = N

    for n_points in (1, 3, 5):
        run_minion_bounded(
            "L_BFGS_B",
            f"L_BFGS_B N={n_points}",
            n_points,
            {"use_local_search": True},
        )

    N = 0
    state["best"] = np.inf
    try:
        res = mpy.Minimizer(
            func_wrapper,
            bounds_list,
            x0=x0,
            relTol=0.0,
            algo="ARRDE",
            maxevals=maxevals,
            callback=None,
            seed=seed,
            options={
                "population_size": 0,
                "use_local_search": True,
                "func_noise_ratio": noise_ratio,
            },
        ).optimize()
        result["ARRDE"] = res.fun
    except MaxEvalExceeded:
        result["ARRDE"] = state["best"]
    N_dict["ARRDE"] = N

    N = 0
    state["best"] = np.inf
    try:
        res = mpy.Minimizer(
            func_wrapper,
            bounds_list,
            x0=x0,
            relTol=0.0,
            algo="CMAES",
            maxevals=maxevals,
            callback=None,
            seed=seed,
            options={
                "population_size": 0,
            },
        ).optimize()
        result["CMAES"] = res.fun
    except MaxEvalExceeded:
        result["CMAES"] = state["best"]
    N_dict["CMAES"] = N

    N = 0
    state["best"] = np.inf
    try:
        res = mpy.L_BFGS(
            func_wrapper,
            x0=x0,
            relTol=0.0,
            maxevals=maxevals,
            callback=None,
            seed=seed,
            options={
                "population_size": 0,
                "N_points_derivative": 1,
                "use_local_search": True,
                "func_noise_ratio": noise_ratio,
            },
        ).optimize()
        result["L_BFGS"] = res.fun
    except MaxEvalExceeded:
        result["L_BFGS"] = state["best"]
    N_dict["L_BFGS"] = N

    N = 0
    state["best"] = np.inf
    try:
        res_minimize = minimize(
            func_scipy,
            x0=x0[0],
            method="L-BFGS-B",
            options={"maxfun": maxevals},
            bounds=bounds_list,
        )
        result["Scipy L_BFGS_B"] = res_minimize.fun
    except MaxEvalExceeded:
        result["Scipy L_BFGS_B"] = state["best"]
    N_dict["Scipy L_BFGS_B"] = N

    N = 0
    state["best"] = np.inf
    try:
        res_minimize = minimize(
            func_scipy,
            x0=x0[0],
            method="BFGS",
            options={"maxiter": maxevals},
        )
        result["Scipy BFGS"] = res_minimize.fun
    except MaxEvalExceeded:
        result["Scipy BFGS"] = state["best"]
    N_dict["Scipy BFGS"] = N

    N = 0
    state["best"] = np.inf
    try:
        def minuit_objective(*args):
            return func_minuit(np.array(args, dtype=float))

        m = iminuit.Minuit(minuit_objective, *x0[0])
        for i, bound in enumerate(bounds_list):
            m.limits[i] = bound
        m.errordef = 1.0
        m.migrad(ncall=maxevals)
        result["Minuit Migrad"] = float(m.fval)
    except MaxEvalExceeded:
        result["Minuit Migrad"] = state["best"]
    except RuntimeError:
        try:
            last_x = np.array(list(m.values), dtype=float)
            result["Minuit Migrad"] = func_clean(last_x)
        except Exception:
            result["Minuit Migrad"] = state["best"]
    except Exception as exc:
        result["Minuit Migrad"] = f"error={type(exc).__name__}"
    N_dict["Minuit Migrad"] = N

    if ntqn_module is not None:
        N = 0
        state["best"] = np.inf
        try:
            _, f_opt, _, _, _, _, _ = ntqn_module.bfgs_e(
                func_scipy,
                ntqn_grad,
                np.array(x0[0], dtype=float),
                eps_f=ntqn_eps_f,
                eps_g=ntqn_eps_g,
                options={
                    "display": 0,
                    "max_iter": maxevals,
                    "max_feval": maxevals,
                    "max_geval": maxevals,
                    "qn_hist_size": 10,
                    "terminate": 0,
                },
            )
            result["NTQN L_BFGS"] = f_opt
        except MaxEvalExceeded:
            result["NTQN L_BFGS"] = state["best"]
        N_dict["NTQN L_BFGS"] = N

    for key in result:
        if key == "Function":
            print("Function : ", result[key])
        if key not in ["Dimensions", "Function"]:
            print(f"\t{key:<20} : {str(result[key]):<20} \t N_evals : {N_dict[key]:<10}")
    print("")


def main():
    global noise_ratio

    noise_ratio = 0.0
    maxevals = 10000
    dimension = 30
    bounds = (-10, 10)
    cec_dimension = 30
    cec_bounds = (-100, 100)

    ntqn_module, ntqn_path = load_ntqn()
    if ntqn_module is not None:
        print(f"Using NTQN from: {ntqn_path}")

    for func_name, test_func in TEST_FUNCTIONS.items():
        test_optimization_noise(
            test_func,
            bounds,
            dimension,
            func_name,
            maxevals,
            None,
            ntqn_module=ntqn_module,
        )

    for func_number in CEC2017_FUNCTION_NUMBERS:
        cec_func = mpy.CEC2017Functions(function_number=func_number, dimension=cec_dimension)
        test_optimization_noise(
            cec_func,
            cec_bounds,
            cec_dimension,
            f"CEC2017 func_{func_number}",
            maxevals,
            None,
            ntqn_module=ntqn_module,
        )


if __name__ == "__main__":
    main()
