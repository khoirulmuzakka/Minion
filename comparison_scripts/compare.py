import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata
import math
import matplotlib.pyplot as plt


try:
    from IPython.display import display  # type: ignore
except ImportError:  # pragma: no cover - CLI fallback
    def display(obj):
        print(obj)

def MWUT(A, B, alpha=0.05, mode="ranksum"):
    """
    Perform the Mann-Whitney U Test (Wilcoxon rank-sum test) to compare two independent samples.

    The function assumes that *smaller values are better* (e.g., errors),
    and returns:
        1  if A is significantly better (lower) than B,
       -1  if A is significantly worse (higher) than B,
        0  if there is no statistically significant difference.

    Parameters
    ----------
    A : array-like
        First independent sample.
    B : array-like
        Second independent sample.
    alpha : float, optional (default=0.05)
        Significance level.
    mode : {"ranksum", "mean", "median"}, optional
        Criterion to decide which side is "better" once significance is established:
        - "ranksum": compare rank sums (nonparametric, consistent with MWU)
        - "mean":    compare means
        - "median":  compare medians

    Returns
    -------
    int
        1  if A is significantly lower than B,
       -1  if A is significantly higher than B,
        0  if no significant difference (p >= alpha) or exact tie by chosen mode.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    if A.ndim != 1 or B.ndim != 1:
        raise ValueError("A and B must be 1D arrays.")

    # Mann-Whitney U test (two-sided)
    # By SciPy's convention, 'statistic' is U for the first sample (A).
    statistic, p_value = mannwhitneyu(A, B, alternative='two-sided')

    # If not significant, return 0 (cannot reject H0)
    if p_value >= alpha:
        return 0

    # At this point, we know the distributions differ significantly.
    # Now decide direction using the requested 'mode'.
    if mode == "ranksum":
        nA = len(A)
        nB = len(B)
        U_A = statistic
        assert (nA == nB)

        # Rank-sum for A: R_A = U_A + nA(nA + 1)/2
        rank_sum_A = U_A + nA * (nA + 1) / 2.0

        # Total rank sum for all observations 1..(nA+nB)
        total_rank_sum = (nA + nB) * (nA + nB + 1) / 2.0

        # Rank-sum for B is what's left
        rank_sum_B = total_rank_sum - rank_sum_A

        if rank_sum_A < rank_sum_B:
            return 1    # A has lower ranks → better
        elif rank_sum_A > rank_sum_B:
            return -1   # A has higher ranks → worse
        else:
            return 0    # extremely rare: exact tie in rank sums

    elif mode == "mean":
        meanA = np.mean(A)
        meanB = np.mean(B)

        if meanA < meanB:
            return 1
        elif meanA > meanB:
            return -1
        else:
            return 0

    elif mode == "median":
        medianA = np.median(A)
        medianB = np.median(B)

        if medianA < medianB:
            return 1
        elif medianA > medianB:
            return -1
        else:
            return 0

    else:
        raise Exception("Unknown mode: {}".format(mode))


def calcRank(arrDict, ideal, mode="ranksum"):
    """
    Rank algorithms by aggregating run-level absolute errors against the ideal.

    Parameters
    ----------
    arrDict : dict[str, array-like]
        Mapping algo -> list/array of run outcomes for a single (dim, func).
    ideal : float
        Known optimum that runs should approach.
    mode : {"ranksum", "mean", "median"}
        Aggregation strategy before ranking (default: "mean").

    Returns
    -------
    dict[str, float]
        rankdata-style ranks per algorithm (1 = best).
    """
    algo_names = list(arrDict.keys())
    experiments = []
    for algo in algo_names:
        experiments.append(np.abs(np.array(arrDict[algo], dtype=float) - ideal))
    try:
        experiments = np.array(experiments, dtype=float)
    except Exception as exc:
        for exp in experiments:
            print(len(exp))
        print(ideal, experiments)
        raise exc

    if experiments.ndim == 1:
        experiments = experiments[:, None]

    if mode == "ranksum":
        flattened_data = experiments.flatten()
        ranks = rankdata(flattened_data)
        ranked_experiments = ranks.reshape(experiments.shape)
        total_ranks = np.sum(ranked_experiments, axis=1)
    elif mode == "mean":
        total_ranks = np.mean(experiments, axis=1)
    elif mode == "median":
        total_ranks = np.median(experiments, axis=1)
    else:
        raise Exception("Unknown mode.")
    final_rank = rankdata(total_ranks, method="average")
    #print(final_rank)
    return dict(zip(algo_names, final_rank))

def calcRankSum(arrDict): 
    algo_names = list(arrDict.keys())
    experiments = []
    for algo in algo_names:
        experiments.append(np.array(arrDict[algo], dtype=float))
    try:
        experiments = np.array(experiments, dtype=float)
    except Exception as exc:
        for exp in experiments:
            print(len(exp))
        raise exc
    #print(arrDict)
    flattened_data = experiments.flatten()
    ranks = rankdata(flattened_data)
    ranked_experiments = ranks.reshape(experiments.shape)
    #print(ranked_experiments)
    sum_rank = np.sum(ranked_experiments, axis=1)
    return dict(zip(algo_names, sum_rank))


goptimum_cec22 = np.array([300, 400, 600, 800, 900, 1800, 2000, 2200, 2300, 2400, 2600, 2700])
goptimum_cec20 = np.array([100, 1100, 700, 1900, 1700, 1600, 2100, 2200, 2400, 2500])
goptimum_cec17 = np.array([100 * i for i in range(1, 31)])
goptimum_cec14 = goptimum_cec17
goptimum_cec11 = None  # inferred from data (min over algorithms) because the official optima are unknown
yearToMin = {2017: goptimum_cec17, 2014: goptimum_cec14, 2020: goptimum_cec20, 2022: goptimum_cec22, 2011: goptimum_cec11, 
             2019: np.ones(10)}
yearToNfuncs = {2017:30, 2014:30, 2020:10, 2022:12, 2011:22, 2019 : 10}


def default_dim_to_year(multiplier=10000):
    """
    Helper to reproduce the DIM_MAX_EVALS maps used in notebooks/scripts.
    """
    return {
        2017: {10: 10*multiplier, 30: 30*multiplier, 50: 50*multiplier, 100: 100*multiplier},
        2014: {10: 10*multiplier, 30: 30*multiplier, 50: 50*multiplier, 100: 100*multiplier},
        2020: {5: 50000, 10: 1000000, 15: 3000000, 20: 10000000},
        2022: {10: 200000, 20: 1000000},
        2011 : {5:50000, 10:100000,15:150000 },
        2019 : {10:100000000}
    }



#################################################################
# LOADING
#################################################################

def _resolve_result_file(res_folder, year, algo, dim, maxevals):
    """
    Return a case-insensitive path to the results file for an algorithm/dimension/maxeval combo.
    """
    fname = f"results_{year}_{algo}_{dim}_{maxevals}.txt"
    path = os.path.join(res_folder, fname)
    if os.path.exists(path):
        return path
    directory = os.path.dirname(path)
    target = os.path.basename(path).lower()
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    for entry in os.listdir(directory):
        if entry.lower() == target:
            return os.path.join(directory, entry)
    raise FileNotFoundError(f"File not found (case-insensitive search): {path}")


def load_results(res_folder, algos, year, dim_maxevals, max_runs=51):
    """
    matrices[algo][idim] = array of shape (runs, n_funcs)
    dims_list[idim]      = the actual dimension value (e.g. 10, 20, 50...)
    """
    matrices = {}
    dims_sorted = list(dim_maxevals.items())  # deterministic order
    for algo in algos:
        per_algo_dims = []
        for dim, maxevals in dims_sorted:
            filepath = _resolve_result_file(res_folder, year, algo, dim, maxevals)
            data = np.loadtxt(filepath, delimiter="\t")[:max_runs]
            per_algo_dims.append(data)
        matrices[algo] = per_algo_dims
    return matrices, [d for d, _ in dims_sorted]


#################################################################
# HELPERS
#################################################################

def valid_funcs_iter(year, n_funcs, drop_index=None):
    """
    Handle CEC quirks like skipping function index 1 in 2017.
    """
    drop_set = set()
    if drop_index is not None:
        if np.isscalar(drop_index):
            drop_set = {int(drop_index)}
        else:
            drop_set = {int(idx) for idx in drop_index}
    for f in range(n_funcs):
        if year == 2017 and f == 1:
            continue
        if f in drop_set:
            continue
        yield f

def dim_weights(num_dims, year=None):
    """
    We only use weights for SE, SR (your scoring metrics).
    Friedman per dim, Z per dim, head2head per dim are unweighted.
    """
    if year == 2011:
        return [1.0] * num_dims  # CEC2011 uses uniform weights across dimensions
    return [0.1 * (i + 1) for i in range(num_dims)]


def infer_glob_min_from_runs(matrices, n_funcs):
    """
    Estimate per-function optima by taking the minimum value observed across all algos/dims/runs.
    Used for competitions where the official optimum is unknown (e.g., CEC2011).
    """
    inferred = np.full(n_funcs, np.inf, dtype=float)
    for dim_list in matrices.values():
        for arr in dim_list:
            inferred = np.minimum(inferred, np.min(arr, axis=0))
    if np.isinf(inferred).any():
        raise ValueError("Could not infer minima from the provided results.")
    return inferred

def compute_mean_abs_errors(matrices, glob_min, n_funcs):
    """
    mean_err[algo][idim][ifunc] = mean over runs of |result - global_min[ifunc]|
    -> dict algo -> list over dims -> array length n_funcs
    """
    mean_err = {algo: [] for algo in matrices.keys()}
    for algo, dim_list in matrices.items():
        for arr in dim_list:  # arr shape (runs, n_funcs)
            ae = np.mean(np.abs(arr - glob_min), axis=0)  # (n_funcs,)
            mean_err[algo].append(ae)
    return mean_err


def compute_best_worst_per_dim_func(matrices):
    """
    Collect best (min) and worst (max) observed fitness per (dimension, function)
    across all algorithms and runs. Returns two lists (best, worst) where each
    element is an array of length n_funcs for a given dimension index.
    """
    algos = list(matrices.keys())
    if not algos:
        raise ValueError("No algorithm data provided.")
    ndims = len(matrices[algos[0]])
    if ndims == 0:
        raise ValueError("No dimension data available.")
    n_funcs = matrices[algos[0]][0].shape[1]

    best_vals = [np.full(n_funcs, np.inf) for _ in range(ndims)]
    worst_vals = [np.full(n_funcs, -np.inf) for _ in range(ndims)]

    for algo in algos:
        if len(matrices[algo]) != ndims:
            raise ValueError("Inconsistent number of dimensions across algorithms.")
        for idim, arr in enumerate(matrices[algo]):
            if arr.shape[1] != n_funcs:
                raise ValueError("Inconsistent number of functions across dimensions.")
            best_vals[idim] = np.minimum(best_vals[idim], np.min(arr, axis=0))
            worst_vals[idim] = np.maximum(worst_vals[idim], np.max(arr, axis=0))

    return best_vals, worst_vals


def compute_mean_abs_errors_to_best(matrices, per_dim_best):
    """
    mean_best_err[algo][idim][ifunc] = mean over runs of |result - f_best(dim, func)|
    """
    mean_best_err = {algo: [] for algo in matrices.keys()}
    for idim, best_vals in enumerate(per_dim_best):
        for algo, dim_list in matrices.items():
            arr = dim_list[idim]
            ae = np.mean(np.abs(arr - best_vals), axis=0)
            mean_best_err[algo].append(ae)
    return mean_best_err


def compute_best_values_per_algo(matrices):
    """
    best_vals[algo][idim][ifunc] = min over runs of raw fitness for that algo/dim/func.
    """
    best_vals = {}
    for algo, dim_list in matrices.items():
        per_dim = []
        for arr in dim_list:
            per_dim.append(np.min(arr, axis=0))
        best_vals[algo] = per_dim
    return best_vals

def per_algo_mean_abs(matrices, idim, ifunc, ideal, year):
    """
    Helper: get mean abs error for each algo at a specific (dim, func),
    averaged over runs.
    Returns dict {algo: float}
    """
    perf = {}
    for algo, dim_list in matrices.items():
        samples = dim_list[idim][:, ifunc]  # (runs,)
        perf[algo] = np.mean(np.abs(samples - ideal))
    return perf


def abs_error_samples(matrices, algo, idim, ifunc, ideal):
    """
    Absolute error samples for a specific algo/dim/function across runs.
    """
    return np.abs(matrices[algo][idim][:, ifunc] - ideal)


#################################################################
# METRICS
#################################################################

def calc_SE(
    mean_err_opt,
    weights,
    year,
    n_funcs,
    glob_min,
    mode_number=1,
    mean_err_best=None,
    per_dim_ranges=None,
    best_err_opt=None,
    best_mode4_denoms=None,
    best_abs_vals=None,
    mode0_denoms=None,
    drop_index=None,
):
    """
    Weighted across dimensions.
    Returns both totals and per-dimension contributions (already weight-scaled).
    Lower = better.
    mode 1 -> |f - f*|
    mode 2 -> |f - f*| / |f*|
    mode 0 -> |f - f*| / (f_max - f*)
    mode 1 -> |f - f*|
    mode 2 -> |f - f*| / |f*|
    mode 3 -> |f - f_best| / (f_worst - f_best)
    mode 4 -> |f(x_best) - f*| / (max_algo f(x_best) - f*)
    mode 5 -> normalized mode 2 per function: (value - min) / (max - min)
    mode 6 -> |f - f_best| / |f_best|
    mode 7 -> normalized mode 6 per function
    mode 8 -> |f - f*| / |f*| scaled as re/(1+re)
    mode 9 -> |f - f_best| / |f_best| scaled as re/(1+re)
    """
    if mode_number not in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
        raise ValueError("mode_number must be between 0 and 9.")
    se = {algo: 0.0 for algo in mean_err_opt.keys()}
    ndims = len(weights)
    se_dim = {algo: [0.0]*ndims for algo in mean_err_opt.keys()}

    for idim in range(ndims):
        coeff = weights[idim]
        count_funcs= 0
        for ifunc in valid_funcs_iter(year, n_funcs, drop_index=drop_index):
            count_funcs += 1
            if mode_number == 0:
                if mode0_denoms is None:
                    raise ValueError("mode_number=0 requires mode0_denoms.")
                denom = mode0_denoms[idim][ifunc]
                if np.isclose(denom, 0.0, atol=1e-12):
                    denom = 1.0
                source = mean_err_opt
            elif mode_number == 1:
                denom = 1.0
                source = mean_err_opt
            elif mode_number == 2:
                denom = np.abs(glob_min[ifunc])
                if np.isclose(denom, 0.0, atol=1e-12):
                    denom = 1.0
                source = mean_err_opt
            elif mode_number == 3:
                if per_dim_ranges is None or mean_err_best is None:
                    raise ValueError("mode_number=3 requires per-dimension ranges and mean_err_best.")
                denom = per_dim_ranges[idim][ifunc]
                if np.isclose(denom, 0.0, atol=1e-12):
                    denom = 1.0
                source = mean_err_best
            elif mode_number == 4:
                if best_err_opt is None or best_mode4_denoms is None:
                    raise ValueError("mode_number=4 requires best_err_opt and denominators.")
                denom = best_mode4_denoms[idim][ifunc]
                if np.isclose(denom, 0.0, atol=1e-12):
                    denom = 1.0
                source = best_err_opt
            elif mode_number == 5:
                denom = np.abs(glob_min[ifunc])
                if np.isclose(denom, 0.0, atol=1e-12):
                    denom = 1.0
                raw_vals = {
                    algo: mean_err_opt[algo][idim][ifunc] / denom
                    for algo in mean_err_opt.keys()
                }
                min_val = min(raw_vals.values())
                max_val = max(raw_vals.values())
                span = max_val - min_val
                if np.isclose(span, 0.0, atol=1e-12):
                    for algo in raw_vals.keys():
                        se_dim[algo][idim] += 0.0
                    continue
                for algo, raw in raw_vals.items():
                    norm = (raw - min_val) / span
                    se[algo] += coeff * norm
                    se_dim[algo][idim] += norm
                continue
            elif mode_number == 8:
                denom = np.abs(glob_min[ifunc])
                if np.isclose(denom, 0.0, atol=1e-12):
                    denom = 1.0
                source = mean_err_opt
                for algo in source.keys():
                    raw = source[algo][idim][ifunc] / denom
                    transformed = raw / (1.0 + raw)
                    se[algo] += coeff * transformed
                    se_dim[algo][idim] += transformed
                continue
            elif mode_number == 6:
                if mean_err_best is None or best_abs_vals is None:
                    raise ValueError("mode_number=6 requires mean_err_best and best_abs_vals.")
                denom = np.abs(best_abs_vals[idim][ifunc])
                if np.isclose(denom, 0.0, atol=1e-12):
                    denom = 1.0
                source = mean_err_best
            elif mode_number == 7:
                if mean_err_best is None or best_abs_vals is None:
                    raise ValueError("mode_number=7 requires mean_err_best and best_abs_vals.")
                denom = np.abs(best_abs_vals[idim][ifunc])
                if np.isclose(denom, 0.0, atol=1e-12):
                    denom = 1.0
                raw_vals = {
                    algo: mean_err_best[algo][idim][ifunc] / denom
                    for algo in mean_err_best.keys()
                }
                min_val = min(raw_vals.values())
                max_val = max(raw_vals.values())
                span = max_val - min_val
                if np.isclose(span, 0.0, atol=1e-12):
                    for algo in raw_vals.keys():
                        se_dim[algo][idim] += 0.0
                    continue
                for algo, raw in raw_vals.items():
                    norm = (raw - min_val) / span
                    se[algo] += coeff * norm
                    se_dim[algo][idim] += norm
                continue
            else:  # mode_number == 9
                if mean_err_best is None or best_abs_vals is None:
                    raise ValueError("mode_number=9 requires mean_err_best and best_abs_vals.")
                denom = np.abs(best_abs_vals[idim][ifunc])
                if np.isclose(denom, 0.0, atol=1e-12):
                    denom = 1.0
                for algo in mean_err_best.keys():
                    raw = mean_err_best[algo][idim][ifunc] / denom
                    transformed = raw / (1.0 + raw)
                    se[algo] += coeff * transformed
                    se_dim[algo][idim] += transformed
                continue
            
            

            for algo in source.keys():
                contrib = coeff * (source[algo][idim][ifunc] / denom)
                se[algo] += contrib
                se_dim[algo][idim] += (source[algo][idim][ifunc] / denom)

    for algo in se.keys():
        se[algo] = se[algo] / count_funcs
        for idim in range(ndims):
           se_dim[algo][idim] = se_dim[algo][idim] / count_funcs

    return se, se_dim


def calc_SR(matrices, glob_min, weights, year, n_funcs, rank_mode="mean", drop_index=None):
    """
    Weighted Sum Rank (SR) computed with configurable run-level aggregation.
    Returns both totals and per-dimension contributions (already weight-scaled).
    """
    sr = {algo: 0.0 for algo in matrices.keys()}
    ndims = len(weights)
    sr_dim = {algo: [0.0]*ndims for algo in matrices.keys()}
    algos = list(matrices.keys())

    if rank_mode == "ranksum" : 
        FR_per_dim = calc_friedman_per_dim(matrices, year, n_funcs, drop_index=drop_index)
        for idim in range(ndims):
            coeff = weights[idim]
            for algo in algos:
                sr_dim[algo][idim] = FR_per_dim[algo][idim]
                sr[algo] += coeff * FR_per_dim[algo][idim]
        return sr, sr_dim
    else : 
        for idim in range(ndims):
            coeff = weights[idim]
            count_funcs = 0
            for ifunc in valid_funcs_iter(year, n_funcs, drop_index=drop_index):
                arrDict = {
                    algo: matrices[algo][idim][:, ifunc]
                    for algo in matrices.keys()
                }

                arrSize0 = None
                for algo, arr in arrDict.items(): 
                    arrSize = arr.shape[0] 
                    if arrSize0 is None : 
                        arrSize0 = arrSize
                    else : 
                        if arrSize != arrSize0 : 
                            raise Exception("Inconsistent run counts among algorithms")

                #print(arrSize0)

                ranks= calcRank(arrDict, glob_min[ifunc], mode=rank_mode)
                #print(ranks)
                #print(ranks)
                for algo, r in ranks.items():
                    contrib = coeff * r
                    sr[algo] += contrib
                    sr_dim[algo][idim] += r

                count_funcs += 1

        for algo in algos:
            sr[algo]= sr[algo]/count_funcs
            for idim in range(ndims):
                sr_dim[algo][idim] = sr_dim[algo][idim]/count_funcs
        #print(sr)
        #print(sr_dim)
        #print()
        return sr, sr_dim



def calc_friedman_per_dim(matrices, year, n_funcs,drop_index=None):
    """
    Friedman-style mean rank per DIMENSION (unweighted):

    For each dimension idim:
      For each function ifunc:
        - We have N_runs measurements for each algorithm.
        - For each run j, rank the algorithms (1 = best) on that run of that function.
        - Average those per-run ranks to get a mean rank per algorithm for that function.
      Then:
        - Average these function-level mean ranks over all functions in this dimension.

    Returns:
      FR_dim[algo][idim] = average rank on that dim (≈ in range [1, n_algos]).
    """
    algos = list(matrices.keys())
    ndims = len(matrices[algos[0]])
    FR_dim = {algo: [0.0] * ndims for algo in algos}

    for idim in range(ndims):
        ranks_accum = {algo: 0.0 for algo in algos}
        count_funcs = 0

        for ifunc in valid_funcs_iter(year, n_funcs, drop_index=drop_index):
            # Build result matrix for this (dim, func): shape (n_algos, N_runs)
            # matrices[algo][idim] has shape (N_runs, n_funcs), so [:, ifunc] is (N_runs,)
            results = []
            for algo in algos:
                arr = matrices[algo][idim][:, ifunc]
                arr = np.array(arr, dtype=float)

                # Optional: adjust w.r.t. glob_min if needed
                # arr = np.abs(arr - glob_min)   # uncomment if you want distance to optimum

                results.append(arr)

            results = np.array(results, dtype=float)  # shape: (n_algos, N_runs)

            # Check equal number of runs
            arrSize0 = results.shape[1]
            if not np.all(results.shape[1] == arrSize0):
                raise Exception("Inconsistent run counts among algorithms")

            # Rank per run (per column): ranks in [1..n_algos] for each run
            # np.apply_along_axis applies rankdata over axis=0 (the algorithms dimension)
            per_run_ranks = np.apply_along_axis(rankdata, 0, results)
            #print(per_run_ranks.shape)

            # Mean rank per algorithm for this function (average over runs)
            mean_ranks_for_func = np.mean(per_run_ranks, axis=1)  # shape (n_algos,)

            # Accumulate these per-function mean ranks
            for algo_idx, algo in enumerate(algos):
                ranks_accum[algo] += mean_ranks_for_func[algo_idx]

            count_funcs += 1

        # Average over functions for this dimension
        for algo in algos:
            FR_dim[algo][idim] = ranks_accum[algo] / count_funcs

    return FR_dim


def calc_head2head_per_dim(matrices, glob_min, year, n_funcs, ref_algo, alpha=0.05, mode="mean", drop_index=None):
    """
    Head-to-head per dimension (unweighted) in the direction ref_algo vs others.

    Rather than comparing averaged errors, this runs Mann–Whitney on the
    run-level absolute errors for every (dim, func).
    """
    opponents = [algo for algo in matrices.keys() if algo != ref_algo]
    ndims = len(matrices[ref_algo])

    h2h_dim = {
        algo: {
            "wins":  [0.0]*ndims,
            "ties":  [0.0]*ndims,
            "loses": [0.0]*ndims,
        }
        for algo in opponents
    }

    for idim in range(ndims):
        for ifunc in valid_funcs_iter(year, n_funcs, drop_index=drop_index):
            ref_samples = abs_error_samples(matrices, ref_algo, idim, ifunc, glob_min[ifunc])
            for algo in opponents:
                algo_samples = abs_error_samples(matrices, algo, idim, ifunc, glob_min[ifunc])
                result = MWUT(ref_samples, algo_samples, alpha=alpha, mode=mode)
                if result == 1:
                    h2h_dim[algo]["wins"][idim] += 1.0
                elif result == -1:
                    h2h_dim[algo]["loses"][idim] += 1.0
                else:
                    h2h_dim[algo]["ties"][idim] += 1.0

    return h2h_dim




#################################################################
# MASTER
#################################################################

def evaluate_all(
    res_folder,
    year,
    algos,
    ref_algo,
    DIM_MAX_EVALS,
    rank_mode="ranksum",
    mwut_mode="ranksum",
    alpha=0.05,
    return_se_sr_per_dim=False,
    mode_number=1,
    drop_index=None,
):
    # load raw matrices and dimension labels
    matrices, dims_list = load_results(res_folder, algos, year, DIM_MAX_EVALS)

    # external info from your env
    n_funcs = yearToNfuncs[year]   # int
    official_glob_min = yearToMin[year]     # np.array length n_funcs (or None for inferred years)
    needs_true_opt = mode_number in (0, 1, 2, 4, 5, 8)
    if official_glob_min is None:
        if needs_true_opt:
            raise ValueError("Selected mode requires known global optima (unavailable for CEC2011).")
        glob_min = infer_glob_min_from_runs(matrices, n_funcs)
    else:
        glob_min = official_glob_min

    # precompute mean abs error (avg over runs) per algo / dim / func
    mean_err = compute_mean_abs_errors(matrices, glob_min, n_funcs)

    mean_err_best = None
    per_dim_ranges = None
    best_err_opt = None
    best_mode4_denoms = None
    best_abs_vals = None
    best_vals = None
    worst_vals = None
    mode0_denoms = None
    if mode_number in (0, 3, 6, 7, 9):
        best_vals, worst_vals = compute_best_worst_per_dim_func(matrices)
        if mode_number in (3, 6, 7, 9):
            mean_err_best = compute_mean_abs_errors_to_best(matrices, best_vals)
        if mode_number == 3:
            per_dim_ranges = [
                np.maximum(worst_vals[idim] - best_vals[idim], 0.0)
                for idim in range(len(best_vals))
            ]
        if mode_number in (6, 7, 9):
            best_abs_vals = [np.abs(best_vals[idim]) for idim in range(len(best_vals))]
        if mode_number == 0:
            mode0_denoms = [
                np.maximum(worst_vals[idim] - glob_min, 0.0)
                for idim in range(len(worst_vals))
            ]
    if mode_number == 4:
        algo_best_vals = compute_best_values_per_algo(matrices)
        best_err_opt = {algo: [] for algo in algos}
        ndims = len(dims_list)
        best_max_per_dim = []
        for idim in range(ndims):
            stacked = []
            for algo in algos:
                best_vals = algo_best_vals[algo][idim]
                best_err_opt[algo].append(np.abs(best_vals - glob_min))
                stacked.append(best_vals)
            stacked = np.array(stacked)
            best_max_per_dim.append(np.max(stacked, axis=0))
        best_mode4_denoms = [
            np.maximum(best_max_per_dim[idim] - glob_min, 0.0)
            for idim in range(len(best_max_per_dim))
        ]

    # weights are ONLY for SE and SR
    weights = dim_weights(len(dims_list), year=year)

    # --- weighted global summaries ---
    SE, SE_dim = calc_SE(
        mean_err,
        weights,
        year,
        n_funcs,
        glob_min,
        mode_number=mode_number,
        mean_err_best=mean_err_best,
        per_dim_ranges=per_dim_ranges,
        best_err_opt=best_err_opt,
        best_mode4_denoms=best_mode4_denoms,
        best_abs_vals=best_abs_vals,
        mode0_denoms=mode0_denoms,
        drop_index=drop_index,
    )
    SR, SR_dim = calc_SR(matrices, glob_min, weights, year, n_funcs, rank_mode=rank_mode, drop_index=drop_index)

    # --- per-dimension stats (UNWEIGHTED, per your requirement) ---
    FR_dim  = calc_friedman_per_dim(matrices,year, n_funcs, drop_index=drop_index)  # Friedman avg rank per dim
    opponents = [algo for algo in algos if algo != ref_algo]
    H2Hdim  = calc_head2head_per_dim(
        matrices, glob_min, year, n_funcs, ref_algo, alpha=alpha, mode=mwut_mode, drop_index=drop_index
    )  # MWUT wins/ties/loses per dim

    # Build per-dimension Friedman table
    FR_cols = {
        f"FR@{dims_list[idim]}D": {algo: FR_dim[algo][idim] for algo in algos}
        for idim in range(len(dims_list))
    }
    FR_per_dim_df = pd.DataFrame(FR_cols)

    # Build per-dimension U table (already per algo per dim)
    U_per_dim_df = pd.DataFrame()
    Z_per_dim_df = pd.DataFrame()

    # Build per-dimension head-to-head W/T/L table
    h2h_cols = {}
    for idim, dim_val in enumerate(dims_list):
        h2h_cols[f"wins@{dim_val}D_{ref_algo}_vs"]  = {algo: H2Hdim[algo]["wins"][idim]  for algo in opponents}
        h2h_cols[f"ties@{dim_val}D_{ref_algo}_vs"]  = {algo: H2Hdim[algo]["ties"][idim]  for algo in opponents}
        h2h_cols[f"loses@{dim_val}D_{ref_algo}_vs"] = {algo: H2Hdim[algo]["loses"][idim] for algo in opponents}
    h2h_per_dim_df = pd.DataFrame(h2h_cols)

    SE_per_dim_df = None
    SR_per_dim_df = None
    if return_se_sr_per_dim:
        SE_cols = {
            f"SE@{dims_list[idim]}D": {algo: SE_dim[algo][idim] for algo in algos}
            for idim in range(len(dims_list))
        }
        SR_cols = {
            f"SR@{dims_list[idim]}D": {algo: SR_dim[algo][idim] for algo in algos}
            for idim in range(len(dims_list))
        }
        SE_per_dim_df = pd.DataFrame(SE_cols)
        SR_per_dim_df = pd.DataFrame(SR_cols)

    # final SE/SR-based scoring (your original logic)
    SE_min = np.min(list(SE.values()))
    SR_min = np.min(list(SR.values()))
    score1 = {algo: 50.0 * SE_min / SE[algo] for algo in algos}
    score2 = {algo: 50.0 * SR_min / SR[algo] for algo in algos}
    final_score = {algo: score1[algo] + score2[algo] for algo in algos}

    summary_df = pd.DataFrame({
        "SE_weighted":  pd.Series(SE),
        "SR_weighted":  pd.Series(SR),
        "Score1(SE)":   pd.Series(score1),
        "Score2(SR)":   pd.Series(score2),
        "FinalScore":   pd.Series(final_score),
    }).sort_values("FinalScore", ascending=False)

    # return all the per-dimension views users care about
    # FR_per_dim_df: Friedman avg rank per dim
    # U_per_dim_df:  U wins per dim
    # Z_per_dim_df:  summed Mann–Whitney dominance vs ref per dim
    # h2h_per_dim_df: wins/ties/loses vs ref per dim
    if return_se_sr_per_dim:
        return (
            summary_df,
            FR_per_dim_df,
            h2h_per_dim_df,
            SE_per_dim_df,
            SR_per_dim_df,
        )
    return summary_df, FR_per_dim_df, h2h_per_dim_df


def run_report(
    res_folder,
    year,
    algos,
    ref_algo,
    DIM_MAX_EVALS,
    rank_mode="ranksum",
    mwut_mode="ranksum",
    alpha=0.05,
    mode_number=1,
    drop_index=None,
    algo_display_map=None,
):
    """
    Convenience wrapper that mirrors the typical notebook usage pattern.
    """
    n_funcs = yearToNfuncs[year]
    valid_problem_count = sum(1 for _ in valid_funcs_iter(year, n_funcs, drop_index=drop_index))

    (
        summary_df,
        FR_per_dim_df,
        h2h_per_dim_df,
        SE_per_dim_df,
        SR_per_dim_df,
    ) = evaluate_all(
        res_folder,
        year,
        algos,
        ref_algo,
        DIM_MAX_EVALS,
        rank_mode=rank_mode,
        mwut_mode=mwut_mode,
        alpha=alpha,
        return_se_sr_per_dim=True,
        mode_number=mode_number,
        drop_index=drop_index,
    )

    print("=== Weighted summary (SE, SR, scores) ===")
    print(summary_df)
    print()

    print("=== Friedman average rank per dimension ===")
    print(FR_per_dim_df)
    print()
    
    if SR_per_dim_df is not None:
        print("=== SR per dimension ===")
        print(SR_per_dim_df)
        print()

    if SE_per_dim_df is not None:
        print("=== SE per dimension ===")
        print(SE_per_dim_df)
        print()

    print(f"=== Head-to-head per dimension ({ref_algo} vs others) ===")
    dims_order = list(DIM_MAX_EVALS.keys())
    if not h2h_per_dim_df.empty:
        for opponent in h2h_per_dim_df.index:
            for dim in dims_order:
                wins_col = f"wins@{dim}D_{ref_algo}_vs"
                ties_col = f"ties@{dim}D_{ref_algo}_vs"
                loses_col = f"loses@{dim}D_{ref_algo}_vs"
                if wins_col not in h2h_per_dim_df.columns:
                    continue
                wins = h2h_per_dim_df.at[opponent, wins_col]
                ties = h2h_per_dim_df.at[opponent, ties_col]
                loses = h2h_per_dim_df.at[opponent, loses_col]
                score = wins - loses
                print(f"{ref_algo} vs {opponent} @ {dim}D : {wins}/{ties}/{loses}  score={score}")
            print()
    else:
        print("  (no opponents to compare)")

    if SE_per_dim_df is not None and SR_per_dim_df is not None:
        latex_table = generate_latex_table(
            summary_df,
            SE_per_dim_df,
            SR_per_dim_df,
            h2h_per_dim_df,
            dims_order,
            ref_algo,
            algo_display_map=algo_display_map,
            n_valid_funcs=valid_problem_count,
        )
        tex_path = os.path.join(res_folder, "table.tex")
        with open(tex_path, "w", encoding="utf-8") as tex_file:
            tex_file.write(latex_table)
        print(f"LaTeX table saved to {tex_path}")

    return summary_df, FR_per_dim_df, h2h_per_dim_df


def generate_latex_table(
    summary_df,
    se_per_dim_df,
    sr_per_dim_df,
    h2h_per_dim_df,
    dims,
    ref_algo,
    float_format="{:.3f}",
    caption=None,
    label=None,
    algo_display_map=None,
    n_valid_funcs=None,
    use_booktabs=False,
    max_dims_per_table=2,
):
    """
    Build a LaTeX table that summarizes SE, SR, the combined S score, and the
    ArrDE head-to-head results per dimension.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Output of evaluate_all(...)[0]. Must contain a "FinalScore" column.
    se_per_dim_df, sr_per_dim_df : pandas.DataFrame
        DataFrames returned by evaluate_all(..., return_se_sr_per_dim=True)
        containing columns named like "SE@10D".
    h2h_per_dim_df : pandas.DataFrame
        Head-to-head table returned by evaluate_all(...)[2]. Can be empty.
    dims : Sequence[int]
        Ordered list of dimensions (e.g., [10, 30, 50]).
    ref_algo : str
        Name of the reference algorithm (ArrDE) used in head-to-head columns.
    float_format : str or Callable[[float], str], optional
        How numeric entries are formatted. Defaults to "{:.3f}".
    caption, label : str, optional
        Optional LaTeX caption/label strings.
    algo_display_map : dict, optional
        Mapping from internal algo tokens to display names.
    n_valid_funcs : int, optional
        Number of valid problems per dimension (used to show ArrDE vs ArrDE W/T/L).
    use_booktabs : bool, optional
        When True, emits \\toprule/\\midrule/\\bottomrule (requires \\usepackage{booktabs}).
    max_dims_per_table : int, optional
        Maximum number of dimension blocks per table before splitting (default: 2).
    """
    if se_per_dim_df is None or sr_per_dim_df is None:
        raise ValueError("SE/SR per-dimension DataFrames are required to build the table.")
    if "FinalScore" not in summary_df.columns:
        raise ValueError("summary_df must contain a 'FinalScore' column.")

    def _escape(value):
        text = str(value)
        replacements = {
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    if callable(float_format):
        format_number = float_format
    else:
        def format_number(val):
            return float_format.format(val)

    def score_ratio(min_val, value):
        if np.isclose(value, 0.0):
            return 1.0 if np.isclose(min_val, 0.0) else 0.0
        return float(min_val / value)

    dims = list(dims)
    algo_display_map = algo_display_map or {}
    if algo_display_map:
        ordered_algos = [algo for algo in algo_display_map.keys() if algo in summary_df.index]
        remaining = [algo for algo in summary_df.index if algo not in ordered_algos]
        algos = ordered_algos + remaining
    else:
        algos = list(summary_df.index)
    se_mins = {}
    sr_mins = {}
    se_values = {}
    sr_values = {}
    s_values = {}
    for dim in dims:
        se_col = f"SE@{dim}D"
        sr_col = f"SR@{dim}D"
        if se_col not in se_per_dim_df.columns:
            raise KeyError(f"Missing column '{se_col}' in SE per-dimension DataFrame.")
        if sr_col not in sr_per_dim_df.columns:
            raise KeyError(f"Missing column '{sr_col}' in SR per-dimension DataFrame.")
        se_values[dim] = {algo: se_per_dim_df.at[algo, se_col] for algo in algos}
        sr_values[dim] = {algo: sr_per_dim_df.at[algo, sr_col] for algo in algos}
        se_mins[dim] = min(se_values[dim].values())
        sr_mins[dim] = min(sr_values[dim].values())

    def wtl_value(algo, dim):
        if algo == ref_algo:
            if n_valid_funcs is None:
                return r"\textemdash"
            return f"0/{n_valid_funcs}/0"
        if h2h_per_dim_df is None or h2h_per_dim_df.empty:
            return r"\textemdash"
        wins_col = f"wins@{dim}D_{ref_algo}_vs"
        ties_col = f"ties@{dim}D_{ref_algo}_vs"
        loses_col = f"loses@{dim}D_{ref_algo}_vs"
        if wins_col not in h2h_per_dim_df.columns:
            return r"\textemdash"
        if algo not in h2h_per_dim_df.index:
            return r"\textemdash"
        wins = int(round(h2h_per_dim_df.at[algo, wins_col]))
        ties = int(round(h2h_per_dim_df.at[algo, ties_col]))
        loses = int(round(h2h_per_dim_df.at[algo, loses_col]))
        return f"{wins}/{ties}/{loses}"

    def compute_ranks(values_by_algo, higher_better):
        arr = np.array([values_by_algo[algo] for algo in algos], dtype=float)
        arr_rank = rankdata(-arr if higher_better else arr, method="average")
        return {algo: arr_rank[i] for i, algo in enumerate(algos)}

    def rank_to_str(rank):
        rounded = round(rank)
        if np.isclose(rank, rounded):
            return str(int(rounded))
        return f"{rank:.2f}"

    def format_metric(value, rank, best_value, higher_better):
        formatted = f"{format_number(value)} ({rank_to_str(rank)})"
        if higher_better:
            is_best = np.isclose(value, best_value)
        else:
            is_best = np.isclose(value, best_value)
        if is_best:
            return f"\\textbf{{{formatted}}}"
        return formatted

    se_ranks = {}
    sr_ranks = {}
    s_ranks = {}
    s_best = {}
    for dim in dims:
        se_ranks[dim] = compute_ranks(se_values[dim], higher_better=False)
        sr_ranks[dim] = compute_ranks(sr_values[dim], higher_better=False)
        s_values[dim] = {}
        for algo in algos:
            se_val = se_values[dim][algo]
            sr_val = sr_values[dim][algo]
            s_values[dim][algo] = 50.0 * score_ratio(se_mins[dim], se_val) + 50.0 * score_ratio(sr_mins[dim], sr_val)
        s_best[dim] = max(s_values[dim].values())
        s_ranks[dim] = compute_ranks(s_values[dim], higher_better=True)

    s_total_values = {algo: summary_df.at[algo, "FinalScore"] for algo in algos}
    s_total_best = max(s_total_values.values())
    s_total_ranks = compute_ranks(s_total_values, higher_better=True)

    rule_top = "\\toprule" if use_booktabs else "\\hline"
    rule_mid = "\\midrule" if use_booktabs else "\\hline"
    rule_bottom = "\\bottomrule" if use_booktabs else "\\hline"

    dim_chunks = [dims[i:i + max_dims_per_table] for i in range(0, len(dims), max_dims_per_table)] or [dims]
    tables = []

    def multicol_align(has_left, has_right):
        align = ""
        if has_left:
            align += "|"
        align += "c"
        if has_right:
            align += "|"
        return align

    for chunk_idx, chunk_dims in enumerate(dim_chunks):
        include_combined = (chunk_idx == len(dim_chunks) - 1)
        if chunk_dims:
            col_format = "l|" + "|".join(["cccc"] * len(chunk_dims))
        else:
            col_format = "l"
        if include_combined:
            col_format += "|cc"

        lines = [
            "\\begin{center}",
            "\\begin{table*}[ht]",
            "\\centering",
            "\\footnotesize",
            f"\\begin{{tabular}}{{{col_format}}}",
            rule_top,
        ]

        header = ["Algorithm"]
        for idx, dim in enumerate(chunk_dims):
            right_border = idx < len(chunk_dims) - 1 or include_combined
            align = multicol_align(True, right_border)
            header.append(f"\\multicolumn{{4}}{{{align}}}{{{dim}D}}")
        if include_combined:
            combined_align = multicol_align(True, False)
            header.append(f"\\multicolumn{{2}}{{{combined_align}}}{{Combined}}")
        lines.append(" & ".join(header) + r" \\")

        sub_header = [""]
        for _ in chunk_dims:
            sub_header.extend([r"$\mathcal{E}$", r"$R$", r"$S$", "W/T/L"])
        if include_combined:
            sub_header.extend([r"$S_{\text{tot}}$", "S-year"])
        lines.append(" & ".join(sub_header) + r" \\")
        lines.append(rule_mid)

        for algo in algos:
            row = [_escape(algo_display_map.get(algo, algo))]
            for dim in chunk_dims:
                se_val = se_values[dim][algo]
                sr_val = sr_values[dim][algo]
                s_score = s_values[dim][algo]
                row.append(format_metric(se_val, se_ranks[dim][algo], se_mins[dim], higher_better=False))
                row.append(format_metric(sr_val, sr_ranks[dim][algo], sr_mins[dim], higher_better=False))
                row.append(format_metric(s_score, s_ranks[dim][algo], s_best[dim], higher_better=True))
                row.append(wtl_value(algo, dim))
            if include_combined:
                s_total = s_total_values[algo]
                row.append(format_metric(s_total, s_total_ranks[algo], s_total_best, higher_better=True))
                row.append("")
            lines.append(" & ".join(row) + r" \\")

        lines.append(rule_bottom)
        lines.append("\\end{tabular}")
        if caption and include_combined:
            lines.append(f"\\caption{{{_escape(caption)}}}")
        if label and include_combined:
            lines.append(f"\\label{{{label}}}")
        lines.append("\\end{table*}")
        lines.append("\\end{center}")

        tables.append("\n".join(lines))

    return "\n\n".join(tables)


def analyze_results(
    res_folder,
    alg_map,
    ref_key,
    year,
    maxevals=300000,
    n_dim=30,
    n_problems=30,
    delimiter="\t",
    drop_index=1,
    alpha=0.05,
    mwut_mode="mean",
    mode_number=1,
):
    """
    Notebook-friendly analysis helper mirroring the legacy cec_comp implementation.

    mode_number interpretations:
      0 => |f - f*| / (f_max - f*)
      1 => |f - f*|
      2 => |f - f*| / |f*|
      3 => |f - f_best| / (f_worst - f_best)
      4 => |f(x_best) - f*| / (max_algo f(x_best) - f*)
      5 => normalized mode 2 per function
      6 => |f - f_best| / |f_best|
      7 => normalized mode 6 per function
      8 => (|f - f*| / |f*|) / (1 + |f - f*| / |f*|)
      9 => (|f - f_best| / |f_best|) / (1 + |f - f_best| / |f_best|)
    """
    if mode_number not in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
        raise ValueError("mode_number must be between 0 and 9.")

    loaded = {}
    for display_name, token in alg_map.items():
        fname = _resolve_result_file(res_folder, year, token, n_dim, maxevals)
        loaded[display_name] = np.loadtxt(fname, delimiter=delimiter)

    if ref_key not in alg_map:
        raise KeyError(f"Reference key '{ref_key}' must be one of: {list(alg_map.keys())}")

    goptimum_cec22 = np.array([300, 400, 600, 800, 900, 1800, 2000, 2200, 2300, 2400, 2600, 2700])
    goptimum_cec20 = np.array([100, 1100, 700, 1900, 1700, 1600, 2100, 2200, 2400, 2500])
    goptimum_cec17 = np.array([100 * i for i in range(1, 31)])
    goptimum_cec14 = goptimum_cec17

    year_to_geopt = {
        2011: None,
        2014: goptimum_cec14,
        2017: goptimum_cec17,
        2019: np.ones(10),
        2020: goptimum_cec20,
        2022: goptimum_cec22,
    }
    if year not in year_to_geopt:
        raise ValueError(f"Unsupported CEC year: {year}")
    gopt = year_to_geopt[year]
    needs_true_opt = mode_number in (0, 1, 2, 4, 5, 8)
    if gopt is None and needs_true_opt:
        raise ValueError("Selected mode requires known global optima (unavailable for this year).")
    if gopt is not None:
        gopt = np.array(gopt, dtype=float)

    sample_shape = next(iter(loaded.values())).shape[1]
    if sample_shape != n_problems:
        raise ValueError(f"n_problems ({n_problems}) does not match results columns ({sample_shape}).")

    denom_opt = None
    if mode_number in (2, 5, 8):
        denom_opt = np.abs(gopt)
        denom_opt = np.where(np.isclose(denom_opt, 0.0, atol=1e-12), 1.0, denom_opt)

    per_func_best = None
    per_func_worst = None
    denom_range = None
    denom_best = None
    denom_mode0 = None
    if mode_number in (0, 3, 6, 7, 9):
        per_func_best = np.full(n_problems, np.inf)
        per_func_worst = np.full(n_problems, -np.inf)
        for arr in loaded.values():
            per_func_best = np.minimum(per_func_best, np.min(arr, axis=0))
            per_func_worst = np.maximum(per_func_worst, np.max(arr, axis=0))
    if mode_number == 0:
        denom_mode0 = per_func_worst - gopt
        denom_mode0 = np.where(np.isclose(denom_mode0, 0.0, atol=1e-12), 1.0, denom_mode0)
    if mode_number == 3:
        denom_range = per_func_worst - per_func_best
        denom_range = np.where(np.isclose(denom_range, 0.0, atol=1e-12), 1.0, denom_range)
    if mode_number in (6, 7, 9):
        denom_best = np.abs(per_func_best)
        denom_best = np.where(np.isclose(denom_best, 0.0, atol=1e-12), 1.0, denom_best)

    best_mode4_values = None
    mode4_denoms = None
    if mode_number == 4:
        best_mode4_values = {name: np.min(arr, axis=0) for name, arr in loaded.items()}
        stacked = np.stack(list(best_mode4_values.values()))
        per_func_max_best = np.max(stacked, axis=0)
        mode4_denoms = per_func_max_best - gopt
        mode4_denoms = np.where(np.isclose(mode4_denoms, 0.0, atol=1e-12), 1.0, mode4_denoms)

    if gopt is not None:
        print(gopt)
    else:
        print("Global optimum unavailable for this year (CEC2011).")

    def _error_matrix(arr):
        if mode_number == 0:
            return np.abs(arr - gopt) / denom_mode0
        if mode_number == 1:
            return np.abs(arr - gopt)
        if mode_number == 2:
            return np.abs(arr - gopt) / denom_opt
        if mode_number == 5:
            return np.abs(arr - gopt) / denom_opt
        if mode_number == 8:
            return np.abs(arr - gopt) / denom_opt
        if mode_number == 3:
            return np.abs(arr - per_func_best) / denom_range
        if mode_number in (6, 7):
            return np.abs(arr - per_func_best) / denom_best
        if mode_number == 9:
            return np.abs(arr - per_func_best) / denom_best
        raise ValueError("Unsupported mode for error matrix computation.")

    if mode_number == 4:
        error_mats = {}
        for name, arr in loaded.items():
            errs = np.abs(best_mode4_values[name] - gopt) / mode4_denoms
            error_mats[name] = np.tile(errs, (arr.shape[0], 1))
    else:
        error_mats = {name: _error_matrix(arr) for name, arr in loaded.items()}

    mean_abs = {name: np.mean(error_mats[name], axis=0) for name in loaded.keys()}

    if mode_number in (8, 9):
        for name in mean_abs.keys():
            mean_abs[name] = mean_abs[name] / (1.0 + mean_abs[name])

    if mode_number in (5, 7):
        stacked_means = np.vstack([mean_abs[name] for name in loaded.keys()])
        min_vals = np.min(stacked_means, axis=0)
        max_vals = np.max(stacked_means, axis=0)
        span = max_vals - min_vals
        span_safe = np.where(np.isclose(span, 0.0, atol=1e-12), 1.0, span)
        for name in mean_abs.keys():
            mean_abs[name] = np.where(
                np.isclose(span, 0.0, atol=1e-12),
                0.0,
                (mean_abs[name] - min_vals) / span_safe,
            )

    df_means = pd.DataFrame(mean_abs)
    if drop_index is not None and drop_index in df_means.index:
        df_means = df_means.drop(index=drop_index)

    metric_label = {
        0: "|f - f*| / (f_max - f*)",
        1: "absolute |f - f*|",
        2: "relative |f - f*| / |f*|",
        3: "relative |f - f_best| / (f_worst - f_best)",
        4: "relative |f(x_best) - f*| / (max_algo f(x_best) - f*)",
        5: "relative |f - f*| / |f*| (normalized per function)",
        6: "relative |f - f_best| / |f_best|",
        7: "relative |f - f_best| / |f_best| (normalized per function)",
        8: "(|f - f*| / |f*|) / (1 + |f - f*| / |f*|)",
        9: "(|f - f_best| / |f_best|) / (1 + |f - f_best| / |f_best|)",
    }[mode_number]
    print(f"=== Mean {metric_label} errors ({year}) ===")
    if mode_number in (5, 7):
        print("Note: Normalization is applied per function across algorithms.")
    display(df_means)
    print("\nPer-algorithm overall mean:")
    for c in df_means.columns:
        print(f"{c:15s} : {np.mean(df_means[c]):.6f}")

    print(f"\n------------\nRunning MWUT comparisons (Reference vs Algorithm): {ref_key}")
    ref_arr = error_mats[ref_key]
    mwut_results, counts = {}, {}

    for name, arr in error_mats.items():
        if name == ref_key:
            continue
        outcomes = []
        wins = ties = loses = 0
        for i in range(n_problems):
            if i == drop_index:
                continue
            out = MWUT(ref_arr[:, i], arr[:, i], alpha=alpha, mode=mwut_mode)
            outcomes.append(out)
            if out == 1:
                wins += 1
            elif out == -1:
                loses += 1
            else:
                ties += 1
        mwut_results[name] = np.array(outcomes)
        counts[name] = {"wins": wins, "ties": ties, "loses": loses}
        print(f"{ref_key:15s} vs {name:15s}  -->  W/T/L = {wins}/{ties}/{loses}")

    df_mwut = pd.DataFrame(mwut_results, index=df_means.index)
    print("\n=== MWUT per-problem summary (Reference vs Algorithm) ===")
    display(df_mwut)

    return {
        "df_mean_abs": df_means,
        "df_mwut": df_mwut,
        "counts": counts,
        "loaded": loaded,
    }


#################################################################
# MULTIPLIER SCORING HELPERS
#################################################################

def _resolve_display_name(algo, display_map):
    if not display_map:
        return algo
    for key, value in display_map.items():
        if key.lower() == str(algo).lower():
            return value
    return display_map.get(algo, algo)


def scaled_dim_maxevals(year, multiplier, base_dim_template=None):
    """
    Helper to scale dimension evaluation budgets by a multiplier.

    Parameters
    ----------
    year : int
        CEC year identifier.
    multiplier : float
        Scalar applied to each base evaluation budget.
    base_dim_template : dict[int, float], optional
        Mapping dim -> base value to scale. If omitted, defaults to
        `default_dim_to_year(multiplier=1)[year]`.
    """
    if multiplier is None:
        raise ValueError("multiplier must be provided.")
    if base_dim_template is None:
        base_map = default_dim_to_year(multiplier=1)
        if year not in base_map:
            raise ValueError("Provide base_dim_template for unsupported year.")
        base_dim_template = base_map[year]
    return {dim: int(round(val * multiplier)) for dim, val in base_dim_template.items()}


def compute_scores_vs_multiplier(
    res_folder,
    year,
    algos,
    ref_algo,
    multipliers,
    base_dim_template=None,
    rank_mode="ranksum",
    mwut_mode="ranksum",
    alpha=0.05,
    mode_number=1,
    drop_index=None,
    algo_display_map=None,
):
    """
    Evaluate SE/SR/Final scores for each multiplier value.

    Returns a tidy DataFrame with columns:
        Multiplier, Algorithm, SE_weighted, SR_weighted, Score1(SE),
        Score2(SR), FinalScore
    """
    if not multipliers:
        raise ValueError("multipliers must contain at least one value.")

    records = []
    for multiplier in multipliers:
        dim_max_evals = scaled_dim_maxevals(year, multiplier, base_dim_template=base_dim_template)
        summary_df, _, _, _, _ = evaluate_all(
            res_folder,
            year,
            algos,
            ref_algo,
            dim_max_evals,
            rank_mode=rank_mode,
            mwut_mode=mwut_mode,
            alpha=alpha,
            return_se_sr_per_dim=True,
            mode_number=mode_number,
            drop_index=drop_index,
        )
        for algo, row in summary_df.iterrows():
            display_name = _resolve_display_name(algo, algo_display_map)
            records.append(
                {
                    "Multiplier": multiplier,
                    "Algorithm": algo,
                    "DisplayName": display_name,
                    "SE_weighted": row["SE_weighted"],
                    "SR_weighted": row["SR_weighted"],
                    "Score1(SE)": row["Score1(SE)"],
                    "Score2(SR)": row["Score2(SR)"],
                    "FinalScore": row["FinalScore"],
                }
            )

    df = pd.DataFrame(records)
    return df


def plot_score_trends(
    score_df,
    metrics,
    xlabel=r"$N_{max}/D$",
    figsize=(6, 4),
    legend_labels=None,
    save_as=None,
):
    """
    Plot metric trends over multipliers with consistent styling.
    """
    metrics = list(metrics)
    if not metrics:
        raise ValueError("metrics must contain at least one column name.")

    name_col = "DisplayName" if "DisplayName" in score_df.columns else "Algorithm"
    unique_algos = list(dict.fromkeys(score_df[name_col]))
    if legend_labels:
        desired_order = []
        if name_col == "DisplayName":
            candidates = {label.lower(): label for label in unique_algos}
            for label in legend_labels.values():
                if label is None:
                    continue
                found = candidates.get(str(label).lower())
                if found and found not in desired_order:
                    desired_order.append(found)
        else:
            algo_lower = {algo.lower(): algo for algo in unique_algos}
            for key in legend_labels.keys():
                found = algo_lower.get(str(key).lower())
                if found and found not in desired_order:
                    desired_order.append(found)
        desired_order += [name for name in unique_algos if name not in desired_order]
        unique_algos = desired_order
    colors = plt.cm.get_cmap("tab10", len(unique_algos))

    fig_height = figsize[1] * len(metrics)
    fig, axes = plt.subplots(len(metrics), 1, figsize=(figsize[0], fig_height), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for i, label in enumerate(unique_algos):
            algo_df = score_df[score_df[name_col] == label].sort_values("Multiplier")
            ax.plot(
                algo_df["Multiplier"],
                algo_df[metric],
                label=label,
                marker="o",
                linewidth=2,
                color=colors(i),
            )
        y_label = r"$S$" if metric == "FinalScore" else metric
        ax.set_ylabel(y_label)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel(xlabel)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="best")
    plt.tight_layout()
    if save_as:
        fig.savefig(save_as, dpi=300, bbox_inches="tight")
    return axes


def correct_digits(value, optimum=1.0, max_digits=10):
    """Return number of correct digits (0..max_digits) for a given value."""
    err = abs(value - optimum)

    if err == 0.0:
        return max_digits
    if err >= 1.0:
        return 0

    d = -math.log10(err)
    nd = int(math.floor(d))
    nd = max(0, min(nd, max_digits))
    return nd


def compute_cec100_scores(filename, optima=None, best_k=25, max_digits=10):
    """
    Compute per-function and total CEC 2019 100-digit scores, and produce the filling table.
    """
    data = np.loadtxt(filename)
    n_runs, n_funcs = data.shape

    if optima is None:
        optima = np.ones(n_funcs)
    else:
        optima = np.asarray(optima)
        if optima.shape != (n_funcs,):
            raise ValueError("optima must have one value per function (column).")

    digits = np.zeros_like(data, dtype=int)

    # Convert each run result to correct-digit counts
    for j in range(n_funcs):
        for i in range(n_runs):
            digits[i, j] = correct_digits(data[i, j], optimum=float(optima[j]), max_digits=max_digits)

    # Build filling table (count of runs per digit)
    filling_table = pd.DataFrame(
        0, index=[f"F{j+1}" for j in range(n_funcs)],
        columns=[str(k) for k in range(max_digits + 1)]
    )

    for j in range(n_funcs):
        vals, counts = np.unique(digits[:, j], return_counts=True)
        for v, c in zip(vals, counts):
            filling_table.loc[f"F{j+1}", str(v)] = c

    # Compute per-function scores
    scores = []
    for j in range(n_funcs):
        sorted_digits = sorted(digits[:, j], reverse=True)
        best25 = sorted_digits[:best_k]
        score = np.mean(best25)
        scores.append(score)

    total_score = sum(scores)
    filling_table["Score"] = [round(s, 2) for s in scores]

    return filling_table, scores, total_score
