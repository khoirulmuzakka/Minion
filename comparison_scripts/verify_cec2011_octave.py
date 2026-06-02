#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import glob
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MINIONPY_LIB = os.path.join(REPO_ROOT, "minionpy", "lib")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if MINIONPY_LIB not in sys.path:
    sys.path.insert(0, MINIONPY_LIB)


def _cec2011_problem02_bounds(ndim: int):
    lb = [0.0] * ndim
    for idx in range(3, ndim):
        lb[idx] = -4.0 - 0.25 * int((idx - 4) / 3)

    ub = [0.0] * ndim
    ub[0] = ub[1] = 4.0
    ub[2] = math.pi
    for idx in range(3, ndim):
        ub[idx] = 4.0 + 0.25 * int((idx - 4) / 3)
    return lb, ub


def _cec2011_problem05_bounds(ndim: int):
    lb = [-1.0] * ndim
    lb[:3] = [0.0, 0.0, 0.0]
    ub = [0.0] * ndim
    ub[0] = ub[1] = 4.0
    ub[2] = math.pi
    for idx in range(3, ndim):
        ub[idx] = 4.0 + 0.25 * int((idx - 4) / 3)
    return lb, ub


def _cec2011_problem06_bounds(ndim: int):
    lb = [-1.0] * ndim
    lb[:3] = [0.0, 0.0, 0.0]
    ub = [0.0] * ndim
    ub[0] = ub[1] = 4.0
    ub[2] = math.pi
    for idx in range(3, ndim, 3):
        ub[idx] = 4.0 + 0.25 * int((1 - 4) / 3)
        if idx + 1 < ndim:
            ub[idx + 1] = 4.0 + 0.25 * int((2 - 4) / 3)
        if idx + 2 < ndim:
            ub[idx + 2] = 4.0 + 0.25 * int((3 - 4) / 3)
    return lb, ub


def _from_str(data: str):
    return [float(token) for token in data.split()]


CEC2011_METADATA = {
    1: (6, [-6.4] * 6, [6.35] * 6),
    2: (30, *_cec2011_problem02_bounds(30)),
    3: (1, [-0.6], [0.9]),
    4: (1, [0.0], [5.0]),
    5: (30, *_cec2011_problem05_bounds(30)),
    6: (30, *_cec2011_problem06_bounds(30)),
    7: (20, [0.0] * 20, [2.0 * math.pi] * 20),
    8: (7, [0.0] * 7, [15.0] * 7),
    9: (
        126,
        [0.0] * 126,
        _from_str(
            """
            0.217 0.024 0.076 0.892 0.128 0.25 0.058 0.112 0.062 0.082 0.035 0.09 0.032 0.095 0.022 0.175 0.032
            0.087 0.035 0.024 0.106 0.217 0.024 0.026 0.491 0.228 0.3 0.058 0.112 0.062 0.082 0.035 0.09 0.032
            0.095 0.022 0.175 0.032 0.087 0.035 0.024 0.106 0.216 0.024 0.076 0.216 0.216 0.216 0.058 0.112
            0.062 0.082 0.035 0.09 0.032 0.095 0.022 0.175 0.032 0.087 0.035 0.024 0.081 0.217 0.024 0.076
            0.228 0.228 0.228 0.058 0.112 0.062 0.082 0.035 0.09 0.032 0.095 0.022 0.025 0.032 0.087 0.035
            0.024 0.081 0.124 0.024 0.076 0.124 0.124 0.124 0.058 0.112 0.062 0.082 0.035 0.065 0.032 0.095
            0.022 0.124 0.032 0.087 0.035 0.024 0.106 0.116 0.024 0.076 0.116 0.116 0.116 0.058 0.087 0.062
            0.082 0.035 0.09 0.032 0.095 0.022 0.116 0.032 0.087 0.035 0.024 0.106
            """
        ),
    ),
    10: (12, [0.2] * 6 + [-180.0] * 6, [1.0] * 6 + [180.0] * 6),
    11: (120, [10, 20, 30, 40, 50] * 24, [75, 125, 175, 250, 300] * 24),
    12: (240, [150, 135, 73, 60, 73, 57, 20, 47, 20, 55] * 24, [470, 460, 340, 300, 243, 160, 130, 120, 80, 55.1] * 24),
    13: (6, [100, 50, 80, 50, 50, 50], [500, 200, 300, 150, 200, 120]),
    14: (13, [0, 0, 0, 60, 60, 60, 60, 60, 60, 40, 40, 55, 55], [680, 360, 360, 180, 180, 180, 180, 180, 180, 120, 120, 120, 120]),
    15: (15, [150, 150, 20, 20, 150, 135, 135, 60, 25, 25, 20, 20, 25, 15, 15], [455, 455, 130, 130, 470, 460, 465, 300, 162, 160, 80, 80, 85, 55, 55]),
    16: (
        40,
        _from_str("36 36 60 80 47 68 110 135 135 130 94 94 125 125 125 125 220 220 242 242 254 254 254 254 254 254 10 10 10 47 60 60 60 90 90 90 25 25 25 242"),
        _from_str("114 114 120 190 97 140 300 300 300 300 375 375 500 500 500 500 500 500 550 550 550 550 550 550 550 550 150 150 150 97 190 190 190 200 200 200 110 110 110 550"),
    ),
    17: (
        140,
        _from_str(
            "71 120 125 125 90 90 280 280 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 260 280 280 280 280 260 260 260 260 260 260 260 260 120 120 423 423 3 3 160 160 160 160 160 160 160 160 165 165 165 165 180 180 103 198 100 153 163 95 160 160 196 196 196 196 130 130 137 137 195 175 175 175 175 330 160 160 200 56 115 115 115 207 207 175 175 175 175 360 415 795 795 578 615 612 612 758 755 750 750 713 718 791 786 795 795 795 795 94 94 94 244 244 244 95 95 116 175 2 4 15 9 12 10 112 4 5 5 50 5 42 42 41 17 7 7 26"
        ),
        _from_str(
            "119 189 190 190 190 190 490 490 496 496 496 496 506 509 506 505 506 506 505 505 505 505 505 505 537 537 549 549 501 501 506 506 506 506 500 500 241 241 774 769 19 28 250 250 250 250 250 250 250 250 504 504 504 504 471 561 341 617 312 471 500 302 511 511 490 490 490 490 432 432 455 455 541 536 540 538 540 574 531 531 542 132 245 245 245 307 307 345 345 345 345 580 645 984 978 682 720 718 720 964 958 1007 1006 1013 1020 954 952 1006 1013 1021 1015 203 203 203 379 379 379 190 189 194 321 19 59 83 53 37 34 373 20 38 19 98 10 74 74 105 51 19 19 40"
        ),
    ),
    18: (96, [5, 6, 10, 13] * 24, [15, 15, 30, 25] * 24),
    19: (96, [5, 6, 10, 13] * 24, [15, 15, 30, 25] * 24),
    20: (96, [5, 6, 10, 13] * 24, [15, 15, 30, 25] * 24),
    21: (26, [1900, 2.5, 0, 0, 100, 100, 100, 100, 100, 100, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1.1, 1.1, 1.05, 1.05, 1.05, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi], [2200, 5.0, 1, 1, 500, 500, 500, 500, 500, 600, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 6, 6, 6, 6, 6, math.pi, math.pi, math.pi, math.pi, math.pi]),
    22: (22, [-1000, 3, 0, 0, 100, 100, 30, 400, 800, 0.01, 0.01, 0.01, 0.01, 0.01, 1.05, 1.05, 1.15, 1.7, -math.pi, -math.pi, -math.pi, -math.pi], [0, 5, 1, 1, 400, 500, 300, 1600, 2200, 0.9, 0.9, 0.9, 0.9, 0.9, 6, 6, 6.5, 291, math.pi, math.pi, math.pi, math.pi]),
}


def get_tolerance(problem: int):
    return (0.0, 1e-9)


def _build_cpp_evaluator(problem: int):
    try:
        import minionpy

        return minionpy.CEC2011Functions(problem)
    except Exception:
        import minionpycpp

        dimension, _, _ = CEC2011_METADATA[problem]
        return minionpycpp.CEC2011Functions(problem, dimension)


def _octave_env():
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("SNAP"):
            env.pop(key, None)
    return env


def _candidate_octave_paths():
    candidates = []

    for env_key in ("OCTAVE_CLI", "OCTAVE_BIN", "OCTAVE_EXECUTABLE"):
        value = os.environ.get(env_key)
        if value:
            candidates.append(value)

    for name in ("octave-cli", "octave", "octave-cli.exe", "octave.exe"):
        resolved = shutil.which(name)
        if resolved:
            candidates.append(resolved)

    if os.name == "nt":
        roots = []
        for env_key in ("ProgramFiles", "ProgramFiles(x86)", "LocalAppData"):
            root = os.environ.get(env_key)
            if root:
                roots.append(root)

        patterns = [
            os.path.join("GNU Octave", "Octave-*", "mingw64", "bin", "octave-cli.exe"),
            os.path.join("GNU Octave", "Octave-*", "mingw64", "bin", "octave.exe"),
            os.path.join("Octave", "Octave-*", "mingw64", "bin", "octave-cli.exe"),
            os.path.join("Octave", "Octave-*", "mingw64", "bin", "octave.exe"),
        ]
        for root in roots:
            for pattern in patterns:
                candidates.extend(glob.glob(os.path.join(root, pattern)))

    deduped = []
    seen = set()
    for candidate in candidates:
        normalized = os.path.normcase(os.path.abspath(candidate))
        if normalized not in seen and os.path.exists(candidate):
            deduped.append(candidate)
            seen.add(normalized)
    return deduped


def resolve_octave_binary(explicit_path: str | None = None):
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    candidates.extend(_candidate_octave_paths())

    for candidate in candidates:
        resolved = shutil.which(candidate) if not os.path.isabs(candidate) else candidate
        if resolved and os.path.exists(resolved):
            return resolved

    searched = ", ".join(candidates) if candidates else "PATH and standard install locations"
    raise RuntimeError(
        "Unable to find an Octave executable. "
        "Install GNU Octave, add it to PATH, or pass --octave-bin / set OCTAVE_CLI. "
        f"Searched: {searched}"
    )


def _octave_expression(problem: int, x_expr: str):
    if 1 <= problem <= 8:
        return f"bench_func({x_expr}, {problem})"
    if problem == 9:
        return f"cost_fn({x_expr})"
    if problem == 10:
        return f"antennafunccircular({x_expr}, [50,120], 180, 0.5)"
    if problem == 11:
        return f"fn_DED_5({x_expr})"
    if problem == 12:
        return f"fn_DED_10({x_expr})"
    if problem == 13:
        return f"fn_ELD_6({x_expr})"
    if problem == 14:
        return f"fn_ELD_13({x_expr})"
    if problem == 15:
        return f"fn_ELD_15({x_expr})"
    if problem == 16:
        return f"fn_ELD_40({x_expr})"
    if problem == 17:
        return f"fn_ELD_140({x_expr})"
    if problem == 18:
        return f"fn_HT_ELD_Case_1({x_expr})"
    if problem == 19:
        return f"fn_HT_ELD_Case_2({x_expr})"
    if problem == 20:
        return f"fn_HT_ELD_Case_3({x_expr})"
    if problem == 21:
        return f"messengerfull({x_expr}, MGADSMproblem)"
    if problem == 22:
        return f"cassini2({x_expr}, MGADSMproblem)"
    raise ValueError(f"Unsupported problem {problem}")


def _octave_setup(problem: int):
    if 1 <= problem <= 8:
        return "addpath('cec2011/Probs_1_to_8');"
    if problem == 9:
        return "addpath('cec2011/Prob_9_Transmission_Pricing');"
    if problem == 10:
        return "addpath('cec2011/Prob_10_Circ_Antenna/CEC_CircularAntenna');"
    if problem in (11, 12):
        return "addpath('cec2011/Probs_11_ELD_Package/DED Codes');"
    if problem in (13, 14, 15, 16, 17):
        return "addpath('cec2011/Probs_11_ELD_Package/ELD Codes');"
    if problem in (18, 19, 20):
        return "addpath('cec2011/Probs_11_ELD_Package/Hydrothermal Codes');"
    if problem == 21:
        return "addpath('cec2011/Probs_12_to_13_Package'); load('cec2011/Probs_12_to_13_Package/messengerfull.mat');"
    if problem == 22:
        return "addpath('cec2011/Probs_12_to_13_Package'); load('cec2011/Probs_12_to_13_Package/cassini2.mat');"
    raise ValueError(f"Unsupported problem {problem}")


def _format_matrix_for_octave(matrix: list[list[float]]):
    rows = []
    for row in matrix:
        rows.append(" ".join(f"{value:.17g}" for value in row))
    return "[ " + " ; ".join(rows) + " ]"


def evaluate_with_octave(problem: int, samples: list[list[float]], octave_bin: str | None = None):
    matrix_literal = _format_matrix_for_octave(samples)
    setup = _octave_setup(problem)
    expr = _octave_expression(problem, "X(i,:)")
    resolved_octave = resolve_octave_binary(octave_bin)
    script = f"""
    {setup}
    X = {matrix_literal};
    for i = 1:rows(X)
      val = {expr};
      fprintf('%.17g\\n', val);
    end
    """
    with tempfile.NamedTemporaryFile("w", suffix=".m", delete=False) as handle:
        handle.write(script)
        script_path = handle.name
    try:
        cmd = [resolved_octave, "--quiet", script_path]
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=_octave_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"Octave exited with code {proc.returncode}")
        values = [float(line) for line in proc.stdout.splitlines() if line.strip()]
        if len(values) != len(samples):
            raise RuntimeError(f"Expected {len(samples)} Octave values, got {len(values)}.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        return values
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def evaluate_with_cpp(problem: int, samples: list[list[float]]):
    evaluator = _build_cpp_evaluator(problem)
    values = evaluator(samples)
    return [float(v) for v in values]


def generate_samples(problem: int, sample_count: int, rng: random.Random):
    dimension, lb, ub = CEC2011_METADATA[problem]
    samples = []
    midpoint = [(lo + hi) * 0.5 for lo, hi in zip(lb, ub)]
    samples.append(midpoint)
    if sample_count <= 1:
        return samples

    for _ in range(sample_count - 1):
        row = []
        for lo, hi in zip(lb, ub):
            if hi == lo:
                row.append(lo)
            else:
                span = hi - lo
                # Avoid hitting exact boundaries for sensitive trajectory cases.
                frac = rng.uniform(0.1, 0.9) if problem in (21, 22) else rng.random()
                row.append(lo + frac * span)
        if len(row) != dimension:
            raise RuntimeError(f"Generated wrong dimension for problem {problem}")
        samples.append(row)
    return samples


def compare_problem(problem: int, sample_count: int, seed: int, octave_bin: str | None = None):
    rng = random.Random(seed + problem * 1009)
    samples = generate_samples(problem, sample_count, rng)
    cpp_values = evaluate_with_cpp(problem, samples)
    octave_values = evaluate_with_octave(problem, samples, octave_bin)

    atol, rtol = get_tolerance(problem)
    max_abs = 0.0
    max_rel = 0.0
    failures = []
    for index, (cpp_val, oct_val) in enumerate(zip(cpp_values, octave_values)):
        abs_diff = abs(cpp_val - oct_val)
        scale = max(abs(oct_val), 1.0)
        rel_diff = abs_diff / scale
        max_abs = max(max_abs, abs_diff)
        max_rel = max(max_rel, rel_diff)
        if abs_diff > atol and rel_diff > rtol:
            failures.append((index, cpp_val, oct_val, abs_diff, rel_diff))

    return {
        "problem": problem,
        "samples": len(samples),
        "atol": atol,
        "rtol": rtol,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "failures": failures,
    }


def parse_problems(raw: str):
    problems = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_str, end_str = chunk.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            step = 1 if end >= start else -1
            problems.extend(range(start, end + step, step))
        else:
            problems.append(int(chunk))
    deduped = []
    seen = set()
    for problem in problems:
        if problem not in seen:
            deduped.append(problem)
            seen.add(problem)
    return deduped


def main(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description="Compare CEC2011 C++ translations against the original MATLAB/Octave implementations.")
    parser.add_argument("--problems", default="1-22", help="Problem numbers to check, e.g. '1-10,21,22'.")
    parser.add_argument("--samples", type=int, default=1000, help="Number of test points per problem.")
    parser.add_argument("--seed", type=int, default=20260601, help="Random seed for reproducible sampling.")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of problems to compare in parallel.",
    )
    parser.add_argument(
        "--octave-bin",
        default=None,
        help="Path to the Octave executable. Overrides PATH lookup; also supports OCTAVE_CLI/OCTAVE_BIN.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    problems = parse_problems(args.problems)
    bad = False
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    if args.workers == 1 or len(problems) <= 1:
        results = [compare_problem(problem, args.samples, args.seed, args.octave_bin) for problem in problems]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(compare_problem, problem, args.samples, args.seed, args.octave_bin)
                for problem in problems
            ]
            results = [future.result() for future in futures]

    for result in results:
        problem = result["problem"]
        status = "PASS" if not result["failures"] else "FAIL"
        print(
            f"F{problem:02d} {status}  samples={result['samples']}  "
            f"failed={len(result['failures'])}  "
            f"max_abs={result['max_abs']:.6e}  max_rel={result['max_rel']:.6e}  "
            f"tol=({result['atol']:.1e}, {result['rtol']:.1e})"
        )
        for failure in result["failures"][:3]:
            idx, cpp_val, oct_val, abs_diff, rel_diff = failure
            print(
                f"  sample={idx} cpp={cpp_val:.17g} octave={oct_val:.17g} "
                f"abs_diff={abs_diff:.6e} rel_diff={rel_diff:.6e}"
            )
        bad = bad or bool(result["failures"])

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
