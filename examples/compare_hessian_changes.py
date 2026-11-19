"""Compare current Hessian code against the revision before bound caps/diagnostics.

Run from any directory: python examples/compare_hessian_changes.py
Requires CMake, a C++17 compiler, pybind11, NumPy, git, and cached Eigen headers
from a regular repository build (or pass --eigen-include).
Builds two small isolated extensions; never replaces the installed extension.
Writes docs/hessian_changes_comparison.md and .json with reproducible results.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
from statistics import NormalDist
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "546a835"
WRAPPER = r'''
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "uncertainty.h"
namespace py = pybind11;
using namespace minion::uncertainty;
PYBIND11_MODULE(@MODULE@, m) {
    m.def("hessian", [](py::function f, const std::vector<double>& x,
        const std::vector<std::pair<double,double>>& bounds, py::dict options) {
        HessianOptions o;
        if (options.contains("step_sizes")) o.step_sizes=options["step_sizes"].cast<std::vector<double>>();
        if (options.contains("parameter_scales")) o.parameter_scales=options["parameter_scales"].cast<std::vector<double>>();
        if (options.contains("relative_step")) o.relative_step=options["relative_step"].cast<double>();
        if (options.contains("step_candidates")) o.step_candidates=options["step_candidates"].cast<int>();
        if (options.contains("step_tolerance")) o.step_tolerance=options["step_tolerance"].cast<double>();
        if (options.contains("eigenvalue_tolerance")) o.eigenvalue_tolerance=options["eigenvalue_tolerance"].cast<double>();
        if (options.contains("batch_size")) o.batch_size=options["batch_size"].cast<int>();
        // Pin policies explicitly so the comparison isolates the latest changes.
        o.curvature_policy=CurvaturePolicy::Regularize;
        minion::MinionFunction objective=[&](const auto& points, void*) {
            return f(points).cast<std::vector<double>>();
        };
        const auto r=Uncertainty(objective,x,bounds).hessian(1.,o);
        py::dict d;
        d["status"]=r.status;d["std"]=r.std;d["covariance"]=py::cast(r.covariance,py::return_value_policy::copy);
        d["hessian"]=py::cast(r.hessian,py::return_value_policy::copy);d["steps"]=r.step_sizes;
        d["stencils"]=r.stencil_types;d["eigenvalues"]=py::cast(r.scaled_eigenvalues,py::return_value_policy::copy);
        d["repaired"]=r.forced_positive_definite;d["covariance_available"]=r.covariance_available;
        d["nfev"]=r.nfev;d["calls"]=r.objective_calls;
#if @CURRENT@
        d["verified"]=r.step_selection_verified;d["limited"]=r.step_limited_by_bounds;
        d["resolved"]=r.curvature_resolved;
#endif
        return d;
    });
}
'''


def build_modules(eigen_include):
    src = ROOT / "build/hessian-comparison-src"
    build = ROOT / "build/hessian-comparison-build"
    src.mkdir(parents=True, exist_ok=True)
    if eigen_include is None:
        cached = sorted((ROOT / ".dependencies").glob("*/eigen-src"))
        if not cached:
            raise RuntimeError("Build the repository first or pass --eigen-include.")
        eigen_include = cached[0]
    eigen_include = eigen_include.resolve()
    if not (eigen_include / "Eigen/Core").is_file():
        raise ValueError("--eigen-include must contain Eigen/Core")
    pybind_dir = subprocess.check_output(
        [sys.executable, "-m", "pybind11", "--cmakedir"], text=True).strip()
    hashes = {}
    cmake = ['cmake_minimum_required(VERSION 3.18)',
             'project(HessianComparison LANGUAGES CXX)',
             'set(CMAKE_CXX_STANDARD 17)',
             'find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)',
             'find_package(pybind11 CONFIG REQUIRED)']
    for label, current in (("baseline", False), ("current", True)):
        snapshot = src / label
        snapshot.mkdir(exist_ok=True)
        for name in ("uncertainty.h", "hessian.cpp"):
            relative = f"minion/uncertainty/{name}"
            content = ((ROOT / relative).read_bytes() if current else
                       subprocess.check_output(["git", "show", f"{BASELINE}:{relative}"], cwd=ROOT))
            (snapshot / name).write_bytes(content)
            hashes[f"{label}/{name}"] = hashlib.sha256(content).hexdigest()
        name = f"hessian_{label}"
        wrapper = WRAPPER.replace("@MODULE@", name).replace("@CURRENT@", str(int(current)))
        (snapshot / "bindings.cpp").write_text(wrapper)
        cmake.extend([
            f'pybind11_add_module({name} MODULE NO_EXTRAS "{label}/bindings.cpp" "{label}/hessian.cpp")',
            f'target_include_directories({name} PRIVATE "{snapshot.as_posix()}" '
            f'"{(ROOT / "minion/utils").as_posix()}" '
            f'"{(ROOT / "minion/algorithms").as_posix()}" "{eigen_include.as_posix()}")',
            f'set_target_properties({name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/modules/$<0:>")',
        ])
    (src / "CMakeLists.txt").write_text("\n".join(cmake) + "\n")
    configure = ["cmake", "-S", str(src), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                 f"-Dpybind11_DIR={pybind_dir}", f"-DPython_EXECUTABLE={sys.executable}"]
    subprocess.run(configure, check=True)
    subprocess.run(["cmake", "--build", str(build), "--config", "Release", "--parallel", "2"], check=True)
    sys.path.insert(0, str(build / "modules"))
    return {key: importlib.import_module(f"hessian_{key}") for key in ("baseline", "current")}, hashes


def run(module, objective, optimum, bounds=(), **options):
    points, calls = [], []
    def tracked(batch):
        batch = np.asarray(batch)
        points.extend(batch.tolist())
        calls.append(len(batch))
        return objective(batch)
    r = module.hessian(tracked, list(optimum), list(bounds), options)
    assert r["nfev"] == sum(calls) and r["calls"] == len(calls)
    if bounds:
        p = np.asarray(points)
        assert np.all(p >= np.asarray(bounds)[:, 0]) and np.all(p <= np.asarray(bounds)[:, 1])
    return r


def scalar_summary(r, exact_sd):
    sd = float(r["std"][0]) if r["covariance_available"] else None
    return dict(status=r["status"], step=r["steps"][0], stencil=r["stencils"][0],
                raw_curvature=float(r["hessian"][0, 0]), sd=sd,
                relative_sd_error=abs(sd/exact_sd-1) if sd is not None else None,
                nfev=r["nfev"], calls=r["calls"], repaired=r["repaired"],
                verified=r.get("verified"), limited=r.get("limited"))


def experiments(modules):
    rows, checks = [], []
    # First isolate finite-difference bias from inference: known local Hessian = 1.
    # These quartic examples are not claimed to have Gaussian estimator distributions.
    functions = [
        ("Interior nonlinear curvature", lambda p: .5*p[:, 0]**2+100*p[:, 0]**4,
         [0.], [(-.1, .1)], dict(relative_step=.04)),
        ("Active-bound nonlinear curvature", lambda p: .5*p[:, 0]**2+.5*p[:, 0]**4,
         [0.], [(0., 1.)], dict(step_candidates=1, relative_step=.5)),
        ("Clean bounded quadratic", lambda p: .5*((p[:, 0]-.5)/.1)**2,
         [.5], [(0., 1.)], dict(relative_step=.02)),
    ]
    for label, f, optimum, bounds, options in functions:
        results = {version: run(module, f, optimum, bounds, **options)
                   for version, module in modules.items()}
        exact_sd = .1 if label == "Clean bounded quadratic" else 1.
        summary = {version: scalar_summary(r, exact_sd) for version, r in results.items()}
        rows.append(dict(case=label, exact_local_sd=exact_sd, results=summary))
        if label != "Clean bounded quadratic":
            assert summary["current"]["relative_sd_error"] < summary["baseline"]["relative_sd_error"]
        else:
            for r in results.values():
                np.testing.assert_allclose(r["std"], [exact_sd], rtol=1e-10)

    # Paired full uncertainty pipelines on exact scalar Gaussian mean fits.
    # The optimum is known analytically, so optimization error is excluded.
    rng = np.random.default_rng(20261119)
    z = NormalDist().inv_cdf(.975)
    mc_rows = []
    for label, truth, sd, bounds, quantum, initial in (
        ("Clean interior Gaussian fits", .5, .1, [(0., 1.)], 0., .02),
        ("Tight bounds and quantized objective", .5, .001, [(.49, .51)], .6, .0005),
        ("Gaussian fits at an active bound", 0., .2/np.sqrt(40), [(0., 3.)], 0., .5),
    ):
        samples = rng.normal(truth, sd, 1000)
        phases = rng.uniform(0., quantum, len(samples)) if quantum else np.zeros(len(samples))
        accum = {v: dict(ratios=[], covered=0, available=0, repaired=0, calls=0,
                         evaluations=0, unverified=0) for v in modules}
        for sample, phase in zip(samples, phases):
            optimum = np.clip(sample, bounds[0][0], bounds[0][1])
            def objective(p):
                value = 19.+.5*((p[:, 0]-sample)/sd)**2
                return quantum*np.round((value+phase)/quantum)-phase if quantum else value
            for version, module in modules.items():
                r = run(module, objective, [optimum], bounds, relative_step=initial)
                a = accum[version]
                a["calls"] += r["calls"]
                a["evaluations"] += r["nfev"]
                a["repaired"] += r["repaired"]
                a["unverified"] += "verified" in r and not all(r["verified"])
                if r["covariance_available"]:
                    width = r["std"][0]
                    a["available"] += 1
                    a["ratios"].append(width/sd)
                    a["covered"] += abs(optimum-truth) <= z*width
        summary = {}
        for version, a in accum.items():
            summary[version] = dict(
                available=a["available"], repairs=a["repaired"],
                median_sd_ratio=float(np.median(a["ratios"])) if a["ratios"] else None,
                coverage_available=a["covered"]/a["available"] if a["available"] else None,
                covered_fraction_all=a["covered"]/len(samples),
                mean_evaluations=a["evaluations"]/len(samples), mean_calls=a["calls"]/len(samples),
                unverified=a["unverified"] if version=="current" else None)
        mc_rows.append(dict(case=label, trials=len(samples), results=summary))
        if quantum == 0:
            for a in summary.values():
                assert abs(a["median_sd_ratio"]-1) < 1e-7

    # No-bound control: estimates and function counts must be unchanged; only
    # the newly reported reliability information should differ.
    rotation = np.array([[1., -1.], [1., 1.]])/np.sqrt(2.)
    hessian = rotation @ np.diag([200., .0002]) @ rotation.T
    exact_cov = (rotation/np.array([200., .0002])) @ rotation.T
    def difficult(p):
        return 19.+.5*np.einsum("bi,ij,bj->b", p, hessian, p)+1e-8*np.all(p==0, axis=1)
    paired = {v: run(m, difficult, [0., 0.]) for v,m in modules.items()}
    np.testing.assert_allclose(paired["baseline"]["covariance"], paired["current"]["covariance"], rtol=1e-12)
    assert paired["baseline"]["nfev"] == paired["current"]["nfev"]
    assert paired["baseline"]["calls"] == paired["current"]["calls"]
    assert not paired["current"]["resolved"]
    checks.append(dict(case="Noisy nearly-degenerate unbounded fit",
                       exact_sd=float(np.sqrt(exact_cov[0, 0])),
                       previous_sd=paired["baseline"]["std"][0], current_sd=paired["current"]["std"][0],
                       evaluations=paired["current"]["nfev"], calls=paired["current"]["calls"],
                       curvature_resolved=paired["current"]["resolved"]))
    return dict(local_curvature=rows, coverage=mc_rows, controls=checks)


def write_report(result, hashes):
    lines = ["# Impact of bound-aware steps and curvature diagnostics", "",
             f"Baseline: git revision `{BASELINE}`. Current: working-tree C++ source snapshots.",
             "Both versions were compiled as isolated Python extensions with the same compiler,",
             "Eigen headers and bindings. The installed extension was not modified.", "",
             "Reproduce: `python examples/compare_hessian_changes.py`.", "",
             "## Accuracy of local curvature", "",
             "The nonlinear cases compare with the exact inverse local Hessian; these are",
             "derivative-accuracy checks, not claims of Gaussian statistical coverage.", "",
             "| Case | Version | Selected h | Stencil | SD | Exact local SD | SD error | Evaluations/calls |",
             "|---|---|---:|---|---:|---:|---:|---:|"]
    for case in result["local_curvature"]:
        for version, r in case["results"].items():
            lines.append(f'| {case["case"]} | {version} | {r["step"]:.6g} | {r["stencil"]} | '
                         f'{r["sd"]:.6g} | {case["exact_local_sd"]:.6g} | '
                         f'{100*r["relative_sd_error"]:.3f}% | {r["nfev"]}/{r["calls"]} |')
    lines += ["", "## Gaussian mean fits: paired coverage experiments", "",
              "Each case uses 1,000 independent datasets summarized by their exact Gaussian",
              "sample means, with both implementations evaluated at the same exact constrained",
              "optimum. Nominal intervals are the returned symmetric marginal 95% intervals.",
              "The bounded quadratic likelihood has a known unconstrained estimator SD.",
              "At an active bound, the constrained estimator is non-Gaussian; Gaussian",
              "intervals need not attain 95% coverage even with exact curvature.", "",
              "For quantization, the objective is rounded in increments of 0.6 with a random",
              "phase per dataset, fixed for all evaluations and shared between versions.",
              "This is computational noise, separate from the Gaussian measurement noise.",
              "Optimization error is deliberately excluded. The tight-bound case is adverse:",
              "shrinking steps can make curvature indistinguishable from rounding noise.", "",
              "Coverage is conditional on covariance being available. The accompanying",
              "available count matters; unavailable results are not silently excluded from",
              "the JSON's `covered_fraction_all`. Monte Carlo sampling errors for coverage",
              "near 95% are about 0.7 percentage points for 1,000 trials.", "",
              "| Case | Version | Available / 1000 | Median SD / reference SD | 95% coverage, available | Repairs | Mean evaluations/calls |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for case in result["coverage"]:
        for version, r in case["results"].items():
            lines.append(f'| {case["case"]} | {version} | {r["available"]} | {r["median_sd_ratio"]:.5g} | '
                         f'{100*r["coverage_available"]:.2f}% | {r["repairs"]} | '
                         f'{r["mean_evaluations"]:.3f}/{r["mean_calls"]:.3f} |')
    lines += ["", "## New diagnostics", "",
              "| Case | Unverified step selections / 1000 |",
              "|---|---:|"]
    for case in result["coverage"]:
        r = case["results"]["current"]
        lines.append(f'| {case["case"]} | {r["unverified"]} |')
    control = result["controls"][0]
    lines += ["", f'The unbounded nearly-degenerate control returns the same SD before and after: '
              f'**{control["current_sd"]:.6g}**, against an exact SD of **{control["exact_sd"]:.6g}**.',
              f'Both versions use **{control["evaluations"]} evaluations / {control["calls"]} calls**.',
              'The new implementation flags unresolved raw curvature.',
              "Regularization still does not recover the true uncertainty.", "",
              "## Interpretation", "",
              "A bound-distance cap is a locality constraint, not a general accuracy guarantee.",
              "It can reduce nonlinear finite-difference bias, but can increase noise error.",
              "A verification flag means coordinate probes passed the consistency heuristic;",
              "it does not certify weak coupled directions or statistical coverage.",
              "If the admissible local region is too small",
              "relative to evaluation noise, improve objective accuracy, reconsider bounds,",
              "or use an independent inference method rather than treating repaired intervals",
              "as validated uncertainties.", ""]
    report = "\n".join(lines)
    (ROOT / "docs/hessian_changes_comparison.md").write_text(report)
    (ROOT / "docs/hessian_changes_comparison.json").write_text(json.dumps(
        dict(baseline=BASELINE, source_sha256=hashes, results=result), indent=2)+"\n")
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eigen-include", type=Path)
    args = parser.parse_args()
    modules, hashes = build_modules(args.eigen_include)
    write_report(experiments(modules), hashes)
