#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <libcmaes/cmaes.h>
#include <libcmaes/genopheno.h>

namespace py = pybind11;

namespace {

template <class TGenoPheno>
py::dict run_cmaes_impl(
    const py::function& objective,
    const std::vector<double>& x0,
    double sigma0,
    const std::vector<std::pair<double, double>>& bounds,
    const std::string& algo,
    int lambda,
    std::uint64_t seed,
    int maxevals) {
    using namespace libcmaes;

    auto fitfunc = [objective](const double* x, const int n) -> double {
        py::gil_scoped_acquire gil;
        py::list point;
        for (int i = 0; i < n; ++i) {
            point.append(x[i]);
        }
        py::object value = objective(point);
        return py::cast<double>(value);
    };
    libcmaes::FitFunc fitfunc_obj = fitfunc;

    CMASolutions solutions;

    if constexpr (std::is_same_v<TGenoPheno, GenoPheno<NoBoundStrategy>>) {
        CMAParameters<TGenoPheno> params(x0, sigma0, lambda, seed);
        params.set_str_algo(algo);
        if (maxevals > 0) {
            params.set_max_fevals(maxevals);
        }
        solutions = cmaes<TGenoPheno>(fitfunc_obj, params);
    } else {
        std::vector<double> lbounds;
        std::vector<double> ubounds;
        lbounds.reserve(bounds.size());
        ubounds.reserve(bounds.size());
        for (const auto& bound : bounds) {
            lbounds.push_back(bound.first);
            ubounds.push_back(bound.second);
        }

        GenoPheno<pwqBoundStrategy> gp(lbounds.data(), ubounds.data(), static_cast<int>(bounds.size()));
        CMAParameters<TGenoPheno> params(x0, sigma0, lambda, seed, gp);
        params.set_str_algo(algo);
        if (maxevals > 0) {
            params.set_max_fevals(maxevals);
        }
        solutions = cmaes<TGenoPheno>(fitfunc_obj, params);
    }

    const Candidate best = solutions.best_candidate();
    py::dict result;
    result["best_f"] = best.get_fvalue();
    result["best_x"] = best.get_x();
    result["xmean"] = std::vector<double>(solutions.xmean().data(), solutions.xmean().data() + solutions.xmean().size());
    result["sigma"] = solutions.sigma();
    result["nevals"] = solutions.nevals();
    result["niter"] = solutions.niter();
    result["elapsed_ms"] = solutions.elapsed_time();
    result["status"] = solutions.run_status();
    result["status_msg"] = solutions.status_msg();
    return result;
}

py::dict run_cmaes(
    const py::function& objective,
    const std::vector<double>& x0,
    double sigma0,
    const std::vector<std::pair<double, double>>& bounds,
    const std::string& algo,
    int lambda,
    std::uint64_t seed,
    int maxevals) {
    if (x0.empty()) {
        throw std::invalid_argument("x0 must not be empty");
    }
    if (!bounds.empty() && bounds.size() != x0.size()) {
        throw std::invalid_argument("bounds and x0 must have the same dimension");
    }
    if (sigma0 <= 0.0) {
        sigma0 = 0.3;
    }

    if (bounds.empty()) {
        return run_cmaes_impl<libcmaes::GenoPheno<libcmaes::NoBoundStrategy>>(
            objective, x0, sigma0, bounds, algo, lambda, seed, maxevals);
    }
    return run_cmaes_impl<libcmaes::GenoPheno<libcmaes::pwqBoundStrategy>>(
        objective, x0, sigma0, bounds, algo, lambda, seed, maxevals);
}

} // namespace

PYBIND11_MODULE(libcmaes_bridge, m) {
    m.doc() = "Minimal libcmaes bridge for benchmark comparisons";

    m.def(
        "optimize",
        &run_cmaes,
        py::arg("objective"),
        py::arg("x0"),
        py::arg("sigma0") = 0.3,
        py::arg("bounds") = std::vector<std::pair<double, double>>{},
        py::arg("algo") = "acmaes",
        py::arg("lambda") = -1,
        py::arg("seed") = 0,
        py::arg("maxevals") = -1);
}
