#include "minimizer_base.h"

MinimizerBase::MinimizerBase(MinionFunction func, const std::vector<std::pair<double, double>>& bounds, const std::vector<double>& x0,
                             void* data, std::function<void(MinionResult*)> callback, double relTol, int maxevals, std::string boundStrategy, int seed)
    : func(func), bounds(bounds), x0(x0), data(data), callback(callback), relTol(relTol), maxevals(maxevals), boundStrategy(boundStrategy), seed(seed){
    if (!bounds.empty() && bounds[0].first >= bounds[0].second) {
        throw std::invalid_argument("Invalid bounds.");
    }
    if (!x0.empty() && x0.size() != bounds.size()) {
        throw std::invalid_argument("x0 must have the same dimension as the length of the bounds.");
    }
     if (seed != -1) set_global_seed(seed);
}

