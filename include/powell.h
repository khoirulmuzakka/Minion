#ifndef POWELL_H
#define POWELL_H

#include "minimizer_base.h"

class Powell : public MinimizerBase {
public:
    Powell(MinionFunction func, const std::vector<std::pair<double, double>>& bounds, const std::vector<double>& x0 = {},
           void* data = nullptr, std::function<void(MinionResult*)> callback = nullptr,
           double relTol = 0.0001, int maxevals = 100000, std::string boundStrategy = "reflect-random", int seed=-1);

    MinionResult optimize() override;

private:
    double Powell::line_minimization(std::vector<double>& x, const std::vector<double>& direction, size_t& nfev);
    std::vector<std::vector<double>> xtemp;
};

#endif // POWELL_H
