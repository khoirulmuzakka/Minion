#include "cec2011.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <Eigen/Dense>

namespace minion {

CEC2011Functions::CEC2011Functions(int function_number, int dimension)
    : CECBase(function_number, dimension) {
    testfunc = &CEC2011::evaluate;
}

namespace {
constexpr double kTwoPi = 2.0 * PI;

constexpr std::array<std::array<double, 4>, 10> kProblem3Coeff = {
    {{{0.002918487, -0.008045787, 0.006749947, -0.001416647}},
     {{9.509977, -35.00994, 42.83329, -17.33333}},
     {{26.82093, -95.56079, 113.0398, -44.29997}},
     {{208.7241, -719.8052, 827.7466, -316.6655}},
     {{1.350005, -6.850027, 12.16671, -6.666689}},
     {{0.01921995, -0.0794532, 0.110566, -0.05033333}},
     {{0.1323596, -0.469255, 0.5539323, -0.2166664}},
     {{7.339981, -25.27328, 29.93329, -11.99999}},
     {{-0.3950534, 1.679353, -1.777829, 0.4974987}},
     {{-2.504665e-05, 0.01005854, -0.01986696, 0.00983347}}}};

template <size_t N, typename Deriv>
void rk4Step(std::array<double, N> &state, double t, double dt, Deriv &&deriv) {
    std::array<double, N> k1{};
    deriv(t, state, k1);

    std::array<double, N> temp{};
    for (size_t j = 0; j < N; ++j) {
        temp[j] = state[j] + 0.5 * dt * k1[j];
    }

    std::array<double, N> k2{};
    deriv(t + 0.5 * dt, temp, k2);
    for (size_t j = 0; j < N; ++j) {
        temp[j] = state[j] + 0.5 * dt * k2[j];
    }

    std::array<double, N> k3{};
    deriv(t + 0.5 * dt, temp, k3);
    for (size_t j = 0; j < N; ++j) {
        temp[j] = state[j] + dt * k3[j];
    }

    std::array<double, N> k4{};
    deriv(t + dt, temp, k4);

    for (size_t j = 0; j < N; ++j) {
        state[j] += dt * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]) / 6.0;
    }
}

template <size_t N, typename Deriv>
std::array<double, N> rk4Integrate(std::array<double, N> state, double t0, double t1,
                                   int steps, Deriv &&deriv) {
    const double dt = (t1 - t0) / static_cast<double>(steps);
    double t = t0;
    for (int i = 0; i < steps; ++i) {
        rk4Step(state, t, dt, deriv);
        t += dt;
    }
    return state;
}

std::array<double, 10> buildProblem3Rates(double u) {
    const double u2 = u * u;
    const double u3 = u2 * u;
    std::array<double, 10> k{};
    for (size_t row = 0; row < kProblem3Coeff.size(); ++row) {
        const auto &c = kProblem3Coeff[row];
        k[row] = c[0] + c[1] * u + c[2] * u2 + c[3] * u3;
    }
    return k;
}

struct TersoffParams {
    double R1;
    double R2;
    double A;
    double B;
    double lambda1;
    double lambda2;
    double lambda3;
    double c;
    double d;
    double n;
    double gamma;
    double h;
};

const TersoffParams kProblem5Params{3.0,
                                    0.2,
                                    3.2647e+3,
                                    9.5373e+1,
                                    3.2394,
                                    1.3258,
                                    1.3258,
                                    4.8381,
                                    2.0417,
                                    22.956,
                                    0.33675,
                                    0.0};

const TersoffParams kProblem6Params{2.85,
                                    0.15,
                                    1.8308e+3,
                                    4.7118e+2,
                                    2.4799,
                                    1.7322,
                                    1.7322,
                                    1.0039e+05,
                                    1.6218e+01,
                                    7.8734e-01,
                                    1.0999e-06,
                                    -5.9826e-01};

struct LineSpec {
    int from;
    int to;
    double reactance;
    double capacity;
    double cost;
};

constexpr std::array<LineSpec, 7> kBaseLines = {{{1, 2, 0.4, 1.0, 0.0},
                                                 {1, 4, 0.6, 0.8, 0.0},
                                                 {1, 5, 0.2, 1.0, 0.0},
                                                 {2, 3, 0.2, 1.0, 0.0},
                                                 {2, 4, 0.4, 1.0, 0.0},
                                                 {3, 5, 0.2, 1.0, 20.0},
                                                 {6, 2, 0.3, 1.0, 30.0}}};

constexpr std::array<LineSpec, 15> kCandidateLines = {{{1, 2, 0.4, 1.0, 40.0},
                                                       {1, 3, 0.38, 1.0, 38.0},
                                                       {1, 4, 0.6, 0.8, 60.0},
                                                       {1, 5, 0.2, 1.0, 20.0},
                                                       {1, 6, 0.68, 0.7, 68.0},
                                                       {2, 3, 0.2, 1.0, 20.0},
                                                       {2, 4, 0.4, 1.0, 40.0},
                                                       {2, 5, 0.31, 1.0, 31.0},
                                                       {6, 2, 0.3, 1.0, 30.0},
                                                       {3, 4, 0.69, 0.82, 59.0},
                                                       {3, 5, 0.2, 1.0, 20.0},
                                                       {6, 3, 0.48, 1.0, 48.0},
                                                       {4, 5, 0.63, 0.75, 63.0},
                                                       {4, 6, 0.30, 1.0, 30.0},
                                                       {5, 6, 0.61, 0.78, 61.0}}};

constexpr std::array<double, 6> kPgen = {0.5, 0.0, 1.65, 0.0, 0.0, 5.45};
constexpr std::array<double, 6> kPload = {0.8, 2.4, 0.4, 1.6, 2.4, 0.0};

double evaluateProblem1(const double *x, int nx) {
    if (nx < 6) {
        throw std::runtime_error("Problem01 expects dimension 6");
    }
    const double theta = kTwoPi / 100.0;
    double sum = 0.0;
    for (int t = 0; t <= 100; ++t) {
        const double inner = x[3] * t * theta + x[4] * std::sin(x[5] * t * theta);
        const double y_t = x[0] * std::sin(x[1] * t * theta + x[2] * std::sin(inner));
        const double y0 =
            std::sin(5.0 * t * theta - 1.5 * std::sin(4.8 * t * theta + 2.0 * std::sin(4.9 * t * theta)));
        const double diff = y_t - y0;
        sum += diff * diff;
    }
    return sum;
}

double evaluateProblem2(const double *x, int nx) {
    if (nx % 3 != 0) {
        throw std::runtime_error("Problem02 dimension must be divisible by 3");
    }
    const int n = nx / 3;
    std::vector<std::array<double, 3>> atoms(n);
    for (int i = 0; i < n; ++i) {
        atoms[i] = {x[3 * i], x[3 * i + 1], x[3 * i + 2]};
    }

    double v = 0.0;
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double r2 = 0.0;
            for (int k = 0; k < 3; ++k) {
                double diff = atoms[i][k] - atoms[j][k];
                r2 += diff * diff;
            }
            const double r = std::sqrt(r2);
            if (r == 0.0) {
                continue;
            }
            const double inv_r6 = 1.0 / std::pow(r, 6.0);
            const double inv_r12 = inv_r6 * inv_r6;
            v += inv_r12 - 2.0 * inv_r6;
        }
    }
    return v;
}

double evaluateProblem3(const double *x, int nx) {
    if (nx < 1) {
        throw std::runtime_error("Problem03 expects dimension 1");
    }
    const double u = x[0];
    const auto rates = buildProblem3Rates(u);
    auto deriv = [&rates](double /*t*/, const std::array<double, 7> &state, std::array<double, 7> &dstate) {
        dstate[0] = -rates[0] * state[0];
        dstate[1] = rates[0] * state[0] - (rates[1] + rates[2]) * state[1] + rates[3] * state[4];
        dstate[2] = rates[1] * state[1];
        dstate[3] = -rates[5] * state[3] + rates[4] * state[4];
        dstate[4] = rates[2] * state[1] + rates[5] * state[3] - (rates[3] + rates[4] + rates[7] + rates[8]) * state[4] +
                    rates[6] * state[5] + rates[9] * state[6];
        dstate[5] = rates[7] * state[4] - rates[6] * state[5];
        dstate[6] = rates[8] * state[4] - rates[9] * state[6];
    };
    std::array<double, 7> state = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    state = rk4Integrate(state, 0.0, 0.78, 400, deriv);
    return state[6] * 1.0e3;
}

double evaluateProblem4(const double *x, int nx) {
    if (nx < 1) {
        throw std::runtime_error("Problem04 expects dimension 1");
    }
    const double u = x[0];
    auto deriv = [u](double /*t*/, const std::array<double, 2> &state, std::array<double, 2> &dstate) {
        const double exp_term = std::exp(25.0 * state[0] / (state[0] + 2.0));
        dstate[0] = -(2.0 + u) * (state[0] + 0.25) + (state[1] + 0.5) * exp_term;
        dstate[1] = 0.5 - state[1] - (state[1] + 0.5) * exp_term;
    };
    std::array<double, 2> state = {0.09, 0.09};
    const int steps = 400;
    const double dt = 0.78 / static_cast<double>(steps);
    double cost = 0.0;
    double t = 0.0;
    for (int i = 0; i < steps; ++i) {
        cost += state[0] * state[0] + state[1] * state[1] + 0.1 * u * u;
        rk4Step(state, t, dt, deriv);
        t += dt;
    }
    return cost;
}

double evaluateTersoff(const double *x, int nx, const TersoffParams &params) {
    if (nx % 3 != 0) {
        throw std::runtime_error("Tersoff problems expect dimension multiple of 3");
    }
    const int np = nx / 3;
    std::vector<std::array<double, 3>> atoms(np);
    for (int i = 0; i < np; ++i) {
        atoms[i] = {x[3 * i], x[3 * i + 1], x[3 * i + 2]};
    }

    auto idx = [np](int i, int j) { return i * np + j; };
    std::vector<double> r(np * np, 0.0);
    std::vector<double> fcr(np * np, 0.0);
    std::vector<double> VRr(np * np, 0.0);
    std::vector<double> VAr(np * np, 0.0);

    for (int i = 0; i < np; ++i) {
        for (int j = 0; j < np; ++j) {
            if (i == j) {
                continue;
            }
            double dist2 = 0.0;
            for (int k = 0; k < 3; ++k) {
                const double diff = atoms[i][k] - atoms[j][k];
                dist2 += diff * diff;
            }
            const double dist = std::sqrt(dist2);
            r[idx(i, j)] = dist;
            double fcr_val;
            if (dist < (params.R1 - params.R2)) {
                fcr_val = 1.0;
            } else if (dist > (params.R1 + params.R2)) {
                fcr_val = 0.0;
            } else {
                fcr_val = 0.5 - 0.5 * std::sin((PI / 2.0) * (dist - params.R1) / params.R2);
            }
            fcr[idx(i, j)] = fcr_val;
            VRr[idx(i, j)] = params.A * std::exp(-params.lambda1 * dist);
            VAr[idx(i, j)] = params.B * std::exp(-params.lambda2 * dist);
        }
    }

    std::vector<double> energy(np, 0.0);
    const double lambda3_pow3 = std::pow(params.lambda3, 3.0);
    for (int i = 0; i < np; ++i) {
        for (int j = 0; j < np; ++j) {
            if (i == j) {
                continue;
            }
            double jeta = 0.0;
            for (int k = 0; k < np; ++k) {
                if (i == k || j == k) {
                    continue;
                }
                const double rd1 = r[idx(i, k)];
                const double rd3 = r[idx(k, j)];
                const double rd2 = r[idx(i, j)];
                if (rd1 == 0.0 || rd2 == 0.0) {
                    continue;
                }
                const double ctheta = (rd1 * rd1 + rd2 * rd2 - std::pow(rd3, 3.0)) / (2.0 * rd1 * rd2);
                const double G = 1.0 + std::pow(params.c, 2.0) / std::pow(params.d, 2.0) -
                                 std::pow(params.c, 2.0) / (std::pow(params.d, 2.0) + std::pow(params.h - ctheta, 2.0));
                const double exponent = lambda3_pow3 * std::pow(rd2 - rd1, 3.0);
                jeta += fcr[idx(i, k)] * G * std::exp(exponent);
            }
            const double Bij = std::pow(1.0 + std::pow(params.gamma * jeta, params.n), -0.5 / params.n);
            energy[i] += fcr[idx(i, j)] * (VRr[idx(i, j)] - Bij * VAr[idx(i, j)]) * 0.5;
        }
    }
    return std::accumulate(energy.begin(), energy.end(), 0.0);
}

double evaluateProblem7(const double *x, int nx) {
    const int d = nx;
    const int var = 2 * d - 1;
    const int total = 2 * var;
    std::vector<double> hsum(total, 0.0);
    for (int kk = 1; kk <= total; ++kk) {
        if (kk % 2 == 1) {
            const int i = (kk + 1) / 2;
            double val = 0.0;
            for (int j = i; j <= d; ++j) {
                double summ = 0.0;
                const int lower = std::abs(2 * i - j - 1) + 1;
                for (int idx = lower; idx <= j; ++idx) {
                    const int xi = std::min(std::max(idx, 1), d) - 1;
                    summ += x[xi];
                }
                val += std::cos(summ);
            }
            hsum[kk - 1] = val;
        } else {
            const int i = kk / 2;
            double val = 0.0;
            for (int j = i + 1; j <= d; ++j) {
                double summ = 0.0;
                const int lower = std::abs(2 * i - j) + 1;
                for (int idx = lower; idx <= j; ++idx) {
                    const int xi = std::min(std::max(idx, 1), d) - 1;
                    summ += x[xi];
                }
                val += std::cos(summ);
            }
            hsum[kk - 1] = val + 0.5;
        }
    }
    return *std::max_element(hsum.begin(), hsum.end());
}

double evaluateProblem8(const double *x, int nx) {
    std::vector<int> switches(nx);
    for (int i = 0; i < nx; ++i) {
        double val = std::ceil(x[i]);
        if (!std::isfinite(val)) {
            val = 1.0;
        }
        int idx = static_cast<int>(val);
        idx = std::clamp(idx, 1, static_cast<int>(kCandidateLines.size()));
        switches[i] = idx - 1;
    }

    std::vector<LineSpec> lines(kBaseLines.begin(), kBaseLines.end());
    lines.reserve(kBaseLines.size() + switches.size());
    for (int idx : switches) {
        lines.push_back(kCandidateLines[static_cast<size_t>(idx)]);
    }

    constexpr int busCount = static_cast<int>(kPgen.size());
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(busCount, busCount);
    for (const auto &line : lines) {
        const double bline = 1.0 / line.reactance;
        const int k = line.from - 1;
        const int m = line.to - 1;
        B(k, m) -= bline;
        B(m, k) = B(k, m);
        B(k, k) += bline;
        B(m, m) += bline;
    }
    B(0, 0) = 1.0e7;

    Eigen::VectorXd delP(busCount);
    for (int i = 0; i < busCount; ++i) {
        delP(i) = kPgen[i] - kPload[i];
    }

    const Eigen::VectorXd delta = B.inverse() * delP;

    std::vector<double> flows(lines.size(), 0.0);
    for (size_t idx = 0; idx < lines.size(); ++idx) {
        const auto &line = lines[idx];
        flows[idx] = (delta(line.from - 1) - delta(line.to - 1)) / line.reactance;
    }

    double cost = 30.0;
    for (size_t i = kBaseLines.size(); i < lines.size(); ++i) {
        cost += lines[i].cost;
    }

    double penalty = 0.0;
    for (size_t i = 0; i < lines.size(); ++i) {
        penalty += 5000.0 * std::max(std::abs(flows[i]) - lines[i].capacity, 0.0);
    }

    for (size_t cand = 0; cand < kCandidateLines.size(); ++cand) {
        const int count = std::count(switches.begin(), switches.end(), static_cast<int>(cand));
        if (count > 3) {
            penalty += 1000.0;
        }
    }

    return cost + penalty;
}

} // namespace

namespace CEC2011 {

void evaluate(double *x, double *f, int nx, int mx, int func_num) {
    for (int i = 0; i < mx; ++i) {
        const double *xi = x + i * nx;
        switch (func_num) {
        case 1:
            f[i] = evaluateProblem1(xi, nx);
            break;
        case 2:
            f[i] = evaluateProblem2(xi, nx);
            break;
        case 3:
            f[i] = evaluateProblem3(xi, nx);
            break;
        case 4:
            f[i] = evaluateProblem4(xi, nx);
            break;
        case 5:
            f[i] = evaluateTersoff(xi, nx, kProblem5Params);
            break;
        case 6:
            f[i] = evaluateTersoff(xi, nx, kProblem6Params);
            break;
        case 7:
            f[i] = evaluateProblem7(xi, nx);
            break;
        case 8:
            f[i] = evaluateProblem8(xi, nx);
            break;
        default:
            throw std::runtime_error("CEC2011 function not implemented yet");
        }
    }
}

} // namespace CEC2011

} // namespace minion
