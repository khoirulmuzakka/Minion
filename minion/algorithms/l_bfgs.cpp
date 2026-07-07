#include "l_bfgs.h"

namespace minion {

namespace {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

struct LBFGSParam {
    int m = 10;
    double epsilon = 1e-10;
    double epsilon_rel = 1e-10;
    int past = 3;
    double delta = 1e-20;
    int max_iterations = 0;
    int max_linesearch = 20;
    double min_step = 1e-20;
    double max_step = 1e20;
    double ftol = 1e-4;
    double wolfe = 0.9;
};

class CompactLBFGSMatrix {
private:
    int n_ = 0;
    int m_ = 0;
    int ncorr_ = 0;
    double theta_ = 1.0;
    Matrix S_;
    Matrix Y_;
    Vector ys_;

public:
    void reset(int n, int m)
    {
        n_ = n;
        m_ = std::max(1, m);
        ncorr_ = 0;
        theta_ = 1.0;
        S_.resize(n_, m_);
        Y_.resize(n_, m_);
        ys_.resize(m_);
    }

    void add_correction(const Vector& s, const Vector& y)
    {
        const double ys = s.dot(y);
        if (!(ys > 0.0)) {
            return;
        }

        if (ncorr_ < m_) {
            S_.col(ncorr_) = s;
            Y_.col(ncorr_) = y;
            ys_[ncorr_] = ys;
            ncorr_++;
        } else {
            S_.leftCols(m_ - 1) = S_.rightCols(m_ - 1);
            Y_.leftCols(m_ - 1) = Y_.rightCols(m_ - 1);
            ys_.head(m_ - 1) = ys_.tail(m_ - 1);
            S_.col(m_ - 1) = s;
            Y_.col(m_ - 1) = y;
            ys_[m_ - 1] = ys;
        }

        theta_ = y.squaredNorm() / ys;
    }

    void apply_Hv(const Vector& v, double a, Vector& res) const
    {
        res = a * v;
        if (ncorr_ <= 0) {
            return;
        }

        Vector alpha = Vector::Zero(ncorr_);
        for (int i = ncorr_ - 1; i >= 0; --i) {
            alpha[i] = S_.col(i).dot(res) / ys_[i];
            res.noalias() -= alpha[i] * Y_.col(i);
        }

        res /= theta_;
        for (int i = 0; i < ncorr_; ++i) {
            const double beta = Y_.col(i).dot(res) / ys_[i];
            res.noalias() += (alpha[i] - beta) * S_.col(i);
        }
    }

    std::vector<double> compute_hessian_diagonal() const
    {
        std::vector<double> diagonal(static_cast<size_t>(n_), 1.0 / theta_);
        Vector res(n_);
        for (int i = 0; i < n_; ++i) {
            Vector e = Vector::Zero(n_);
            e[i] = 1.0;
            apply_Hv(e, 1.0, res);
            diagonal[static_cast<size_t>(i)] = std::max(std::fabs(res[i]), 1e-16);
        }
        return diagonal;
    }
};

class LineSearch {
private:
    static double quad_interp(double step_lo, double step_hi, double fx_lo, double fx_hi, double dg_lo)
    {
        const double fdiff = fx_hi - fx_lo;
        const double sdiff = step_hi - step_lo;
        const double smid = (step_hi + step_lo) / 2.0;
        double candid = (fdiff * step_lo - smid * sdiff * dg_lo) / (fdiff - sdiff * dg_lo);
        const bool bad = !std::isfinite(candid) ||
            candid <= std::min(step_lo, step_hi) ||
            candid >= std::max(step_lo, step_hi) ||
            std::min(std::fabs(candid - step_lo), std::fabs(candid - step_hi)) < 0.01 * std::fabs(sdiff);
        return bad ? smid : candid;
    }

public:
    template <typename Foo>
    static bool search(Foo& f, const LBFGSParam& param, const Vector& xp, const Vector& drt, double,
                       double& step, double& fx, Vector& grad, double& dg, Vector& x, std::string& message)
    {
        if (step <= 0.0) {
            message = "line search skipped: initial step must be positive";
            return false;
        }
        const double fx_init = fx;
        const double dg_init = dg;
        if (dg_init > 0.0) {
            message = "line search skipped: moving direction increases the objective function value";
            return false;
        }

        const double test_decr = param.ftol * dg_init;
        const double test_curv = -param.wolfe * dg_init;

        double step_hi = 0.0, fx_hi = 0.0;
        double step_lo = 0.0, fx_lo = fx_init, dg_lo = dg_init;
        Vector x_lo = xp, grad_lo = grad;

        int iter = 0;
        for (;;) {
            x.noalias() = xp + step * drt;
            fx = f(x, grad);
            dg = grad.dot(drt);

            if (fx - fx_init > step * test_decr || (step_lo > 0.0 && fx >= fx_lo)) {
                step_hi = step;
                fx_hi = fx;
                break;
            }
            if (std::fabs(dg) <= test_curv) {
                return true;
            }

            step_hi = step_lo;
            fx_hi = fx_lo;
            step_lo = step;
            fx_lo = fx;
            dg_lo = dg;
            x_lo.swap(x);
            grad_lo.swap(grad);

            if (dg >= 0.0) {
                break;
            }
            if (++iter >= param.max_linesearch) {
                x.swap(x_lo);
                grad.swap(grad_lo);
                message = "line search reached fallback step";
                return true;
            }
            step *= 2.0;
        }

        for (;;) {
            step = quad_interp(step_lo, step_hi, fx_lo, fx_hi, dg_lo);
            x.noalias() = xp + step * drt;
            fx = f(x, grad);
            dg = grad.dot(drt);

            if (fx - fx_init > step * test_decr || fx >= fx_lo) {
                step_hi = step;
                fx_hi = fx;
            } else {
                if (std::fabs(dg) <= test_curv) {
                    return true;
                }
                if (dg * (step_hi - step_lo) >= 0.0) {
                    step_hi = step_lo;
                    fx_hi = fx_lo;
                }
                step_lo = step;
                fx_lo = fx;
                dg_lo = dg;
                x_lo.swap(x);
                grad_lo.swap(grad);
            }

            if (++iter >= param.max_linesearch) {
                step = step_lo;
                fx = fx_lo;
                dg = dg_lo;
                x.swap(x_lo);
                grad.swap(grad_lo);
                message = "line search reached fallback step";
                return true;
            }
        }
    }
};

struct SolverCore {
    LBFGSParam param;
    CompactLBFGSMatrix bfgs;
    Vector fx_hist;
    Vector xp;
    Vector grad;
    Vector gradp;
    Vector drt;
    double gnorm = 0.0;
    bool had_issue = false;
    std::string message;

    explicit SolverCore(const LBFGSParam& p) : param(p) {}
};

}  // namespace

struct L_BFGS::InternalState {
    SolverCore core;

    explicit InternalState(const LBFGSParam& param) : core(param) {}
};

L_BFGS::~L_BFGS() {
    delete state;
    state = nullptr;
}

void L_BFGS::initialize() {
    hasInitialized = true;
}

double L_BFGS::fun_and_grad(const VectorXd& x, VectorXd& grad){
    if (Nevals > maxevals) throw MaxevalExceedError("Maxevals has been exceeded.");
    if (!state) throw std::runtime_error("L_BFGS internal state is not initialized.");

    const int m = std::max(static_cast<int>(std::ceil((static_cast<double>(N_points) - 1.0) / 2.0)), 1);
    std::vector<double> x_vec(x.data(), x.data() + x.size());
    std::vector<std::vector<double>> X;
    X.push_back(x_vec);

    std::vector<double> hvec;
    std::vector<double> sec_der = state->core.bfgs.compute_hessian_diagonal();
    const double ferr = std::max(std::fabs(last_f), 1.0) * func_noise_ratio;
    for (int i = 0; i < x.size(); i++) {
        double h_min = std::pow(epsilon, 0.5) * std::max(1.0, std::fabs(x[i]));
        double h_max = 0.01 * std::max(1.0, std::fabs(x[i]));
        double curvature = std::max(std::fabs(sec_der[static_cast<size_t>(i)]), 1e-16);
        double h_est = 2.0 * std::sqrt(std::max(ferr, 1e-32) / curvature);
        double h = std::min(h_max, std::max(h_min, h_est));
        if (h <= 0.0) {
            h = std::min(h_max, std::max(h_min, fin_diff_rel_step * std::max(1.0, std::fabs(x[i]))));
        }
        hvec.push_back(h);
    }

    if (N_points == 1) {
        for (int i = 0; i < x.size(); i++) {
            std::vector<double> xp = x_vec;
            xp[i] += hvec[static_cast<size_t>(i)];
            X.push_back(xp);
        }
    } else {
        for (int i = 0; i < x.size(); i++) {
            for (int j = 1; j <= m; j++) {
                double h = hvec[static_cast<size_t>(i)];
                std::vector<double> xpp = x_vec;
                std::vector<double> xpm = x_vec;
                xpp[i] += j * h;
                xpm[i] -= j * h;
                X.push_back(xpp);
                X.push_back(xpm);
            }
        }
    }

    auto F = func(X, data);
    Nevals += F.size();
    double f = F[0];
    last_f = f;

    size_t best_idx = findArgMin(F);
    if (F[best_idx] < f_best) {
        best = X[best_idx];
        f_best = F[best_idx];
        minionResult = MinionResult(best, f_best, 1, Nevals, false, "");
        history.push_back(minionResult);
    }

    std::vector<double> grad_vec;
    if (N_points == 1) {
        for (int i = 0; i < x.size(); i++) grad_vec.push_back((F[static_cast<size_t>(i + 1)] - f) / (X[static_cast<size_t>(i + 1)][i] - x[i]));
    } else {
        int k = 1;
        for (int i = 0; i < x.size(); i++) {
            double h = hvec[static_cast<size_t>(i)];
            double grad_val = 0.0;
            for (int j = 1; j <= m; j++) {
                double cj = j / (m * (m + 1) * (2.0 * m + 1));
                grad_val += cj * (F[static_cast<size_t>(k)] - F[static_cast<size_t>(k + 1)]);
                k += 2;
            }
            grad_vec.push_back(3.0 * grad_val / h);
        }
    }

    grad = Eigen::Map<Eigen::VectorXd>(grad_vec.data(), static_cast<Eigen::Index>(grad_vec.size()));
    return f;
}

MinionResult L_BFGS::optimize() {
    try {
        history.clear();
        auto defaultKey = DefaultSettings().getDefaultSettings("L_BFGS");
        for (auto el : optionMap) defaultKey[el.first] = el.second;
        Options options(defaultKey);

        LBFGSParam param;
        param.m              = options.get<int>("m", 10);
        param.epsilon        = options.get<double>("g_epsilon", 1e-10);
        param.epsilon_rel    = options.get<double>("g_epsilon_rel", 1e-10);
        param.past           = 3;
        param.delta          = options.get<double>("f_reltol", 1e-20);
        param.max_iterations = options.get<int>("max_iterations", 0);
        param.max_linesearch = options.get<int>("max_linesearch", 20);
        param.min_step       = 1e-20;
        param.max_step       = 1e20;
        param.ftol           = options.get<double>("c_1", 1e-4);
        param.wolfe          = options.get<double>("c_2", 0.9);

        N_points = options.get<int>("N_points_derivative", 1);
        func_noise_ratio = options.get<double>("func_noise_ratio", 1e-10);
        Nevals = 0;
        f_best = std::numeric_limits<double>::max();
        best.clear();

        delete state;
        state = new InternalState(param);
        state->core.had_issue = false;
        state->core.message.clear();

        auto bestx = findBestPoint(x0);
        Vector x = Eigen::Map<Vector>(bestx.data(), static_cast<Eigen::Index>(bestx.size()));

        state->core.bfgs.reset(static_cast<int>(x.size()), param.m);
        state->core.xp.resize(x.size());
        state->core.grad.resize(x.size());
        state->core.gradp.resize(x.size());
        state->core.drt.resize(x.size());
        if (param.past > 0) state->core.fx_hist.resize(param.past);

        auto fun = [&](const Vector& xin, Vector& gradient) -> double { return fun_and_grad(xin, gradient); };

        double fx = fun(x, state->core.grad);
        state->core.gnorm = state->core.grad.norm();
        if (param.past > 0) state->core.fx_hist[0] = fx;

        if (state->core.gnorm <= param.epsilon || state->core.gnorm <= param.epsilon_rel * x.norm()) {
            minionResult = MinionResult(best, f_best, 1, Nevals, true, state->core.message);
            history.push_back(minionResult);
            auto ret = getBestFromHistory();
            ret.nfev = Nevals;
            return ret;
        }

        state->core.drt = -state->core.grad;
        double step = 1.0 / state->core.drt.norm();
        constexpr double eps = std::numeric_limits<double>::epsilon();
        Vector vecs(x.size()), vecy(x.size());
        int k = 1;

        try {
            for (;;) {
                state->core.xp = x;
                state->core.gradp = state->core.grad;
                double dg = state->core.grad.dot(state->core.drt);
                if (step < param.min_step) step = param.min_step;

                std::string ls_message;
                const bool ls_ok = LineSearch::search(
                    fun, param, state->core.xp, state->core.drt, param.max_step,
                    step, fx, state->core.grad, dg, x, ls_message);
                if (!ls_ok) {
                    state->core.had_issue = true;
                    state->core.message = ls_message;
                    break;
                }
                if (!ls_message.empty()) {
                    state->core.had_issue = true;
                    if (state->core.message.empty()) state->core.message = ls_message;
                }

                state->core.gnorm = state->core.grad.norm();
                if (state->core.gnorm <= param.epsilon || state->core.gnorm <= param.epsilon_rel * x.norm()) break;

                if (param.past > 0) {
                    const double fxd = state->core.fx_hist[k % param.past];
                    if (k >= param.past &&
                        std::abs(fxd - fx) <= param.delta * std::max(std::max(std::abs(fx), std::abs(fxd)), 1.0)) break;
                    state->core.fx_hist[k % param.past] = fx;
                }

                if (param.max_iterations != 0 && k >= param.max_iterations) break;

                vecs = x - state->core.xp;
                vecy = state->core.grad - state->core.gradp;
                if (vecs.dot(vecy) > eps * vecy.squaredNorm()) state->core.bfgs.add_correction(vecs, vecy);

                state->core.bfgs.apply_Hv(state->core.grad, -1.0, state->core.drt);
                if (!state->core.drt.allFinite() || state->core.drt.norm() == 0.0) {
                    state->core.had_issue = true;
                    state->core.message = "stopped: inverse-Hessian direction became invalid";
                    break;
                }
                step = 1.0;
                ++k;
            }
        } catch (const MaxevalExceedError&) {
            state->core.had_issue = true;
            state->core.message = "stopped: maximum evaluations exceeded";
        } catch (const std::exception& e) {
            state->core.had_issue = true;
            state->core.message = std::string("stopped: ") + e.what();
            std::cerr << "[Warning] " << e.what() << std::endl;
        } catch (...) {
            state->core.had_issue = true;
            state->core.message = "stopped: unknown error";
            std::cerr << "[Warning] Unknown error." << std::endl;
        }

        const bool success = !state->core.had_issue &&
            (state->core.gnorm <= param.epsilon || state->core.gnorm <= param.epsilon_rel * x.norm());
        minionResult = MinionResult(best, f_best, k, Nevals, success, state->core.message);
        history.push_back(minionResult);
        auto ret = getBestFromHistory();
        ret.nfev = Nevals;
        ret.success = success;
        ret.message = state->core.message;
        return ret;
    } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
    }
}

}  // namespace minion
