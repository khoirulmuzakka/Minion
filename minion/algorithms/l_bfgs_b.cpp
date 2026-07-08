#include "l_bfgs_b.h"

#include <Eigen/Cholesky>
#include <Eigen/LU>

namespace minion {

namespace {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using IndexSet = std::vector<int>;

struct LBFGSBParam {
    int m = 10;
    double epsilon = 1e-10;
    double epsilon_rel = 1e-10;
    int past = 3;
    double delta = 1e-20;
    int max_iterations = 0;
    int max_submin = 10;
    int max_linesearch = 20;
    double min_step = 1e-20;
    double max_step = 1e20;
    double ftol = 1e-4;
    double wolfe = 0.9;
};

class CompactBFGSMatrix {
private:
    int n_ = 0;
    int m_ = 0;
    int ncorr_ = 0;
    double theta_ = 1.0;
    Matrix S_;
    Matrix Y_;
    Vector ys_;
    Matrix Minv_;
    Matrix M_;
    Eigen::LDLT<Matrix> Minv_solver_;

    void rebuild_compact_state()
    {
        if (ncorr_ <= 0) {
            Minv_.resize(0, 0);
            M_.resize(0, 0);
            return;
        }

        Matrix S = S_.leftCols(ncorr_);
        Matrix Y = Y_.leftCols(ncorr_);
        Matrix STY = S.transpose() * Y;
        Matrix STS = S.transpose() * S;

        Minv_.setZero(2 * ncorr_, 2 * ncorr_);
        Minv_.topLeftCorner(ncorr_, ncorr_) = (-ys_.head(ncorr_)).asDiagonal();
        Minv_.topRightCorner(ncorr_, ncorr_) = STY.triangularView<Eigen::StrictlyLower>().transpose();
        Minv_.bottomLeftCorner(ncorr_, ncorr_) = STY.triangularView<Eigen::StrictlyLower>();
        Minv_.bottomRightCorner(ncorr_, ncorr_) = theta_ * STS;

        Minv_solver_.compute(Minv_);
        M_ = Minv_solver_.solve(Matrix::Identity(2 * ncorr_, 2 * ncorr_));
    }

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
        Minv_.resize(0, 0);
        M_.resize(0, 0);
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
        rebuild_compact_state();
    }

    int num_corrections() const { return ncorr_; }
    double theta() const { return theta_; }

    void apply_Wtv(const Vector& v, Vector& res) const
    {
        res.resize(2 * ncorr_);
        if (ncorr_ <= 0) {
            return;
        }
        res.head(ncorr_) = Y_.leftCols(ncorr_).transpose() * v;
        res.tail(ncorr_) = theta_ * (S_.leftCols(ncorr_).transpose() * v);
    }

    void apply_Mv(const Vector& v, Vector& res) const
    {
        res.resize(2 * ncorr_);
        if (ncorr_ <= 0) {
            return;
        }
        res = M_ * v;
    }

    Vector W_row(int row) const
    {
        Vector res(2 * ncorr_);
        for (int j = 0; j < ncorr_; ++j) {
            res[j] = Y_(row, j);
            res[ncorr_ + j] = theta_ * S_(row, j);
        }
        return res;
    }

    Matrix W_rows(const IndexSet& rows, bool scale_second_half) const
    {
        Matrix res(rows.size(), 2 * ncorr_);
        if (ncorr_ <= 0 || rows.empty()) {
            res.setZero();
            return res;
        }
        for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
            for (int j = 0; j < ncorr_; ++j) {
                res(i, j) = Y_(rows[i], j);
                res(i, ncorr_ + j) = scale_second_half ? theta_ * S_(rows[i], j) : S_(rows[i], j);
            }
        }
        return res;
    }

    Matrix dense_B_subset(const IndexSet& rows) const
    {
        const int nsub = static_cast<int>(rows.size());
        Matrix B = theta_ * Matrix::Identity(nsub, nsub);
        if (ncorr_ <= 0 || nsub <= 0) {
            return B;
        }
        Matrix Wp = W_rows(rows, true);
        B.noalias() -= Wp * M_ * Wp.transpose();
        return B;
    }

    Matrix dense_cross_block(const IndexSet& left, const IndexSet& right) const
    {
        Matrix B = Matrix::Zero(left.size(), right.size());
        if (ncorr_ <= 0 || left.empty() || right.empty()) {
            return B;
        }
        Matrix Wl = W_rows(left, true);
        Matrix Wr = W_rows(right, true);
        B.noalias() = -(Wl * M_ * Wr.transpose());
        return B;
    }

    std::vector<double> compute_hessian_diagonal() const
    {
        std::vector<double> diagonal(static_cast<size_t>(n_), theta_);
        if (ncorr_ <= 0) {
            return diagonal;
        }
        for (int i = 0; i < n_; ++i) {
            Vector wi = W_row(i);
            diagonal[static_cast<size_t>(i)] = std::max(theta_ - wi.dot(M_ * wi), 1e-16);
        }
        return diagonal;
    }
};

class CauchyPoint {
private:
    static int search_greater(const Vector& brk, const IndexSet& ord, const double& t, int start = 0)
    {
        int i = start;
        for (; i < static_cast<int>(ord.size()); ++i) {
            if (brk[ord[i]] > t) {
                break;
            }
        }
        return i;
    }

public:
    static void compute(
        const CompactBFGSMatrix& bfgs, const Vector& x0, const Vector& g, const Vector& lb, const Vector& ub,
        Vector& xcp, Vector& vecc, IndexSet& newact_set, IndexSet& fv_set)
    {
        const int n = static_cast<int>(x0.size());
        xcp = x0;
        vecc = Vector::Zero(2 * bfgs.num_corrections());
        newact_set.clear();
        fv_set.clear();

        Vector brk(n), vecd(n);
        IndexSet ord;
        ord.reserve(n);
        const double inf = std::numeric_limits<double>::infinity();
        for (int i = 0; i < n; ++i) {
            if (lb[i] == ub[i]) {
                brk[i] = 0.0;
            } else if (g[i] < 0.0) {
                brk[i] = (x0[i] - ub[i]) / g[i];
            } else if (g[i] > 0.0) {
                brk[i] = (x0[i] - lb[i]) / g[i];
            } else {
                brk[i] = inf;
            }

            const bool iszero = (brk[i] == 0.0);
            vecd[i] = iszero ? 0.0 : -g[i];

            if (brk[i] == inf) {
                fv_set.push_back(i);
            } else if (!iszero) {
                ord.push_back(i);
            }
        }

        std::sort(ord.begin(), ord.end(), [&](int a, int b) { return brk[a] < brk[b]; });

        const int nord = static_cast<int>(ord.size());
        const int nfree = static_cast<int>(fv_set.size());
        if (nfree < 1 && nord < 1) {
            return;
        }

        Vector vecp;
        bfgs.apply_Wtv(vecd, vecp);
        double fp = -vecd.squaredNorm();
        Vector cache;
        bfgs.apply_Mv(vecp, cache);
        double fpp = -bfgs.theta() * fp - vecp.dot(cache);
        double deltatmin = -fp / fpp;

        double il = 0.0;
        int b = 0;
        double iu = (nord < 1) ? inf : brk[ord[b]];
        double deltat = iu - il;
        bool crossed_all = false;
        Vector wact(2 * bfgs.num_corrections());

        while (deltatmin >= deltat) {
            vecc.noalias() += deltat * vecp;
            const int act_begin = b;
            const int act_end = search_greater(brk, ord, iu, b) - 1;

            if ((nfree == 0) && (act_end == nord - 1)) {
                for (int i = act_begin; i <= act_end; ++i) {
                    const int act = ord[i];
                    xcp[act] = (vecd[act] > 0.0) ? ub[act] : lb[act];
                    newact_set.push_back(act);
                }
                crossed_all = true;
                break;
            }

            fp += deltat * fpp;
            for (int i = act_begin; i <= act_end; ++i) {
                const int act = ord[i];
                xcp[act] = (vecd[act] > 0.0) ? ub[act] : lb[act];
                const double zact = xcp[act] - x0[act];
                const double gact = g[act];
                const double ggact = gact * gact;
                wact = bfgs.W_row(act);
                bfgs.apply_Mv(wact, cache);
                fp += ggact + bfgs.theta() * gact * zact - gact * cache.dot(vecc);
                fpp -= (bfgs.theta() * ggact + 2.0 * gact * cache.dot(vecp) + ggact * cache.dot(wact));
                vecp.noalias() += gact * wact;
                vecd[act] = 0.0;
                newact_set.push_back(act);
            }

            deltatmin = -fp / fpp;
            il = iu;
            b = act_end + 1;
            if (b >= nord) {
                break;
            }
            iu = brk[ord[b]];
            deltat = iu - il;
        }

        const double eps = std::numeric_limits<double>::epsilon();
        if (fpp < eps) {
            deltatmin = -fp / eps;
        }

        if (!crossed_all) {
            deltatmin = std::max(deltatmin, 0.0);
            vecc.noalias() += deltatmin * vecp;
            const double tfinal = il + deltatmin;
            for (int coord : fv_set) {
                xcp[coord] = x0[coord] + tfinal * vecd[coord];
            }
            for (int i = b; i < nord; ++i) {
                const int coord = ord[i];
                xcp[coord] = x0[coord] + tfinal * vecd[coord];
                fv_set.push_back(coord);
            }
        }
    }
};

class SubspaceMinimization {
public:
    static void compute(
        const CompactBFGSMatrix& bfgs, const Vector& x0, const Vector& xcp, const Vector& g,
        const Vector& lb, const Vector& ub, const IndexSet& fv_set, Vector& drt)
    {
        drt = xcp - x0;
        const int nfree = static_cast<int>(fv_set.size());
        if (nfree < 1) {
            return;
        }

        Matrix Bff = bfgs.dense_B_subset(fv_set);
        Vector c(nfree);
        for (int i = 0; i < nfree; ++i) {
            c[i] = g[fv_set[i]];
        }

        std::vector<char> is_free(static_cast<size_t>(x0.size()), 0);
        for (int coord : fv_set) {
            is_free[static_cast<size_t>(coord)] = 1;
        }
        IndexSet active_set;
        active_set.reserve(static_cast<int>(x0.size()) - nfree);
        for (int i = 0; i < x0.size(); ++i) {
            if (!is_free[static_cast<size_t>(i)]) active_set.push_back(i);
        }
        if (!active_set.empty()) {
            Matrix Bfa = bfgs.dense_cross_block(fv_set, active_set);
            Vector da(active_set.size());
            for (int i = 0; i < static_cast<int>(active_set.size()); ++i) {
                da[i] = drt[active_set[i]];
            }
            c.noalias() += Bfa * da;
        }

        Eigen::LDLT<Matrix> solver(Bff);
        Vector y = solver.solve(-c);
        if (solver.info() != Eigen::Success) {
            return;
        }

        Vector lower(nfree), upper(nfree);
        for (int i = 0; i < nfree; ++i) {
            const int coord = fv_set[i];
            lower[i] = lb[coord] - x0[coord];
            upper[i] = ub[coord] - x0[coord];
        }

        if ((y.array() >= lower.array()).all() && (y.array() <= upper.array()).all()) {
            for (int i = 0; i < nfree; ++i) drt[fv_set[i]] = y[i];
            return;
        }

        Vector yfallback = y;
        Vector lambda = Vector::Zero(nfree);
        Vector mu = Vector::Zero(nfree);

        IndexSet L, U, P;
        IndexSet yL, yU, yP;
        for (int iter = 0; iter < 10; ++iter) {
            L.clear(); U.clear(); P.clear();
            yL.clear(); yU.clear(); yP.clear();

            for (int i = 0; i < nfree; ++i) {
                if ((y[i] < lower[i]) || (y[i] == lower[i] && lambda[i] >= 0.0)) {
                    L.push_back(fv_set[i]);
                    yL.push_back(i);
                    y[i] = lower[i];
                    mu[i] = 0.0;
                } else if ((y[i] > upper[i]) || (y[i] == upper[i] && mu[i] >= 0.0)) {
                    U.push_back(fv_set[i]);
                    yU.push_back(i);
                    y[i] = upper[i];
                    lambda[i] = 0.0;
                } else {
                    P.push_back(fv_set[i]);
                    yP.push_back(i);
                    lambda[i] = 0.0;
                    mu[i] = 0.0;
                }
            }

            if (!P.empty()) {
                Vector rhs(yP.size());
                for (int i = 0; i < static_cast<int>(yP.size()); ++i) rhs[i] = c[yP[i]];

                if (!L.empty()) {
                    Matrix Bpl(yP.size(), yL.size());
                    for (int i = 0; i < static_cast<int>(yP.size()); ++i)
                        for (int j = 0; j < static_cast<int>(yL.size()); ++j)
                            Bpl(i, j) = Bff(yP[i], yL[j]);
                    Vector lvec(yL.size());
                    for (int i = 0; i < static_cast<int>(yL.size()); ++i) lvec[i] = lower[yL[i]];
                    rhs.noalias() += Bpl * lvec;
                }

                if (!U.empty()) {
                    Matrix Bpu(yP.size(), yU.size());
                    for (int i = 0; i < static_cast<int>(yP.size()); ++i)
                        for (int j = 0; j < static_cast<int>(yU.size()); ++j)
                            Bpu(i, j) = Bff(yP[i], yU[j]);
                    Vector uvec(yU.size());
                    for (int i = 0; i < static_cast<int>(yU.size()); ++i) uvec[i] = upper[yU[i]];
                    rhs.noalias() += Bpu * uvec;
                }

                Matrix Bpp(yP.size(), yP.size());
                for (int i = 0; i < static_cast<int>(yP.size()); ++i)
                    for (int j = 0; j < static_cast<int>(yP.size()); ++j)
                        Bpp(i, j) = Bff(yP[i], yP[j]);
                Eigen::LDLT<Matrix> subsolver(Bpp);
                if (subsolver.info() != Eigen::Success) break;
                Vector yp = subsolver.solve(-rhs);
                for (int i = 0; i < static_cast<int>(yP.size()); ++i) y[yP[i]] = yp[i];
            }

            if (!L.empty()) {
                Vector yfull = y;
                Vector Bl = Vector::Zero(yL.size());
                for (int i = 0; i < static_cast<int>(yL.size()); ++i) {
                    for (int j = 0; j < nfree; ++j) Bl[i] += Bff(yL[i], j) * yfull[j];
                    lambda[yL[i]] = Bl[i] + c[yL[i]];
                }
            }
            if (!U.empty()) {
                Vector yfull = y;
                Vector Bu = Vector::Zero(yU.size());
                for (int i = 0; i < static_cast<int>(yU.size()); ++i) {
                    for (int j = 0; j < nfree; ++j) Bu[i] += Bff(yU[i], j) * yfull[j];
                    mu[yU[i]] = -(Bu[i] + c[yU[i]]);
                }
            }

            bool L_ok = true, U_ok = true, P_ok = true;
            for (int idx : yL) if (lambda[idx] < 0.0) { L_ok = false; break; }
            for (int idx : yU) if (mu[idx] < 0.0) { U_ok = false; break; }
            for (int idx : yP) if (y[idx] < lower[idx] || y[idx] > upper[idx]) { P_ok = false; break; }
            if (L_ok && U_ok && P_ok) {
                for (int i = 0; i < nfree; ++i) drt[fv_set[i]] = y[i];
                return;
            }
        }

        y = y.cwiseMax(lower).cwiseMin(upper);
        for (int i = 0; i < nfree; ++i) drt[fv_set[i]] = y[i];
        if (drt.dot(g) <= -std::numeric_limits<double>::epsilon()) return;

        y = yfallback.cwiseMax(lower).cwiseMin(upper);
        for (int i = 0; i < nfree; ++i) drt[fv_set[i]] = y[i];
        if (drt.dot(g) <= -std::numeric_limits<double>::epsilon()) return;

        for (int i = 0; i < nfree; ++i) drt[fv_set[i]] = yfallback[i];
    }
};

class LineSearch {
private:
    struct SearchState {
        bool brackt = false;
        int stage = 1;
        double finit = 0.0;
        double ginit = 0.0;
        double gtest = 0.0;
        double gx = 0.0;
        double gy = 0.0;
        double fx = 0.0;
        double fy = 0.0;
        double stx = 0.0;
        double sty = 0.0;
        double stmin = 0.0;
        double stmax = 0.0;
        double width = 0.0;
        double width1 = 0.0;
    };

    static void dcstep(double& stx, double& fx, double& dx,
                       double& sty, double& fy, double& dy,
                       double& stp, double fp, double dp, bool& brackt,
                       double stpmin, double stpmax)
    {
        double gamma, p, q, r, s, sgnd, stpc, stpf, stpq, theta;
        sgnd = dp * (dx / std::fabs(dx));

        if (fp > fx) {
            theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
            s = std::max(std::fabs(theta), std::max(std::fabs(dx), std::fabs(dp)));
            gamma = s * std::sqrt(std::max(0.0, (theta / s) * (theta / s) - (dx / s) * (dp / s)));
            if (stp < stx) gamma = -gamma;
            p = (gamma - dx) + theta;
            q = ((gamma - dx) + gamma) + dp;
            r = p / q;
            stpc = stx + r * (stp - stx);
            stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.0) * (stp - stx);
            stpf = (std::fabs(stpc - stx) < std::fabs(stpq - stx)) ? stpc : stpc + (stpq - stpc) / 2.0;
            brackt = true;
        } else if (sgnd < 0.0) {
            theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
            s = std::max(std::fabs(theta), std::max(std::fabs(dx), std::fabs(dp)));
            gamma = s * std::sqrt(std::max(0.0, (theta / s) * (theta / s) - (dx / s) * (dp / s)));
            if (stp > stx) gamma = -gamma;
            p = (gamma - dp) + theta;
            q = ((gamma - dp) + gamma) + dx;
            r = p / q;
            stpc = stp + r * (stx - stp);
            stpq = stp + (dp / (dp - dx)) * (stx - stp);
            stpf = (std::fabs(stpc - stp) > std::fabs(stpq - stp)) ? stpc : stpq;
            brackt = true;
        } else if (std::fabs(dp) < std::fabs(dx)) {
            theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
            s = std::max(std::fabs(theta), std::max(std::fabs(dx), std::fabs(dp)));
            gamma = s * std::sqrt(std::max(0.0, (theta / s) * (theta / s) - (dx / s) * (dp / s)));
            if (stp > stx) gamma = -gamma;
            p = (gamma - dp) + theta;
            q = (gamma + (dx - dp)) + gamma;
            r = p / q;
            if ((r < 0.0) && (gamma != 0.0)) {
                stpc = stp + r * (stx - stp);
            } else if (stp > stx) {
                stpc = stpmax;
            } else {
                stpc = stpmin;
            }
            stpq = stp + (dp / (dp - dx)) * (stx - stp);

            if (brackt) {
                stpf = (std::fabs(stpc - stp) < std::fabs(stpq - stp)) ? stpc : stpq;
                if (stp > stx) {
                    stpf = std::min(stp + 0.66 * (sty - stp), stpf);
                } else {
                    stpf = std::max(stp + 0.66 * (sty - stp), stpf);
                }
            } else {
                stpf = (std::fabs(stpc - stp) > std::fabs(stpq - stp)) ? stpc : stpq;
                stpf = std::min(stpmax, stpf);
                stpf = std::max(stpmin, stpf);
            }
        } else {
            if (brackt) {
                theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp;
                s = std::max(std::fabs(theta), std::max(std::fabs(dy), std::fabs(dp)));
                gamma = s * std::sqrt(std::max(0.0, (theta / s) * (theta / s) - (dy / s) * (dp / s)));
                if (stp > sty) gamma = -gamma;
                p = (gamma - dp) + theta;
                q = ((gamma - dp) + gamma) + dy;
                r = p / q;
                stpf = stp + r * (sty - stp);
            } else if (stp > stx) {
                stpf = stpmax;
            } else {
                stpf = stpmin;
            }
        }

        if (fp > fx) {
            sty = stp;
            fy = fp;
            dy = dp;
        } else {
            if (sgnd < 0.0) {
                sty = stx;
                fy = fx;
                dy = dx;
            }
            stx = stp;
            fx = fp;
            dx = dp;
        }
        stp = stpf;
    }

public:
    template <typename Foo>
    static bool search(Foo& f, const LBFGSBParam& param, const Vector& xp, const Vector& drt, double step_max,
                       double& step, double& fx, Vector& grad, double& dg, Vector& x, std::string& message)
    {
        if (step <= 0.0) {
            message = "line search skipped: initial step must be positive";
            return false;
        }
        if (step > step_max) {
            message = "line search skipped: initial step exceeds feasible maximum";
            return false;
        }
        if (dg >= 0.0) {
            message = "line search skipped: moving direction does not decrease the objective function value";
            return false;
        }

        const double xtol = 0.1;
        SearchState s;
        s.finit = fx;
        s.ginit = dg;
        s.gtest = param.ftol * s.ginit;
        s.gx = s.ginit;
        s.gy = s.ginit;
        s.fx = s.finit;
        s.fy = s.finit;
        s.stmax = step + 4.0 * step;
        s.width = step_max - param.min_step;
        s.width1 = s.width / 0.5;

        double best_step = 0.0;
        double best_fx = fx;
        double best_dg = dg;
        Vector best_x = xp;
        Vector best_grad = grad;

        step = std::clamp(step, param.min_step, std::min(step_max, param.max_step));
        for (int iter = 0; iter < param.max_linesearch; ++iter) {
            x.noalias() = xp + step * drt;
            fx = f(x, grad);
            dg = grad.dot(drt);

            if (!std::isfinite(fx) || !std::isfinite(dg)) {
                step = s.brackt ? (s.stx + 0.5 * (s.sty - s.stx)) : std::max(param.min_step, 0.5 * step);
                continue;
            }

            const double ftest = s.finit + step * s.gtest;
            if ((s.stage == 1) && (fx <= ftest) && (dg >= 0.0)) {
                s.stage = 2;
            }
            if ((fx <= ftest) && (std::fabs(dg) <= param.wolfe * (-s.ginit))) {
                return true;
            }

            if (fx < best_fx || (fx == best_fx && std::fabs(dg) < std::fabs(best_dg))) {
                best_step = step;
                best_fx = fx;
                best_dg = dg;
                best_x = x;
                best_grad = grad;
            }

            if ((s.brackt && ((step <= s.stmin) || (step >= s.stmax))) ||
                (s.brackt && ((s.stmax - s.stmin) <= xtol * s.stmax)) ||
                (step == step_max && (fx <= ftest) && (dg <= s.gtest)) ||
                (step == param.min_step && ((fx > ftest) || (dg >= s.gtest)))) {
                break;
            }

            if ((s.stage == 1) && (fx <= s.fx) && (fx > ftest)) {
                double fm = fx - step * s.gtest;
                double fxm = s.fx - s.stx * s.gtest;
                double fym = s.fy - s.sty * s.gtest;
                double gm = dg - s.gtest;
                double gxm = s.gx - s.gtest;
                double gym = s.gy - s.gtest;
                dcstep(s.stx, fxm, gxm, s.sty, fym, gym, step, fm, gm, s.brackt, s.stmin, s.stmax);
                s.fx = fxm + s.stx * s.gtest;
                s.fy = fym + s.sty * s.gtest;
                s.gx = gxm + s.gtest;
                s.gy = gym + s.gtest;
            } else {
                dcstep(s.stx, s.fx, s.gx, s.sty, s.fy, s.gy, step, fx, dg, s.brackt, s.stmin, s.stmax);
            }

            if (s.brackt) {
                if (std::fabs(s.sty - s.stx) >= 0.66 * s.width1) {
                    step = s.stx + 0.5 * (s.sty - s.stx);
                }
                s.width1 = s.width;
                s.width = std::fabs(s.sty - s.stx);
                s.stmin = std::min(s.stx, s.sty);
                s.stmax = std::max(s.stx, s.sty);
            } else {
                s.stmin = step + 1.1 * (step - s.stx);
                s.stmax = step + 4.0 * (step - s.stx);
            }

            step = std::clamp(step, param.min_step, std::min(step_max, param.max_step));
            if ((s.brackt && ((step <= s.stmin) || (step >= s.stmax))) ||
                (s.brackt && ((s.stmax - s.stmin) <= xtol * s.stmax))) {
                step = s.stx;
            }
        }

        if (best_step > 0.0) {
            step = best_step;
            fx = best_fx;
            dg = best_dg;
            x = best_x;
            grad = best_grad;
            if (message.empty()) {
                message = "line search reached fallback step";
            }
        } else {
            x = xp;
            fx = s.finit;
            dg = s.ginit;
            message = "line search failed to find a usable step";
            return false;
        }
        return true;
    }
};

struct SolverCore {
    LBFGSBParam param;
    CompactBFGSMatrix bfgs;
    Vector fx_hist;
    Vector xp;
    Vector grad;
    Vector gradp;
    Vector drt;
    double projgnorm = 0.0;
    bool had_issue = false;
    std::string message;

    explicit SolverCore(const LBFGSBParam& p) : param(p) {}
};

static void force_bounds(Vector& x, const Vector& lb, const Vector& ub)
{
    x = x.cwiseMax(lb).cwiseMin(ub);
}

static double proj_grad_norm(const Vector& x, const Vector& g, const Vector& lb, const Vector& ub)
{
    return ((x - g).cwiseMax(lb).cwiseMin(ub) - x).cwiseAbs().maxCoeff();
}

static double max_step_size(const Vector& x0, const Vector& drt, const Vector& lb, const Vector& ub)
{
    double step = std::numeric_limits<double>::infinity();
    for (int i = 0; i < x0.size(); ++i) {
        if (drt[i] > 0.0) {
            step = std::min(step, (ub[i] - x0[i]) / drt[i]);
        } else if (drt[i] < 0.0) {
            step = std::min(step, (lb[i] - x0[i]) / drt[i]);
        }
    }
    return step;
}

}  // namespace

struct L_BFGS_B::InternalState {
    SolverCore core;

    explicit InternalState(const LBFGSBParam& param) : core(param) {}
};

L_BFGS_B::~L_BFGS_B() {
    delete state;
    state = nullptr;
}

void L_BFGS_B::initialize() {
    hasInitialized = true;
}

double L_BFGS_B::fun_and_grad(const VectorXd& x, VectorXd& grad){
    if (Nevals > maxevals) throw MaxevalExceedError("Maxevals has been exceeded.");
    if (!state) throw std::runtime_error("L_BFGS_B internal state is not initialized.");

    const int m = std::min(std::max(static_cast<int>(std::ceil((static_cast<double>(N_points) - 1.0) / 2.0)), 1), 8);
    std::vector<double> x_vec(x.data(), x.data() + x.size());
    std::vector<std::vector<double>> X;
    X.push_back(x_vec);

    std::vector<double> hvec;
    std::vector<double> sec_der = state->core.bfgs.compute_hessian_diagonal();
    const double ferr = std::max(std::fabs(last_f), 1.0) * func_noise_ratio;

    for (int i = 0; i < x.size(); i++) {
        double d_low = x[i] - actual_bounds[i].first;
        double d_high = actual_bounds[i].second - x[i];
        double h_low_corrected = std::max(0.0, d_low / m - 1e-16);
        double h_high_corrected = std::max(0.0, d_high / m - 1e-16);
        double h_max_from_bound = std::min(h_low_corrected, h_high_corrected);
        double h_min = std::pow(epsilon, 0.5) * std::max(1.0, std::fabs(x[i]));
        double h_max = 0.01 * std::max(1.0, std::fabs(x[i]));
        double curvature = std::max(std::fabs(sec_der[static_cast<size_t>(i)]), 1e-16);
        double h_est = 2.0 * std::sqrt(std::max(ferr, 1e-32) / curvature);
        double h = std::min(h_max, std::max(h_min, h_est));
        h = std::min(h, h_max_from_bound);
        if (h <= 0.0) {
            h = std::min(h_max, std::max(h_min, fin_diff_rel_step * std::max(1.0, std::fabs(x[i]))));
        }
        hvec.push_back(h);
    }

    if (N_points == 1) {
        for (int i = 0; i < x.size(); i++) {
            std::vector<double> xp = x_vec;
            double h = hvec[static_cast<size_t>(i)];
            if (xp[i] + h > bounds[i].second) h = -h;
            xp[i] += h;
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
        updateBestSoFar(minionResult);
    }

    std::vector<double> grad_vec;
    if (N_points == 1) {
        for (int i = 0; i < x.size(); i++) {
            grad_vec.push_back((F[static_cast<size_t>(i + 1)] - f) / (X[static_cast<size_t>(i + 1)][i] - x[i]));
        }
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

MinionResult L_BFGS_B::optimize() {
    try {
        resetBestSoFar();
        auto defaultKey = DefaultSettings().getDefaultSettings("L_BFGS_B");
        for (auto el : optionMap) defaultKey[el.first] = el.second;
        Options options(defaultKey);

        LBFGSBParam param;
        param.m              = options.get<int>("m", 10);
        param.epsilon        = options.get<double>("g_epsilon", 1e-10);
        param.epsilon_rel    = options.get<double>("g_epsilon_rel", 1e-10);
        param.past           = 3;
        param.delta          = options.get<double>("f_reltol", 1e-20);
        param.max_iterations = options.get<int>("max_iterations", 0);
        param.max_submin     = 10;
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

        Vector lb(bounds.size()), ub(bounds.size());
        for (int i = 0; i < bounds.size(); i++) {
            lb[i] = bounds[i].first;
            ub[i] = bounds[i].second;
        }

        if (x0.empty()) {
            x0 = latin_hypercube_sampling(bounds, 1);
        }

        std::vector<double> xinit = (x0.size() == 1) ? x0[0] : findBestPoint(x0);
        Vector x = Eigen::Map<Vector>(xinit.data(), static_cast<Eigen::Index>(xinit.size()));
        force_bounds(x, lb, ub);

        state->core.bfgs.reset(static_cast<int>(x.size()), param.m);
        state->core.xp.resize(x.size());
        state->core.grad.resize(x.size());
        state->core.gradp.resize(x.size());
        state->core.drt.resize(x.size());
        if (param.past > 0) state->core.fx_hist.resize(param.past);

        auto fun = [&](const Vector& xin, Vector& gradient) -> double { return fun_and_grad(xin, gradient); };

        double fx = fun(x, state->core.grad);
        state->core.projgnorm = proj_grad_norm(x, state->core.grad, lb, ub);
        if (param.past > 0) state->core.fx_hist[0] = fx;

        if (state->core.projgnorm <= param.epsilon || state->core.projgnorm <= param.epsilon_rel * x.norm()) {
            minionResult = MinionResult(best, f_best, 1, Nevals, true, state->core.message);
            updateBestSoFar(minionResult);
            auto ret = getBestSoFar();
            ret.nfev = Nevals;
            return ret;
        }

        Vector xcp, vecc;
        IndexSet newact_set, fv_set;
        CauchyPoint::compute(state->core.bfgs, x, state->core.grad, lb, ub, xcp, vecc, newact_set, fv_set);
        state->core.drt = xcp - x;
        if (state->core.drt.norm() > 0.0) state->core.drt.normalize();

        constexpr double eps = std::numeric_limits<double>::epsilon();
        Vector vecs(x.size()), vecy(x.size());
        int k = 1;

        try {
            for (;;) {
                state->core.xp = x;
                state->core.gradp = state->core.grad;
                double dg = state->core.grad.dot(state->core.drt);
                double step_max = max_step_size(x, state->core.drt, lb, ub);

                if (dg >= 0.0 || step_max <= param.min_step) {
                    state->core.drt = xcp - x;
                    state->core.bfgs.reset(static_cast<int>(x.size()), param.m);
                    dg = state->core.grad.dot(state->core.drt);
                    step_max = max_step_size(x, state->core.drt, lb, ub);
                    if (dg >= 0.0 || step_max <= param.min_step) {
                        state->core.had_issue = true;
                        state->core.message = (step_max <= param.min_step)
                            ? "stopped: no feasible descent step from current iterate"
                            : "stopped: search direction is not a descent direction";
                        break;
                    }
                }

                step_max = std::min(param.max_step, step_max);
                double step = std::min(1.0, step_max);
                std::string ls_message;
                const bool ls_ok = LineSearch::search(
                    fun, param, state->core.xp, state->core.drt, step_max,
                    step, fx, state->core.grad, dg, x, ls_message);
                if (!ls_ok) {
                    state->core.had_issue = true;
                    state->core.message = ls_message;
                    break;
                }
                if (!ls_message.empty()) {
                    state->core.had_issue = true;
                    if (state->core.message.empty()) {
                        state->core.message = ls_message;
                    }
                }

                state->core.projgnorm = proj_grad_norm(x, state->core.grad, lb, ub);
                if (state->core.projgnorm <= param.epsilon || state->core.projgnorm <= param.epsilon_rel * x.norm()) break;

                if (param.past > 0) {
                    double fxd = state->core.fx_hist[k % param.past];
                    if (k >= param.past &&
                        std::abs(fxd - fx) <= param.delta * std::max(std::max(std::abs(fx), std::abs(fxd)), 1.0)) break;
                    state->core.fx_hist[k % param.past] = fx;
                }

                if (param.max_iterations != 0 && k >= param.max_iterations) break;

                vecs = x - state->core.xp;
                vecy = state->core.grad - state->core.gradp;
                if (vecs.dot(vecy) > eps * vecy.squaredNorm()) state->core.bfgs.add_correction(vecs, vecy);

                force_bounds(x, lb, ub);
                CauchyPoint::compute(state->core.bfgs, x, state->core.grad, lb, ub, xcp, vecc, newact_set, fv_set);
                SubspaceMinimization::compute(state->core.bfgs, x, xcp, state->core.grad, lb, ub, fv_set, state->core.drt);
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
            (state->core.projgnorm <= param.epsilon || state->core.projgnorm <= param.epsilon_rel * x.norm());
        minionResult = MinionResult(best, f_best, k, Nevals, success, state->core.message);
        updateBestSoFar(minionResult);
        auto ret = getBestSoFar();
        ret.nfev = Nevals;
        ret.success = success;
        ret.message = state->core.message;
        return ret;
    } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
    }
}

}  // namespace minion
