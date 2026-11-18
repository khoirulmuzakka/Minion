#include "uncertainty.h"
#include <map>
#include <stdexcept>

namespace minion::uncertainty {
namespace {
struct Stencil {
    std::vector<double> first_nodes, first_weights, second_nodes, second_weights;
    std::string type;
    double step;
};
struct Curvature {
    Eigen::MatrixXd H;
    Eigen::VectorXd gradient, diagonal;
    std::vector<double> steps;
    std::vector<std::string> types;
    size_t nfev = 0, calls = 0;
};

std::vector<Curvature> differentiate(const MinionFunction& f, void* data, const std::vector<double>& x,
                       const std::vector<std::pair<double,double>>& bounds,
                       const std::vector<double>& base_steps, int batch_size, int n_scales, bool diagonal_only=false) {
    const size_t n = x.size();
    using Terms = std::vector<std::pair<size_t,double>>;
    struct Plan {
        std::vector<Stencil> stencils;
        std::vector<Terms> diagonal, gradients;
        std::vector<std::vector<Terms>> mixed;
    };
    std::vector<Plan> plans;
    std::map<std::vector<double>,size_t> indices;
    std::vector<std::vector<double>> points;
    for (int level=0; level<n_scales; ++level) {
        auto steps=base_steps;
        for (double& h:steps) h=std::ldexp(h,level);
        std::vector<Stencil> stencils;
        for (size_t i = 0; i < n; ++i) {
            double h = steps[i];
            const double up = bounds.empty() ? INFINITY : bounds[i].second - x[i];
            const double down = bounds.empty() ? INFINITY : x[i] - bounds[i].first;
            Stencil s;
            if (up >= h && down >= h) {
                s = {{-1,1}, {-0.5,0.5}, {-1,0,1}, {1,-2,1}, "central", h};
            } else {
                const double sign = up >= down ? 1.0 : -1.0;
                h = std::min(h, std::max(up, down) / 3.0);
                s = {{0,sign,2*sign}, {-1.5*sign,2*sign,-0.5*sign},
                     {0,sign,2*sign,3*sign}, {2,-5,4,-1},
                     sign > 0 ? "forward" : "backward", h};
            }
            if (!(h > 0) || !std::isfinite(h) || x[i] + h == x[i] || x[i] - h == x[i])
                throw std::invalid_argument("Hessian step is not representable at the supplied point.");
            stencils.push_back(s);
        }
        auto point_index = [&](size_t i, double a, size_t j, double b) {
            auto p = x;
            p[i] += a * stencils[i].step;
            if (b != 0) p[j] += b * stencils[j].step;
            for (size_t k = 0; k < n; ++k)
                if (!bounds.empty() && (p[k] < bounds[k].first || p[k] > bounds[k].second))
                    throw std::runtime_error("Hessian stencil falls outside bounds.");
            auto inserted = indices.emplace(p, points.size());
            if (inserted.second) points.push_back(p);
            return inserted.first->second;
        };
        std::vector<Terms> diagonal(n), gradients(n);
        std::vector<std::vector<Terms>> mixed;
        if (!diagonal_only) mixed.assign(n, std::vector<Terms>(n));
        point_index(0,0,0,0);
        for (size_t i=0; i<n; ++i) {
            const auto& s = stencils[i];
            for (size_t k=0;k<s.second_nodes.size();++k)
                diagonal[i].push_back({point_index(i,s.second_nodes[k],0,0),s.second_weights[k]/(s.step*s.step)});
            for (size_t k=0;k<s.first_nodes.size();++k)
                gradients[i].push_back({point_index(i,s.first_nodes[k],0,0),s.first_weights[k]/s.step});
            for (size_t j=0;!diagonal_only && j<i;++j) {
                const auto& t=stencils[j];
                for (size_t k=0;k<s.first_nodes.size();++k)
                    for (size_t l=0;l<t.first_nodes.size();++l)
                        mixed[i][j].push_back({point_index(i,s.first_nodes[k],j,t.first_nodes[l]),
                            s.first_weights[k]*t.first_weights[l]/(s.step*t.step)});
            }
        }
        plans.push_back({std::move(stencils),std::move(diagonal),std::move(gradients),std::move(mixed)});
    }
    size_t calls=0;
    std::vector<double> values;
    const size_t batch_limit = batch_size == -1 ? points.size() : static_cast<size_t>(batch_size);
    for (size_t start=0; start<points.size(); start+=batch_limit) {
        const size_t end=start+std::min(points.size()-start,batch_limit);
        std::vector<std::vector<double>> batch(points.begin()+start,points.begin()+end);
        auto v=f(batch,data);
        if (v.size()!=batch.size()) throw std::runtime_error("Objective returned an incorrect batch length.");
        for (double value:v) if (!std::isfinite(value)) throw std::runtime_error("Objective returned a non-finite value during Hessian estimation.");
        values.insert(values.end(),v.begin(),v.end());
        ++calls;
    }
    const size_t center=indices.at(x);
    std::vector<Curvature> results;
    for (const auto& plan:plans) {
        const auto& stencils=plan.stencils;
        const auto& diagonal=plan.diagonal;
        const auto& gradients=plan.gradients;
        const auto& mixed=plan.mixed;
        Curvature result;
        // Subtract the common objective offset before weighted summation.
        auto sum = [&](const Terms& terms) {
            long double value=0;
            for (const auto& term:terms) value+=static_cast<long double>(term.second)*(values[term.first]-values[center]);
            return static_cast<double>(value);
        };
        if (!diagonal_only) result.H=Eigen::MatrixXd::Zero(n,n);
        result.diagonal.resize(n);
        result.gradient.resize(n);
        for (size_t i=0;i<n;++i) {
            result.diagonal[i]=sum(diagonal[i]);
            if (!diagonal_only) result.H(i,i)=result.diagonal[i];
            result.gradient[i]=sum(gradients[i]);
            result.steps.push_back(stencils[i].step); result.types.push_back(stencils[i].type);
            for (size_t j=0;!diagonal_only && j<i;++j) result.H(i,j)=result.H(j,i)=sum(mixed[i][j]);
        }
        if (!result.H.allFinite() || !result.gradient.allFinite() || !result.diagonal.allFinite())
            throw std::runtime_error("Non-finite numerical derivatives.");
        result.nfev=points.size();
        result.calls=calls;
        results.push_back(std::move(result));
    }
    return results;
}

double normal_quantile(double probability) {
    double lo=-40,hi=40;
    for (int i=0;i<100;++i) {
        const double mid=(lo+hi)/2;
        if (0.5*std::erfc(-mid/std::sqrt(2.0)) < probability) lo=mid; else hi=mid;
    }
    return (lo+hi)/2;
}
}

Uncertainty::Uncertainty(MinionFunction objective, const std::vector<double>& optimum,
    const std::vector<std::pair<double,double>>& bounds, void* data)
    : objective_(std::move(objective)),x_(optimum),bounds_(bounds),data_(data) {
    if (!objective_ || x_.empty()) throw std::invalid_argument("An objective and nonempty optimum are required.");
    if (!bounds_.empty() && bounds_.size()!=x_.size()) throw std::invalid_argument("Bounds dimension must match optimum.");
    for (size_t i=0;i<x_.size();++i) {
        if (!std::isfinite(x_[i])) throw std::invalid_argument("Optimum must be finite.");
        if (!bounds_.empty() && (std::isnan(bounds_[i].first) || std::isnan(bounds_[i].second) ||
            !(bounds_[i].first<bounds_[i].second) || x_[i]<bounds_[i].first || x_[i]>bounds_[i].second))
            throw std::invalid_argument("Invalid bounds or optimum outside bounds.");
    }
}

HessianResult Uncertainty::hessian(double scale, const HessianOptions& o) const {
    if (!(scale>0) || !std::isfinite(scale) || !(o.relative_step>0) || !std::isfinite(o.relative_step) ||
        !(o.confidence_level>0 && o.confidence_level<1) ||
        !(o.eigenvalue_tolerance>0 && o.eigenvalue_tolerance<1) ||
        !(o.stability_tolerance>0) || !std::isfinite(o.stability_tolerance) ||
        (o.batch_size != -1 && o.batch_size <= 0) || o.n_scales < 1 || o.n_scales > 30 ||
        (o.probe_scales!=0 && (o.probe_scales<3 || o.probe_scales>30)))
        throw std::invalid_argument("Invalid Hessian options or scale.");
    const size_t n=x_.size();
    std::vector<double> scales(n),steps(n);
    if ((!o.step_sizes.empty() && o.step_sizes.size()!=n) ||
        (!o.parameter_scales.empty() && o.parameter_scales.size()!=n))
        throw std::invalid_argument("Steps and parameter scales must match optimum dimension.");
    for (size_t i=0;i<n;++i) {
        scales[i]=o.parameter_scales.empty()?std::max(1.0,std::fabs(x_[i])):o.parameter_scales[i];
        steps[i]=o.step_sizes.empty()?o.relative_step*scales[i]:o.step_sizes[i];
        if (!(scales[i]>0) || !std::isfinite(scales[i]) || !(steps[i]>0) || !std::isfinite(steps[i]))
            throw std::invalid_argument("Steps and parameter scales must be finite and positive.");
    }
    std::vector<std::string> probe_warnings;
    std::vector<double> probe_errors;
    size_t actual_nfev=0, actual_calls=0;
    std::map<std::vector<double>,double> cache;
    // Reuse axis evaluations from the one-shot probe in the final Hessian.
    MinionFunction cached_objective = [&](const std::vector<std::vector<double>>& points, void* data) {
        std::vector<std::vector<double>> missing;
        for (const auto& point:points) if (cache.find(point)==cache.end()) missing.push_back(point);
        if (!missing.empty()) {
            const auto values=objective_(missing,data);
            if (values.size()!=missing.size()) throw std::runtime_error("Objective returned an incorrect batch length.");
            for (double v:values) if (!std::isfinite(v)) throw std::runtime_error("Objective returned a non-finite value during Hessian estimation.");
            for (size_t i=0;i<missing.size();++i) cache.emplace(missing[i],values[i]);
            actual_nfev+=missing.size();++actual_calls;
        }
        std::vector<double> values;
        for (const auto& point:points) values.push_back(cache.at(point));
        return values;
    };
    if (o.probe_scales) {
        // Only O(n * probe_scales) axis points; no mixed derivatives here.
        // All candidate scales are planned before the single objective call.
        const auto probes=differentiate(cached_objective,data_,x_,bounds_,steps,-1,o.probe_scales,true);
        probe_errors.resize(n,std::numeric_limits<double>::infinity());
        for (size_t i=0;i<n;++i) {
            size_t chosen=0;
            double best=std::numeric_limits<double>::infinity();
            bool stable=false;
            double previous=std::numeric_limits<double>::infinity();
            for (size_t k=1;k<probes.size();++k) {
                if (probes[k].steps[i]<=probes[k-1].steps[i] ||
                    probes[k].types[i]!=probes[k-1].types[i]) {
                    previous=std::numeric_limits<double>::infinity();continue;
                }
                const double lhs=probes[k-1].diagonal[i], rhs=probes[k].diagonal[i];
                const double difference=lhs>0 && rhs>0 ?
                    std::fabs(lhs-rhs)/std::max(lhs,rhs) : std::numeric_limits<double>::infinity();
                // Require three distinct scales, not a single lucky pair.
                const double consistency=std::max(previous,difference);
                if (consistency<=o.stability_tolerance) {
                    // Prefer the largest stable step to reduce noise amplification.
                    chosen=k;probe_errors[i]=consistency;stable=true;
                } else if (!stable && difference<best) {
                    chosen=k;probe_errors[i]=difference;best=difference;
                }
                previous=difference;
            }
            if (!stable) {
                // With no resolved positive pair, try the largest available step,
                // but explicitly report that this selection was not validated.
                if (!std::isfinite(best)) chosen=probes.size()-1;
                probe_warnings.push_back("Step probes found no three distinct, consistent positive curvature scales for parameter "+std::to_string(i)+"; the selected step is unverified.");
            }
            steps[i]=std::ldexp(steps[i],static_cast<int>(chosen));
        }
        probe_warnings.push_back("Step probing checks coordinate curvature only; weak parameter combinations may remain unresolved.");
    }
    const auto candidates=differentiate(o.probe_scales ? cached_objective : objective_,data_,x_,bounds_,steps,o.batch_size,
        o.check_stability ? std::max(2,o.n_scales) : o.n_scales);
    Eigen::VectorXd s=Eigen::Map<Eigen::VectorXd>(scales.data(),n);
    size_t selected=0;
    double error=0, best=std::numeric_limits<double>::infinity();
    for (size_t k=1;k<candidates.size();++k) {
        const auto lhs=(s.asDiagonal()*candidates[k-1].H*s.asDiagonal()).eval();
        const auto rhs=(s.asDiagonal()*candidates[k].H*s.asDiagonal()).eval();
        const double difference=(lhs-rhs).norm()/std::max({lhs.norm(),rhs.norm(),std::numeric_limits<double>::min()});
        if (difference<best) {best=difference;selected=k;error=difference;}
        if (difference<=o.stability_tolerance) {selected=k;error=difference;break;}
    }
    const auto& a=candidates[selected];
    HessianResult r;
    r.estimate=x_;r.hessian=a.H;r.gradient=a.gradient;r.step_sizes=a.steps;r.stencil_types=a.types;
    r.nfev=o.probe_scales ? actual_nfev : a.nfev;
    r.objective_calls=o.probe_scales ? actual_calls : a.calls;
    r.confidence_level=o.confidence_level;
    r.step_probe_errors=probe_errors;r.warnings=probe_warnings;
    const Eigen::MatrixXd scaled=s.asDiagonal()*a.H*s.asDiagonal();
    r.differentiation_error=error;
    if (error>o.stability_tolerance)
        r.warnings.push_back("Hessian is sensitive to differentiation step sizes; no consistent adjacent candidate scales were found. Numerical noise or changing curvature may make uncertainty unreliable.");
    for (size_t i=0;i<n;++i) {
        if (a.types[i]!="central") r.warnings.push_back("One-sided differentiation used for parameter "+std::to_string(i)+".");
        if (!bounds_.empty() && std::min(x_[i]-bounds_[i].first,bounds_[i].second-x_[i])<=a.steps[i])
            r.warnings.push_back("Parameter "+std::to_string(i)+" is at or near a bound; Gaussian intervals may be inappropriate.");
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> raw(a.H), solver(scaled);
    if (raw.info()!=Eigen::Success || solver.info()!=Eigen::Success) {
        r.status="decomposition_failed";return r;
    }
    r.eigenvalues=raw.eigenvalues();r.eigenvectors=raw.eigenvectors();
    const auto eigen=solver.eigenvalues();
    r.scaled_eigenvalues=eigen;r.scaled_eigenvectors=solver.eigenvectors();
    const double largest=eigen.cwiseAbs().maxCoeff();
    const double threshold=o.eigenvalue_tolerance*largest;
    Eigen::VectorXd effective_eigen=eigen;
    Eigen::MatrixXd effective_vectors=solver.eigenvectors();
    if (o.force_positive_definite && eigen.minCoeff()<=threshold) {
        // Inspired by MINUIT2 MnPosDef: normalize by the diagonal, then
        // add to that normalized diagonal. Work in parameter-scaled units.
        // https://root.cern.ch/doc/master/MnPosDef_8cxx_source.html
        if (!(largest>0) || !std::isfinite(largest)) {
            r.status="singular";
            r.warnings.push_back("No curvature scale is available for positive-definite repair.");
            return r;
        }
        Eigen::MatrixXd repaired=scaled/largest;
        const double eps=std::max(o.eigenvalue_tolerance,
            2*std::sqrt(std::numeric_limits<double>::epsilon()));
        const double min_diagonal=repaired.diagonal().minCoeff();
        if (min_diagonal<=0) repaired.diagonal().array()+=0.5+eps-min_diagonal;
        const Eigen::VectorXd diagonal=repaired.diagonal();
        const Eigen::VectorXd inverse_root=diagonal.array().sqrt().inverse();
        const Eigen::MatrixXd normalized=inverse_root.asDiagonal()*repaired*inverse_root.asDiagonal();
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> normalized_solver(normalized);
        if (normalized_solver.info()!=Eigen::Success) {
            r.status="decomposition_failed";return r;
        }
        const double pmin=normalized_solver.eigenvalues().minCoeff();
        const double pmax=std::max(1.0,normalized_solver.eigenvalues().maxCoeff());
        if (pmin<=eps*pmax) {
            const double addition=std::max(0.001,2*eps)*pmax-pmin;
            repaired.diagonal().array()+=addition*diagonal.array();
        }
        repaired*=largest;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> repaired_solver(repaired);
        if (repaired_solver.info()!=Eigen::Success || !repaired_solver.eigenvalues().allFinite() ||
            repaired_solver.eigenvalues().minCoeff()<=0) {
            r.status="regularization_failed";return r;
        }
        // Also resolve modes below the API's scaled rank tolerance, even
        // when diagonal normalization alone makes them look well conditioned.
        const double target=2*o.eigenvalue_tolerance*repaired_solver.eigenvalues().maxCoeff();
        if (repaired_solver.eigenvalues().minCoeff()<=target) {
            repaired.diagonal().array()+=target-repaired_solver.eigenvalues().minCoeff();
            repaired_solver.compute(repaired);
            if (repaired_solver.info()!=Eigen::Success || !repaired_solver.eigenvalues().allFinite() ||
                repaired_solver.eigenvalues().minCoeff()<=0) {
                r.status="regularization_failed";return r;
            }
        }
        effective_eigen=repaired_solver.eigenvalues();
        effective_vectors=repaired_solver.eigenvectors();
        r.regularized_hessian=s.cwiseInverse().asDiagonal()*repaired*s.cwiseInverse().asDiagonal();
        if (!r.regularized_hessian.allFinite()) {r.status="regularization_failed";return r;}
        r.regularization_relative_change=(repaired/largest-scaled/largest).norm()/(scaled/largest).norm();
        r.forced_positive_definite=true;
        r.warnings.push_back("Hessian forced positive-definite using MINUIT-style diagonal inflation. Reported uncertainty is a regularized approximation; check step sizes and the optimum.");
    }
    double smallest_positive=std::numeric_limits<double>::infinity();
    Eigen::MatrixXd factors=Eigen::MatrixXd::Zero(n,n);
    for (size_t k=0;k<n;++k) {
        if (r.forced_positive_definite || eigen[k]>threshold) {
            ++r.retained_rank;smallest_positive=std::min(smallest_positive,effective_eigen[k]);
            factors.col(k)=s.asDiagonal()*effective_vectors.col(k)*std::sqrt(scale/effective_eigen[k]);
        } else r.discarded_modes.push_back(k);
    }
    const bool incomplete=!r.discarded_modes.empty();
    if (eigen.minCoeff() < -threshold)
        r.warnings.push_back("Hessian has negative curvature; this may reflect numerical differentiation (including too-small steps) or a point that is not a local minimum.");
    if (incomplete && (!o.discard_nonpositive || r.retained_rank==0)) {
        r.status=eigen.minCoeff() < -threshold ? "indefinite" : "singular";
        r.warnings.push_back(r.retained_rank==0 ?
            "No resolved positive Hessian modes; uncertainty is unavailable." :
            "Non-positive or unresolved Hessian modes found; strict mode leaves uncertainty unavailable.");
        return r;
    }
    if (incomplete) {
        r.warnings.push_back("Discarded "+std::to_string(r.discarded_modes.size())+
            " non-positive or unresolved Hessian modes. Reported covariance, standard uncertainties and intervals include only retained positive modes and are incomplete. Zero contributions from discarded modes do not mean zero uncertainty. Try larger differentiation steps and check the optimum.");
    } else {
        r.condition_number=effective_eigen.maxCoeff()/smallest_positive;
        if (r.condition_number>1/std::sqrt(o.eigenvalue_tolerance)) r.warnings.push_back("Scaled Hessian is poorly conditioned.");
    }
    r.covariance=factors*factors.transpose();
    r.covariance=(0.5*(r.covariance+r.covariance.transpose())).eval();
    if (!r.covariance.allFinite() || (r.covariance.diagonal().array()<0).any() ||
        (!incomplete && (r.covariance.diagonal().array()<=0).any())) {
        r.covariance.resize(0,0);r.status="invalid_covariance";return r;
    }
    r.std.resize(n);r.ci_lower.resize(n);r.ci_upper.resize(n);r.correlation.resize(n,n);
    const double z=normal_quantile(1-(1-o.confidence_level)/2);
    for (size_t i=0;i<n;++i) {
        r.std[i]=std::sqrt(r.covariance(i,i));
        r.ci_lower[i]=x_[i]-z*r.std[i];r.ci_upper[i]=x_[i]+z*r.std[i];
    }
    for(size_t i=0;i<n;++i) for(size_t j=0;j<n;++j)
        r.correlation(i,j)=r.std[i]>0 && r.std[j]>0 ?
            (r.covariance(i,j)/r.std[i])/r.std[j] : std::numeric_limits<double>::quiet_NaN();
    if ((r.gradient.array()*s.array()).matrix().norm()>1e-3*std::max(1.0,std::sqrt(scale*scaled.norm())))
        r.warnings.push_back("Supplied point may not be stationary.");
    r.covariance_available=true;r.uncertainty_complete=!incomplete;
    r.covariance_valid=!incomplete && !r.forced_positive_definite;
    r.status=r.forced_positive_definite?"forced_positive_definite":incomplete?"partial":(r.warnings.empty()?"ok":"warning");
    return r;
}
}
