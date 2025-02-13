#include "l_bfgs_b.h"

typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;

namespace minion {

void L_BFGS_B::initialize() {
    hasInitialized=true;
}; 

double L_BFGS_B::fun_and_grad(const VectorXd& x, VectorXd& grad){
    if (Nevals >maxevals) throw MaxevalExceedError("Maxevals has been exceeded.");

    std::vector<double> x_vec(x.data(), x.data() + x.size());
    std::vector<std::vector<double>> X ;
    X.push_back(x_vec);
    for (int i=0; i<x.size(); i++){
        std::vector<double> xp = x_vec;
        double h= fin_diff_rel_step * std::max(1.0, fabs(x[i])); 
        if (xp[i]+h > bounds[i].second ) h=-h; 
        xp[i] += h; 
        X.push_back(xp);
    };

    auto F = func(X, data);
    Nevals += F.size();
    double f = F[0];
    
    size_t best_idx = findArgMin(F);
    if (F[best_idx] < f_best){
        best = X[best_idx];
        f_best = F[best_idx];
        minionResult = MinionResult(best, f_best, 1, Nevals, false, "");
        history.push_back(minionResult);
    };

    std::vector<double> grad_vec; 
    for (int i=0; i<x.size(); i++) grad_vec.push_back( (F[i+1]-f)/( X[i+1][i]-x[i] ) );
    grad = Eigen::Map<Eigen::VectorXd> (grad_vec.data(), grad_vec.size());
    return f;
}

MinionResult L_BFGS_B::optimize() {
    try {
        history.clear();
        LBFGSpp::LBFGSBParam<double> param;
        auto defaultKey = DefaultSettings().getDefaultSettings("L_BFGS_B");
        for (auto el : optionMap) defaultKey[el.first] = el.second;
        Options options(defaultKey);

        param.m              = options.get<int> ("m", 10); 
        param.epsilon        = options.get<double> ("g_epsilon", 1e-10); 
        param.epsilon_rel    = options.get<double> ("g_epsilon_rel", 1e-10); 
        param.past           = 3;
        param.delta          = options.get<double> ("f_reltol", 1e-20); 
        param.max_iterations = options.get<int> ("max_iterations", 0); 
        param.max_submin     = 10;
        param.max_linesearch = options.get<int> ("max_linesearch", 20); 
        param.min_step       = 1e-20;
        param.max_step       = 1e+20;
        param.ftol           = options.get<double> ("c_1", 1e-4); 
        param.wolfe          = options.get<double> ("c_2", 0.9); 
        solver = new LBFGSpp::LBFGSBSolver<double>(param);
        
        double fd_relstep = options.get<double> ("finite_diff_rel_step", 0.0); 
        if (fd_relstep !=0.0) fin_diff_rel_step= fabs(std::min(fd_relstep, 1.0));
        Nevals=0;

        // Variable bounds
        VectorXd lb = Vector::Constant(bounds.size(), -10.0);
        VectorXd ub = Vector::Constant(bounds.size(), 10.0);

        for (int i =0; i<bounds.size(); i++){
            lb[i] = bounds[i].first;
            ub[i] = bounds[i].second;
        }; 

        if (x0.empty()){
            auto x0 = latin_hypercube_sampling(bounds, 1)[0]; 
        };

        Eigen::VectorXd x=  Eigen::Map<Eigen::VectorXd> (x0.data(), x0.size());
        double final_f;
        int niter=0;
        auto fun = [&] (const Eigen::VectorXd& x, Eigen::VectorXd& grad) -> double {
            return  fun_and_grad(x, grad);
        };
        try {
            niter = solver->minimize ( fun, x, final_f, lb, ub);
        } catch (const MaxevalExceedError& e) {};

        minionResult = MinionResult(best, f_best, niter, Nevals, false, "");
        history.push_back(minionResult);
        return getBestFromHistory();

    } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
    };
}

}