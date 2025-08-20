#include "l_bfgs.h"

namespace minion {

void L_BFGS::initialize() {
    hasInitialized=true;
}; 

double L_BFGS::fun_and_grad(const VectorXd& x, VectorXd& grad){
    if (Nevals >maxevals) throw MaxevalExceedError("Maxevals has been exceeded.");
    int m = std::max(std::ceil ((double (N_points)-1.0)/2.0), 1.0);
    std::vector<double> x_vec(x.data(), x.data() + x.size());
    std::vector<std::vector<double>> X ;
    X.push_back(x_vec);

    std::vector<double> hvec; 
    std::vector<double> sec_der = solver->getBFGS_mat().compute_hessian_diagonal(); 
    double ferr = last_f*func_noise_ratio;
    for (int i=0; i<x.size(); i++) {
        double h_min =  std::pow(epsilon, 0.5)*std::max(1.0, fabs(x[i])) ; 
        double h_max = 0.01*std::max(1.0, fabs(x[i])) ;
        double h_est = 2.0*sqrt(ferr/ fabs(sec_der[i]) );
        double h = std::min (h_max, std::max( h_min, h_est ) );
        hvec.push_back(h);
       // std::cout << h_est  << " ";
    }
    //std::cout << " \n";
    if (N_points == 1) {
        for (int i=0; i<x.size(); i++){
            std::vector<double> xp = x_vec;
            double h= hvec[i];
            xp[i] += h; 
            X.push_back(xp);
        };
    } else {
        for (int i=0; i<x.size(); i++) {
            for (int j=1; j<=m; j++) {
                double h= hvec[i];
                std::vector<double> xpp = x_vec;
                std::vector<double> xpm = x_vec;
                xpp[i] += j*h; 
                xpm[i] += -j*h;
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
    if (F[best_idx] < f_best){
        best = X[best_idx];
        f_best = F[best_idx];
        minionResult = MinionResult(best, f_best, 1, Nevals, false, "");
        history.push_back(minionResult);
    };

    std::vector<double> grad_vec; 
    if (N_points==1) {
        for (int i=0; i<x.size(); i++) grad_vec.push_back( (F[i+1]-f)/( X[i+1][i]-x[i] ) );
    } else {
        int k=1;
        for (int i=0; i <x.size(); i++){
            double h= hvec[i];
            double grad_val =0.0; 
            for (int j=1; j<=m; j++) {
                double cj = j/(m*(m+1)*(2.0*m+1));
                grad_val += cj*(F[k]-F[k+1]);
                k=k+2;
            };
            grad_val = grad_val*3.0/h;
            grad_vec.push_back(grad_val);
        };   
    };
    grad = Eigen::Map<Eigen::VectorXd> (grad_vec.data(), grad_vec.size());
    return f;
};

MinionResult L_BFGS::optimize() {
    try {
        history.clear();
        LBFGSpp::LBFGSParam<double> param;
        auto defaultKey = DefaultSettings().getDefaultSettings("L_BFGS");
        for (auto el : optionMap) defaultKey[el.first] = el.second;
        Options options(defaultKey);

        param.m              = options.get<int> ("m", 10); 
        param.epsilon        = options.get<double> ("g_epsilon", 1e-10); 
        param.epsilon_rel    = options.get<double> ("g_epsilon_rel", 1e-10); 
        param.past           = 3;
        param.delta          = options.get<double> ("f_reltol", 1e-20); 
        param.max_iterations = options.get<int> ("max_iterations", 0); 
        param.max_linesearch = options.get<int> ("max_linesearch", 20); 
        param.min_step       = 1e-20;
        param.max_step       = 1e+20;
        param.ftol           = options.get<double> ("c_1", 1e-4); 
        param.wolfe          = options.get<double> ("c_2", 0.9); 
        solver = new LBFGSpp::LBFGSSolver<double>(param);

        N_points             = options.get<int> ("N_points_derivative", 1); 
        func_noise_ratio = options.get<double> ("func_noise_ratio", 1e-10);
        Nevals=0;
        
        auto bestx = findBestPoint(x0);
        Eigen::VectorXd x=  Eigen::Map<Eigen::VectorXd> (bestx.data(), bestx.size());
        double final_f;
        int niter=0;
        auto fun = [&] (const Eigen::VectorXd& x, Eigen::VectorXd& grad) -> double {
            return  fun_and_grad(x, grad);
        };
        try {
            niter = solver->minimize ( fun, x, final_f);
        } catch (const MaxevalExceedError& e) {};

        minionResult = MinionResult(best, f_best, niter, Nevals, false, "");
        history.push_back(minionResult);
        auto ret = getBestFromHistory();
        ret.nfev = Nevals; 
        return ret;

    } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
    };
}

}