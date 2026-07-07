#ifndef TEST_FUNCTIONS_H
#define TEST_FUNCTIONS_H

#include <vector>
#include <cmath>

namespace minion {

// Function prototypes
std::vector<double> sphere(const std::vector<std::vector<double>>& X);
std::vector<double> rosenbrock(const std::vector<std::vector<double>>& X);
std::vector<double>rastrigin(const std::vector<std::vector<double>>& X);
std::vector<double>griewank(const std::vector<std::vector<double>>& X);
std::vector<double>ackley(const std::vector<std::vector<double>>& X);
std::vector<double>zakharov(const std::vector<std::vector<double>>& X);
std::vector<double>michalewicz(const std::vector<std::vector<double>>& X);
std::vector<double>levy(const std::vector<std::vector<double>>& X);
std::vector<double>dixon_price(const std::vector<std::vector<double>>& X);
std::vector<double>bent_cigar(const std::vector<std::vector<double>>& X);
std::vector<double>discus(const std::vector<std::vector<double>>& X);
std::vector<double>weierstrass(const std::vector<std::vector<double>>& X);
std::vector<double>happy_cat(const std::vector<std::vector<double>>& X);
std::vector<double>hgbat(const std::vector<std::vector<double>>& X);
std::vector<double>hcf(const std::vector<std::vector<double>>& X);
std::vector<double>grie_rosen(const std::vector<std::vector<double>>& X);
std::vector<double>easom(const std::vector<std::vector<double>>& X);
std::vector<double>drop_wave(const std::vector<std::vector<double>>& X);

}

#endif // OPTIMIZATION_FUNCTIONS_H
