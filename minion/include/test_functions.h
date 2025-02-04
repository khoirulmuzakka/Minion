#ifndef TEST_FUNCTIONS_H
#define TEST_FUNCTIONS_H

#include <vector>
#include <cmath>

namespace minion {
// Define types for convenience
using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

// Function prototypes
Vector sphere(const Matrix& X);
Vector rosenbrock(const Matrix& X);
Vector rastrigin(const Matrix& X);
Vector griewank(const Matrix& X);
Vector ackley(const Matrix& X);
Vector zakharov(const Matrix& X);
Vector michalewicz(const Matrix& X);
Vector levy(const Matrix& X);
Vector dixon_price(const Matrix& X);
Vector bent_cigar(const Matrix& X);
Vector discus(const Matrix& X);
Vector weierstrass(const Matrix& X);
Vector happy_cat(const Matrix& X);
Vector hgbat(const Matrix& X);
Vector hcf(const Matrix& X);
Vector grie_rosen(const Matrix& X);
Vector easom(const Matrix& X);
Vector drop_wave(const Matrix& X);

}

#endif // OPTIMIZATION_FUNCTIONS_H
