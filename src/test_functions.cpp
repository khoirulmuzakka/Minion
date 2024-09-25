
#include <vector>
#include <cmath>
#include <numeric>
#include "test_functions.h"

namespace minion {

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Function to compute the sum of squares of each row in X (Sphere function)
Vector sphere(const Matrix& X) {
    Vector result(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        double sum_squares = 0.0;
        for (size_t j = 0; j < X[i].size(); ++j) {
            sum_squares += X[i][j] * X[i][j];
        }
        result[i] = sum_squares;
    }
    return result;
}

// Function to compute the Rosenbrock function
Vector rosenbrock(const Matrix& X) {
    Vector result(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < X[i].size() - 1; ++j) {
            double temp1 = 100.0 * (X[i][j + 1] - X[i][j] * X[i][j]);
            double temp2 = (1 - X[i][j]);
            sum += temp1 * temp1 + temp2 * temp2;
        }
        result[i] = sum;
    }
    return result;
}

// Function to compute the Rastrigin function
Vector rastrigin(const Matrix& X) {
    Vector result(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < X[i].size(); ++j) {
            sum += X[i][j] * X[i][j] - 10.0 * cos(2.0 * M_PI * X[i][j]) + 10.0;
        }
        result[i] = sum;
    }
    return result;
}

// Function to compute the Griewank function
Vector griewank(const Matrix& X) {
    Vector result(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        double sum1 = 0.0;
        double prod = 1.0;
        for (size_t j = 0; j < X[i].size(); ++j) {
            sum1 += X[i][j] * X[i][j];
            prod *= cos(X[i][j] / sqrt(j + 1));
        }
        result[i] = sum1 / 4000.0 - prod + 1.0;
    }
    return result;
}

// Function to compute the Ackley function
Vector ackley(const Matrix& X) {
    Vector result(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        double sum1 = 0.0;
        double sum2 = 0.0;
        for (size_t j = 0; j < X[i].size(); ++j) {
            sum1 += X[i][j] * X[i][j];
            sum2 += cos(2.0 * M_PI * X[i][j]);
        }
        result[i] = -20.0 * exp(-0.2 * sqrt(sum1 / X[i].size())) - exp(sum2 / X[i].size()) + 20.0 + exp(1.0);
    }
    return result;
}

// Function to compute the Zakharov function
Vector zakharov(const Matrix& X) {
    Vector result(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        double sum1 = 0.0;
        double sum2 = 0.0;
        for (size_t j = 0; j < X[i].size(); ++j) {
            sum1 += X[i][j] * X[i][j];
            sum2 += 0.5 * (j + 1) * X[i][j];
        }
        result[i] = sum1 + sum2 * sum2 + sum2 * sum2 * sum2;
    }
    return result;
}

// Function to compute the Michalewicz function
Vector michalewicz(const Matrix& X) {
    Vector result(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < X[i].size(); ++j) {
            sum += sin(X[i][j]) * pow(sin((j + 1) * X[i][j] * X[i][j] / M_PI), 20);
        }
        result[i] = -sum;
    }
    return result;
}

// Function to compute the Levy function
Vector levy(const Matrix& X) {
    Vector result(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        double term1 = pow(sin(M_PI * (1 + (X[i][0] - 1) / 4)), 2);
        double term2 = 0.0;
        double term3 = pow(sin(2 * M_PI * (X[i].back() - 1)), 2);
        
        for (size_t j = 0; j < X[i].size() - 1; ++j) {
            double wj = 1 + (X[i][j] - 1) / 4;
            term2 += pow(wj - 1, 2) * (1 + 10 * pow(sin(M_PI * wj + 1), 2));
        }
        
        result[i] = term1 + term2 + term3;
    }
    return result;
}

// Function to compute the Dixon-Price function
Vector dixon_price(const Matrix& X) {
    Vector result(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        double term1 = pow(X[i][0] - 1, 2);
        double sum = 0.0;
        for (size_t j = 1; j < X[i].size(); ++j) {
            sum += j * pow(2 * X[i][j] * X[i][j] - X[i][j - 1], 2);
        }
        result[i] = term1 + sum;
    }
    return result;
}

// Function to compute the Bent Cigar function
Vector bent_cigar(const Matrix& X) {
    Vector result(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        double sum_squares = 0.0;
        for (size_t j = 1; j < X[i].size(); ++j) {
            sum_squares += X[i][j] * X[i][j];
        }
        result[i] = X[i][0] * X[i][0] + 1e6 * sum_squares;
    }
    return result;
}

// Function to compute the Discus function
Vector discus(const Matrix& X) {
    Vector result(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        double sum_squares = 0.0;
        for (size_t j = 1; j < X[i].size(); ++j) {
            sum_squares += X[i][j] * X[i][j];
        }
        result[i] = 1e6 * X[i][0] * X[i][0] + sum_squares;
    }
    return result;
}

// Function to compute the Weierstrass function
Vector weierstrass(const Matrix& X) {
    const double a = 0.5;
    const int b = 3;
    const size_t k_max = 20;
    const size_t n = X[0].size();

    Vector result(X.size(), 0.0);
    Vector inner_sum(X.size(), 0.0);

    for (size_t i = 0; i < X.size(); ++i) {
        for (int k = 0; k < k_max; ++k) {
            for (int j = 0; j < n; ++j) {
                inner_sum[i] += pow(a, k) * cos(2 * M_PI * pow(b, k) * (X[i][j] + 0.5));
            }
        }
        double sum_cos_term = 0.0;
        for (int k = 0; k < k_max; ++k) {
            sum_cos_term += pow(a, k) * cos(M_PI * pow(b, k));
        }
        result[i] = std::accumulate(inner_sum.begin(), inner_sum.end(), 0.0) - n * sum_cos_term;
    }
    return result;
}

// Function to compute the HappyCat function
Vector happy_cat(const Matrix& X) {
    Vector result(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        double sum_squares = 0.0;
        double sum_x = 0.0;
        for (size_t j = 0; j < X[i].size(); ++j) {
            sum_squares += X[i][j] * X[i][j];
            sum_x += X[i][j];
        }
        result[i] = ((pow(sum_squares - X[i].size(), 2)) + (0.5 * sum_squares + sum_x) / X[i].size() + 0.5);
    }
    return result;
}

// Function to compute the HGBat function
Vector hgbat(const Matrix& X) {
    Vector result(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        double sum_squares = 0.0;
        double sum_x = 0.0;
        for (size_t j = 0; j < X[i].size(); ++j) {
            sum_squares += X[i][j] * X[i][j];
            sum_x += X[i][j];
        }
        result[i] = (pow(pow(sum_squares, 2), 0.25) + (0.5 * sum_squares + sum_x) / X[i].size() + 0.5);
    }
    return result;
}

// Function to compute the HCF function
Vector hcf(const Matrix& X) {
    Vector result(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        double sum_abs = 0.0;
        double exp_term = 1.0;
        for (size_t j = 0; j < X[i].size(); ++j) {
            sum_abs += fabs(X[i][j]);
            exp_term *= exp(fabs(X[i][j]) / X[i].size());
        }
        result[i] = sum_abs * exp_term;
    }
    return result;
}

// Function to compute the Griewank-Rosenbrock function
Vector grie_rosen(const Matrix& X) {
    Vector result(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        double sum = 1.0;
        for (size_t j = 0; j < X[i].size() - 1; ++j) {
            double temp1 = 100 * pow(X[i][j + 1] - pow(X[i][j], 2), 2);
            double temp2 = pow(1 - X[i][j], 2);
            sum += temp1 + temp2;
        }
        result[i] = sum;
    }
    return result;
}

// Function to compute the Easom function
Vector easom(const Matrix& X) {
    Vector result(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        double x = X[i][0];
        double y = X[i][1];
        double term1 = -cos(x);
        double term2 = -cos(y);
        double term3 = exp(-(x - M_PI)*(x - M_PI) - (y - M_PI)*(y - M_PI));
        result[i] = term1 * term2 * term3;
    }
    return result;
}

// Function to compute the Drop-Wave function
Vector drop_wave(const Matrix& X) {
    Vector result(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        double x = X[i][0];
        double y = X[i][1];
        double numerator = 1 + cos(12 * sqrt(x * x + y * y));
        double denominator = 0.5 * (x * x + y * y) + 2;
        result[i] = -numerator / denominator;
    }
    return result;
}
}