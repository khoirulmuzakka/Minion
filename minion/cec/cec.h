#ifndef CEC_H
#define CEC_H
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>

namespace minion {

#define INF 1.0e99
#define EPS 1.0e-14
#define E  2.7182818284590452353602874713526625
#define PI 3.1415926535897932384626433832795029

extern double  *OShift,*M,*y,*z,*x_bound;
extern int ini_flag,n_flag,func_flag,*SS;
extern const std::string dirPath;

using CECTestFunc = void (*)(double*, double*, int, int, int);

/**
 * @class CECBase
 * @brief base class for all CEC functions.
 */
class CECBase {
protected:
    int dimension_;         ///< Dimension of the problem
    int function_number_;   ///< Function number
    CECTestFunc testfunc;
    
public : 
    size_t Ncalls=0;

public:
    /**
     * @brief Constructor
     * 
     * @param function_number Function number (1-10).
     * @param dimension Dimension of the problem.
     */
    CECBase(int function_number, int dimension);

    /**
     * @brief Destructor.
     */
    ~CECBase(){};

    /**
     * @brief Operator to evaluate CEC2020 test functions.
     * 
     * @param X Input vectors to evaluate.
     * @return Vector of function values corresponding to each input vector.
     */
    virtual std::vector<double> operator()(const std::vector<std::vector<double>>& X);
};


std::string getLibraryDirectory();
std::string getResourcePath(); 
std::string getLibraryPath();
}

#endif