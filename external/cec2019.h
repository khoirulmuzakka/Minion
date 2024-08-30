#ifndef CEC2019_H
#define CEC2019_H

#include "cec.h"

/**
 * @class CEC2019Functions
 * @brief Class encapsulating CEC2019 test functions.
 */
class CEC2019Functions : public CECBase {
public:
    /**
     * @brief Constructor for CEC2019Functions.
     * 
     * @param function_number Function number (1-10).
     * @param dimension Dimension of the problem.
     */
    CEC2019Functions(int function_number, int dimension);

    /**
     * @brief Destructor.
     */
    ~CEC2019Functions(){};
};

namespace CEC2019 {       
    /**
     * All function below are unmodified from the original source :  
     * https://github.com/P-N-Suganthan/CEC2019
    */
    void Lennard_Jones(double *, int, double *); /* Lennard Jones */
    void Hilbert(double *, int, double *); /* Hilbert */
    void Chebyshev(double *, int, double *); /* Chebyshev */
    void schaffer_F7_func (double *, double *, int , double *,double *, int, int); /* Schwefel's F7 */
    void ackley_func (double *, double *, int , double *,double *, int, int); /* Ackley's */
    void rastrigin_func (double *, double *, int , double *,double *, int, int); /* Rastrigin's  */
    void weierstrass_func (double *, double *, int , double *,double *, int, int); /* Weierstrass's  */
    void schwefel_func (double *, double *, int , double *,double *, int, int); /* Schwefel's */
    void escaffer6_func (double *, double *, int , double *,double *, int, int); /* Expanded Scaffer��s F6  */
    void happycat_func (double *, double *, int , double *,double *, int, int); /* HappyCat */
    void griewank_func (double *, double *, int , double *,double *, int, int); /* Griewank's  */

    void shiftfunc (double*,double*,int,double*);
    void rotatefunc (double*,double*,int, double*);
    void sr_func (double *, double *, int, double*, double*, double, int, int); /* shift and rotate */
    void asyfunc (double *, double *x, int, double);
    void oszfunc (double *, double *, int);
    void cec19_test_func(double *, double *,int,int,int);
}
#endif