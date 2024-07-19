#include "cec.h" 

#if defined(_MSC_VER) // Check if compiling with MSVC
#pragma warning(push)
#pragma warning(disable: 4244) // disable warning 4244
#pragma warning(disable: 4201) // disable warning 4201
#pragma warning(disable: 4101) // disable warning 4101
#pragma warning(disable: 4267) // disable warning 4267
#endif

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#endif

double *OShift,*M,*y,*z,*x_bound;
int ini_flag=0,n_flag,func_flag,*SS;


// Function to get the directory of the current file
std::string getDirectoryPath(const std::string& filePath) {
    size_t pos = filePath.find_last_of("/\\");
    return (std::string::npos == pos) ? "" : filePath.substr(0, pos);
}

const std::string dirPath = getDirectoryPath(__FILE__);

CECBase::CECBase(int function_number, int dimension) 
    : dimension_(dimension), function_number_(function_number) {
	ini_flag = 0;
}

std::vector<double> CECBase::operator()(const std::vector<std::vector<double>>& X) {
    //return std::vector<double>(X.size(), 0.0);
    int mx = X.size(); // Number of vectors to evaluate
    int nx = dimension_; // Dimension of the problem
    std::vector<double> f(mx);
    // Allocate memory for temporary arrays
    double *x = new double[mx * nx];
    double *f_temp = new double[mx];
    // Flatten X into a contiguous array
    for (int i = 0; i < mx; i++) {
        for (int j = 0; j < nx; j++) {
            x[i * nx + j] = X[i][j];
        }
    }
    try {
    	testfunc(x, f_temp, nx, mx, function_number_);
	} catch (const std::exception& e) {
                delete[] x;
                delete[] f_temp;
                throw std::runtime_error(e.what());
	};
    // Copy results from f_temp to f
    for (int i = 0; i < mx; i++) {
        f[i] = f_temp[i];
    }
    // Free memory
    delete[] x;
    delete[] f_temp;
    return f;
};
