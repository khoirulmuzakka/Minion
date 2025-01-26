#include "cec.h" 
#include <filesystem>

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


#include <string>
#if defined(_WIN32)
    #include <windows.h>
#else
    #include <dlfcn.h>
#endif


namespace minion {

double *OShift,*M,*y,*z,*x_bound;
int ini_flag=0,n_flag,func_flag,*SS;

std::string getLibraryPath() {
    char path[1024];
#if defined(_WIN32)
    // Get the path of the current module (shared library or executable)
    HMODULE hModule = nullptr;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCSTR>(&getLibraryPath),
                       &hModule);
    GetModuleFileNameA(hModule, path, sizeof(path));
#else
    // Use dladdr to get the path of the current shared library
    Dl_info dl_info;
    dladdr(reinterpret_cast<void*>(&getLibraryPath), &dl_info);
    snprintf(path, sizeof(path), "%s", dl_info.dli_fname);
#endif
    //std ::cout << "Resource : " <<std::string(path)<< "\n";
    return std::string(path);
}

std::string getLibraryDirectory() {
    std::string libraryPath = getLibraryPath();
    size_t pos = libraryPath.find_last_of("/\\");
    return (std::string::npos == pos) ? "" : libraryPath.substr(0, pos);
}

std::string getResourcePath() {
    std::string libraryDir = getLibraryDirectory();
    std::filesystem::path resourcePath = std::filesystem::path(libraryDir) / "../cec_input_data/";
    resourcePath = std::filesystem::canonical(resourcePath);
    return resourcePath.string();
}

const std::string dirPath = getResourcePath();

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
                std::cerr << "Problem occurs when evaluating the test function.\n";
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
    Ncalls+=mx;
    return f;
};

}