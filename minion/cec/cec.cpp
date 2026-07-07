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


#include <string>
#include <vector>
#include "path_utils.h"
#if defined(_WIN32)
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <direct.h>
    #include <windows.h>
#else
    #include <dlfcn.h>
    #include <limits.h>
    #include <sys/stat.h>
    #include <unistd.h>
#endif


namespace minion {

thread_local double *OShift,*M,*y,*z,*x_bound;
thread_local int ini_flag=0,n_flag,func_flag,*SS;
thread_local int cec_instance_count = 0;

void resetThreadLocalCECState() {
    free(M);
    free(OShift);
    free(y);
    free(z);
    free(x_bound);
    free(SS);

    M = nullptr;
    OShift = nullptr;
    y = nullptr;
    z = nullptr;
    x_bound = nullptr;
    SS = nullptr;
    ini_flag = 0;
    n_flag = 0;
    func_flag = 0;
}

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

namespace {
}  // namespace

std::string getResourcePath() {
    const std::string libraryDir = getLibraryDirectory();
    const std::string currentDir = path_utils::get_current_directory();
    const std::vector<std::string> candidates = {
        path_utils::join_path(libraryDir, "../cec_input_data"),
        path_utils::join_path(libraryDir, "../../cec_input_data"),
        path_utils::join_path(libraryDir, "cec_input_data"),
        path_utils::join_path(currentDir, "cec_input_data"),
        path_utils::join_path(currentDir, "../cec_input_data")
    };

    for (const auto& candidate : candidates) {
        if (path_utils::path_exists(candidate)) {
            return path_utils::normalize_path(candidate);
        }
    }

    // Avoid exceptions during DLL initialization. If the data directory is
    // missing, return the primary expected location and let later file-open
    // code report the problem explicitly.
    return path_utils::normalize_path(candidates.front());
}

const std::string dirPath = getResourcePath();

CECBase::CECBase(int function_number, int dimension) 
    : dimension_(dimension), function_number_(function_number) {
    ++cec_instance_count;
}

CECBase::~CECBase() {
    if (cec_instance_count > 0) {
        --cec_instance_count;
    }
    if (cec_instance_count == 0) {
        resetThreadLocalCECState();
    }
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
                //std::cerr << "Problem occurs when evaluating the test function.\n";
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
