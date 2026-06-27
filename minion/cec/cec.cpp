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

char path_separator() {
#if defined(_WIN32)
    return '\\';
#else
    return '/';
#endif
}

bool is_path_separator(char c) {
    return c == '/' || c == '\\';
}

bool has_drive_prefix(const std::string& path) {
    return path.size() > 1 &&
           ((path[0] >= 'A' && path[0] <= 'Z') || (path[0] >= 'a' && path[0] <= 'z')) &&
           path[1] == ':';
}

bool is_absolute_path(const std::string& path) {
    return !path.empty() && (is_path_separator(path.front()) || has_drive_prefix(path));
}

std::string join_path(const std::string& base, const std::string& child) {
    if (base.empty()) {
        return child;
    }
    if (child.empty()) {
        return base;
    }
    if (is_absolute_path(child)) {
        return child;
    }
    if (is_path_separator(base.back())) {
        return base + child;
    }
    return base + path_separator() + child;
}

std::vector<std::string> split_path(const std::string& path) {
    std::vector<std::string> parts;
    std::string current;
    for (char c : path) {
        if (is_path_separator(c)) {
            if (!current.empty()) {
                parts.push_back(current);
                current.clear();
            }
        } else {
            current.push_back(c);
        }
    }
    if (!current.empty()) {
        parts.push_back(current);
    }
    return parts;
}

std::string normalize_lexical_path(const std::string& path) {
    if (path.empty()) {
        return ".";
    }

    const bool absolute = is_absolute_path(path);
    const std::string root = has_drive_prefix(path)
        ? path.substr(0, 2) + path_separator()
        : (absolute ? std::string(1, path_separator()) : "");
    std::vector<std::string> normalized_parts;
    const std::string path_without_root = has_drive_prefix(path) ? path.substr(2) : path;
    for (const std::string& part : split_path(path_without_root)) {
        if (part == ".") {
            continue;
        }
        if (part == "..") {
            if (!normalized_parts.empty() && normalized_parts.back() != "..") {
                normalized_parts.pop_back();
            } else if (!absolute) {
                normalized_parts.push_back(part);
            }
            continue;
        }
        normalized_parts.push_back(part);
    }

    std::string normalized = root;
    for (size_t i = 0; i < normalized_parts.size(); ++i) {
        if (!normalized.empty() && !is_path_separator(normalized.back())) {
            normalized.push_back(path_separator());
        }
        normalized += normalized_parts[i];
    }

    if (normalized.empty()) {
        return absolute ? root : ".";
    }
    return normalized;
}

std::string get_current_directory() {
    char buffer[4096];
#if defined(_WIN32)
    if (_getcwd(buffer, static_cast<int>(sizeof(buffer))) != nullptr) {
        return std::string(buffer);
    }
#else
    if (getcwd(buffer, sizeof(buffer)) != nullptr) {
        return std::string(buffer);
    }
#endif
    return ".";
}

bool path_exists(const std::string& path) {
#if defined(_WIN32)
    DWORD attributes = GetFileAttributesA(path.c_str());
    return attributes != INVALID_FILE_ATTRIBUTES &&
           (attributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
#else
    struct stat info;
    return stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
#endif
}

std::string normalize_path(const std::string& path) {
#if defined(_WIN32)
    char resolved[_MAX_PATH];
    if (_fullpath(resolved, path.c_str(), _MAX_PATH) != nullptr) {
        return std::string(resolved);
    }
#else
    char resolved[PATH_MAX];
    if (realpath(path.c_str(), resolved) != nullptr) {
        return std::string(resolved);
    }
#endif
    return normalize_lexical_path(path);
}

}  // namespace

std::string getResourcePath() {
    const std::string libraryDir = getLibraryDirectory();
    const std::string currentDir = get_current_directory();
    const std::vector<std::string> candidates = {
        join_path(libraryDir, "../cec_input_data"),
        join_path(libraryDir, "../../cec_input_data"),
        join_path(libraryDir, "cec_input_data"),
        join_path(currentDir, "cec_input_data"),
        join_path(currentDir, "../cec_input_data")
    };

    for (const auto& candidate : candidates) {
        if (path_exists(candidate)) {
            return normalize_path(candidate);
        }
    }

    // Avoid exceptions during DLL initialization. If the data directory is
    // missing, return the primary expected location and let later file-open
    // code report the problem explicitly.
    return normalize_path(candidates.front());
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
