#ifndef MINION_PATH_UTILS_H
#define MINION_PATH_UTILS_H

#include <string>
#include <vector>

#if defined(_WIN32)
#include <direct.h>
#include <windows.h>
#else
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace minion::path_utils {

inline char path_separator() {
#if defined(_WIN32)
    return '\\';
#else
    return '/';
#endif
}

inline bool is_path_separator(char c) {
    return c == '/' || c == '\\';
}

inline bool has_drive_prefix(const std::string& path) {
    return path.size() > 1 &&
           ((path[0] >= 'A' && path[0] <= 'Z') || (path[0] >= 'a' && path[0] <= 'z')) &&
           path[1] == ':';
}

inline bool is_absolute_path(const std::string& path) {
    return !path.empty() && (is_path_separator(path.front()) || has_drive_prefix(path));
}

inline std::string join_path(const std::string& base, const std::string& child) {
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

inline std::vector<std::string> split_path(const std::string& path) {
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

inline std::string normalize_lexical_path(const std::string& path) {
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

inline std::string get_current_directory() {
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

inline bool path_exists(const std::string& path) {
#if defined(_WIN32)
    DWORD attributes = GetFileAttributesA(path.c_str());
    return attributes != INVALID_FILE_ATTRIBUTES &&
           (attributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
#else
    struct stat info;
    return stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
#endif
}

inline std::string normalize_path(const std::string& path) {
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

}  // namespace minion::path_utils

#endif
