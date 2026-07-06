#!/bin/bash

# Stop the script on the first error
set -e

# Navigate to the project directory
cd "$(dirname "$0")"

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

python_cmd=""
if command_exists python3; then
  python_cmd="python3"
elif command_exists python; then
  python_cmd="python"
fi

docs_enabled=true
if ! command_exists doxygen; then
  echo "Warning: Doxygen not found. Documentation generation will be skipped."
  docs_enabled=false
fi

if ! command_exists pandoc; then
  echo "Warning: Pandoc not found. HTML documentation will be skipped."
  docs_enabled=false
fi

if [[ -z "${python_cmd}" ]]; then
  echo "Warning: Python 3 not found. HTML documentation will be skipped."
  docs_enabled=false
else
  for doc_module in sphinx sphinx_rtd_theme nbsphinx breathe; do
    if ! "${python_cmd}" -c "import ${doc_module}" >/dev/null 2>&1; then
      echo "Warning: Python module '${doc_module}' not found. HTML documentation will be skipped."
      docs_enabled=false
    fi
  done
fi

# Build type selection
build_type="Release"
if [[ "$1" == "--debug" ]]; then
  build_type="Debug"
elif [[ "$1" == "--release" ]]; then
  build_type="Release"
elif [[ -n "$1" ]]; then
  echo "Usage: $0 [--debug|--release]"
  exit 1
fi

# Cleanup previous builds
echo "Cleaning up old builds..."
#rm -rf dist build *.egg-info
rm -rf minionpy/lib/ lib/

# Create and enter the build directory
mkdir -p build && cd build

# Configure with CMake
echo "Configuring with CMake..."
if [[ "${build_type}" == "Debug" ]]; then
  cmake_args=(
    -G "Unix Makefiles"
    "-DCMAKE_BUILD_TYPE=${build_type}"
    -DMINION_BUILD_BENCHMARK=ON
    -DMINION_BUILD_EXAMPLES=ON
    -DMINION_BUILD_PYTHON=ON
    "-DCMAKE_CXX_FLAGS_DEBUG=-O0 -g3 -fno-omit-frame-pointer"
    "-DCMAKE_C_FLAGS_DEBUG=-O0 -g3 -fno-omit-frame-pointer"
    ..
  )
else
  cmake_args=(
    -G "Unix Makefiles"
    "-DCMAKE_BUILD_TYPE=${build_type}"
    -DMINION_BUILD_BENCHMARK=ON
    -DMINION_BUILD_EXAMPLES=ON
    -DMINION_BUILD_PYTHON=ON
    ..
  )
fi
cmake "${cmake_args[@]}"

# Compile with optimal parallelization
echo "Compiling Minion..."
cmake --build .  --config ${build_type} -- -j$(nproc)
# Move back to root
cd ..

# Run Doxygen if documentation dependencies are installed
if [[ "${docs_enabled}" == true ]]; then
    echo "Generating documentation with Doxygen..."
    doxygen Doxyfile
fi

# Build HTML documentation
if [[ "${docs_enabled}" == true ]] && [ -d "docs" ]; then
    echo "Building HTML documentation..."
    cd docs
    make clean
    make html
    cd ..
elif [[ "${docs_enabled}" != true ]]; then
    echo "Skipping HTML documentation because one or more documentation dependencies are missing."
else
    echo "Warning: 'docs' folder not found. Skipping HTML documentation."
fi

echo "Build complete!"
