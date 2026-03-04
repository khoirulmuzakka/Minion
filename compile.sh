#!/bin/bash

# Stop the script on the first error
set -e

# Navigate to the project directory
cd "$(dirname "$0")"

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
  cmake -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=${build_type} \
    -DMINION_BUILD_CEC=ON \
    -DMINION_BUILD_EXAMPLES=ON \
    -DMINION_BUILD_PYTHON=ON \
    -DCMAKE_CXX_FLAGS_DEBUG="-O0 -g3 -fno-omit-frame-pointer" \
    -DCMAKE_C_FLAGS_DEBUG="-O0 -g3 -fno-omit-frame-pointer" \
    ..
else
  cmake -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=${build_type} \
    -DMINION_BUILD_CEC=ON \
    -DMINION_BUILD_EXAMPLES=ON \
    -DMINION_BUILD_PYTHON=ON \
    ..
fi

# Compile with optimal parallelization
echo "Compiling Minion..."
cmake --build .  --config ${build_type} -- -j$(nproc)
# Move back to root
cd ..

# Run Doxygen if installed
if command -v doxygen &> /dev/null; then
    echo "Generating documentation with Doxygen..."
    doxygen Doxyfile
else
    echo "Warning: Doxygen not found. Skipping documentation generation."
fi

# Build HTML documentation
if [ -d "docs" ]; then
    echo "Building HTML documentation..."
    cd docs
    make clean
    make html
    cd ..
else
    echo "Warning: 'docs' folder not found. Skipping HTML documentation."
fi

echo "Build complete!"
