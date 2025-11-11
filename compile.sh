#!/bin/bash

# Stop the script on the first error
set -e

# Navigate to the project directory
cd "$(dirname "$0")"

# Cleanup previous builds
echo "Cleaning up old builds..."
#rm -rf dist build *.egg-info
rm -rf minionpy/lib/ lib/

# Create and enter the build directory
mkdir -p build && cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake -G "Unix Makefiles" ..

# Compile with optimal parallelization
echo "Compiling Minion..."
cmake --build .  --config Release -- -j$(nproc)

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


#echo "building pip package..."
#python -m build

#echo "installing minionpy ... "
#pip uninstall minionpy
#pip install dist/minionpy-0.1-py3-none-any.whl
#echo "done"