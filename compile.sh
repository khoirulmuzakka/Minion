#!/bin/bash

# Navigate to the project directory
cd "$(dirname "$0")"

# Delete existing build directory (if it exists)
if [ -d "build" ]; then
    rm -rf build
fi

# Create a new build directory
mkdir -p build

# Navigate to the build directory
cd build

# Run CMake to configure the project with Unix Makefiles
cmake -G "Unix Makefiles" ..

# Build the project using the default build tool
cmake --build . --clean-first --config Release -- -j8

# Optional: Pause to see build output (only for debugging, usually not needed)
read -p "Press any key to continue..."
