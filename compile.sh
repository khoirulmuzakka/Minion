#!/bin/bash

# Navigate to the project directory
cd "$(dirname "$0")"


rm -rf dist build *.egg-info

#source ./python_env/bin/activate

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

echo "compiling minion ..."
# Build the project using the default build tool
cmake --build . --clean-first --config Release -- -j8

cd ..
echo "copying resources..."
mkdir -p minionpy/lib
mv lib/minionpycpp* minionpy/lib/
cp -r cec_input_data minionpy/

echo "building pip package..."
python -m build

echo "installing minionpy ... "
pip uninstall minionpy
pip install dist/minionpy-0.1-py3-none-any.whl
echo "done"