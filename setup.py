from setuptools import setup, find_packages

setup(
    name="pyminion",  # Package name (what users will `pip install`)
    version="0.1",    # Version number
    author="Khoirul Faiq Muzakka",  # Replace with your name
    author_email="khoirul.muzakka@gmail.com",  # Replace with your email
    description="Minion is a library for derivative-free optimization algorithms implemented in C++ and Python.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/khoirulmuzakka/Minion",  # Replace with your GitHub repo
    packages=find_packages(),  # Automatically find and include packages
    include_package_data=True,
    package_data={
        "pyminion": ["lib/*", "cec_input_data/*"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Environment :: Console",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
    python_requires=">=3.6",  # Minimum Python version
    install_requires=[
        # List your dependencies here, e.g.:
        "numpy>=1.21.0",
        "scipy>=1.13.1"
    ],
)
