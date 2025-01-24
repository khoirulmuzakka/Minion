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
    package_data={
        "pyminion": ["lib/*"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Minimum Python version
    install_requires=[
        # List your dependencies here, e.g.:
        "numpy>=1.21.0",
        "scipy>=1.13.1"
    ],
)
