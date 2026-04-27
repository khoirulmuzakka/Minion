Contributing to Minion
======================

Minion is developed on GitHub at
`https://github.com/khoirulmuzakka/Minion <https://github.com/khoirulmuzakka/Minion>`_.
Contributions are welcome for bug fixes, new features, tests, examples, and documentation improvements.

How to Contribute
-----------------

1. Fork the repository on GitHub.
2. Clone your fork locally:

   .. code-block:: shell

      git clone https://github.com/yourname/Minion.git
      cd Minion

3. Add the main repository as an upstream remote:

   .. code-block:: shell

      git remote add upstream https://github.com/khoirulmuzakka/Minion.git

4. Create a feature branch for your work:

   .. code-block:: shell

      git checkout -b feature-short-description

5. Make your changes.
6. Run the relevant build or validation steps.
7. Commit with a clear message and push your branch.
8. Open a pull request on GitHub describing the change and any testing you performed.

Development Setup
-----------------

For source builds, the main requirements are:

- CMake 3.18 or newer
- A C++17 compiler
- Eigen3, or allow CMake to fetch dependencies as configured by the project
- Python 3 and ``pybind11`` if you are working on the Python bindings

If you only need the Python package for normal use, install it from PyPI instead:

.. code-block:: shell

   pip install --upgrade minionpy

To build from source, use the helper scripts in the repository root:

- Windows: ``compile.bat``
- Linux/macOS: ``compile.sh``

These scripts are the recommended starting point for local development because they configure the native build consistently with the project.

What to Check Before Submitting
-------------------------------

- The project still builds successfully.
- New or changed behavior is covered by tests or examples when practical.
- Documentation is updated when user-facing behavior changes.
- Code follows the existing style and naming conventions in the surrounding files.

If your change affects the Python bindings, examples, packaging, or documentation, verify those areas directly instead of assuming the C++ build alone is enough.

Reporting Bugs and Requesting Features
--------------------------------------

Use the GitHub issue tracker:

`https://github.com/khoirulmuzakka/Minion/issues <https://github.com/khoirulmuzakka/Minion/issues>`_

When opening an issue, include:

- A clear description of the problem or request
- Steps to reproduce the issue, if applicable
- The platform, compiler, Python version, or package version when relevant
- Any error messages, logs, or screenshots that help narrow down the cause

Documentation Contributions
---------------------------

The documentation is written in reStructuredText under ``docs/source/`` and built with Sphinx. Doxygen output is also used by the docs build.

If your environment already has the required documentation dependencies installed, you can build the docs locally from the ``docs`` directory:

.. code-block:: shell

   make html

On environments where the helper scripts are used, documentation generation may also be handled as part of the project build when the required tools are available.

Pull Request Notes
------------------

- Keep pull requests focused on a single change when possible.
- Explain the motivation for the change, not just the code diff.
- Link related issues if there are any.
- Call out any behavior changes that may affect existing users.

Thank you for contributing to Minion.
