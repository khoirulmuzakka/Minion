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
6. Run the relevant build or validation steps locally.
7. Commit with a clear message and push your branch.
8. Open a pull request on GitHub describing the change and any testing you performed.

Development Setup
-----------------

For source builds, the main requirements are:

- CMake 3.18 or newer
- A C++17 compiler
- Eigen3, or allow CMake to fetch dependencies as configured by the project
- Python 3 and ``pybind11`` if you are working on the Python bindings

To build from source, use the helper scripts in the repository root:

- Windows: ``compile.bat``
- Linux/macOS: ``compile.sh``

These scripts are the recommended starting point for local development because they configure the project consistently with the expected native build layout. For more detailed source-build and installation instructions, see the :doc:`installation guide <installation>`.

Running Tests Locally
---------------------

Before opening a pull request, run the local checks relevant to the part of the project you changed.

C++ test suite
^^^^^^^^^^^^^^

The repository includes a native integration test target in ``tests/test_minion.cpp``. To build and run it manually with CMake:

.. code-block:: shell

   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMINION_BUILD_CEC=ON -DMINION_BUILD_PYTHON=ON -DMINION_BUILD_EXAMPLES=ON -DMINION_BUILD_TESTS=ON
   cmake --build build --target minion_test --config Release

Run the test binary directly if you want to see the full per-algorithm output:

.. code-block:: shell

   build/bin/minion_test.exe

On Linux and macOS, run:

.. code-block:: shell

   ./build/bin/minion_test

Alternatively, you can run the same test through ``ctest``:

.. code-block:: shell

   ctest --test-dir build --output-on-failure -C Release

Note that ``ctest`` does not show the detailed output for passing tests by default; it mainly prints a short summary unless the test fails.

This test checks that:

- the core C++ algorithms run successfully on benchmark problems
- the returned objective values are finite
- the evaluation counts stay within the configured ``maxevals`` budget (with a small slack)
- the Sphere and Rosenbrock runs satisfy simple solution-quality thresholds

The CEC2017 section is mainly an integration and stability check rather than a strict performance benchmark.

Python binding test
^^^^^^^^^^^^^^^^^^^

The repository also includes a Python rewrite of ``tests/test_minion.cpp`` in ``tests/test_minionpy.py``. Run it from the repository root after building or installing ``minionpy``. See the :doc:`installation guide <installation>` if you need the source-build or install steps.

.. code-block:: shell

   python tests/test_minionpy.py

This script exercises the public ``minionpy`` interface and mirrors the same validation logic as the C++ integration test. In addition to the optimization checks described above, it also verifies that:

- vectorized helper functions return finite outputs
- the ``CEC2017Functions`` wrapper evaluates batched inputs correctly

If your change only affects documentation, you do not need to run the full optimization tests, but you should still verify that the docs build locally if you changed reStructuredText, notebooks, or generated API content.

CI Coverage
-----------

GitHub Actions runs the main validation workflow from ``.github/workflows/ci.yml``.

At the time of writing, CI covers:

- C++ integration tests through ``minion_test`` and ``ctest``
- Python package and extension build checks
- Python runtime validation through ``tests/test_minionpy.py``
- packaging jobs for release artifacts

In practice, the main CI workflow builds both the native C++ integration test and the Python extension, runs the C++ test through ``ctest``, and then runs the Python rewrite of the same integration-test logic.

If your change affects the Python bindings, examples, packaging, or documentation, verify those areas directly instead of assuming the C++ build alone is enough.

What to Check Before Submitting
-------------------------------

- The project still builds successfully.
- New or changed behavior is covered by tests or examples when practical.
- Documentation is updated when user-facing behavior changes.
- Code follows the existing style and naming conventions in the surrounding files.

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
