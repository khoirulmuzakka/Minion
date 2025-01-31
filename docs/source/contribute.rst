Contributing to Minion
======================

We welcome contributions to the **Minion** library! Whether you want to fix bugs, add new features, improve documentation, or contribute in other ways, this guide will help you get started.

How to Contribute
-----------------

1. **Fork the Repository**  
   - Go to the Minion GitLab repository and click **Fork**.
   - Clone your fork to your local machine:  
     
     .. code-block:: shell

        git clone https://gitlab.com/yourname/Minion.git
        cd Minion
     

2. **Set Up Your Environment**  
   - Install dependencies:  
     
     .. code-block:: shell

        pip install pybind11 
     
   - Ensure you have a C++ compiler installed.

3. **Create a New Branch**  
   - Use a descriptive name for your branch:  
     
     .. code-block:: shell

        git checkout -b feature-new-algorithm
     

4. **Make Your Changes**  
   - Modify the code or documentation as needed.
   - Follow the existing code style and structure.
   - Add docstrings to new functions and classes.

5. **Run Tests**  
   - Ensure your changes do not break existing functionality. Most importantly, it must be compileable.  

6. **Commit Your Changes**  
   - Use meaningful commit messages:  
     
     .. code-block:: shell

      git add .
      git commit -m "Add new optimization algorithm"
     

7. **Push Your Changes and Open a Merge Request**  
   - Push your changes to your fork:  

     .. code-block:: shell

        git push origin feature-new-algorithm
     
   - Open a **Merge Request (MR)** on GitLab.
   - Describe your changes and request a review.

Reporting Issues
----------------

If you find a bug or have a feature request, please create an issue on GitLab.  
Provide a clear description, steps to reproduce (if applicable), and any relevant logs or screenshots.

Code Style Guide
----------------

- Follow **PEP 8** for Python code.
- Use **Doxygen-style comments** for C++ code.
- Write meaningful variable and function names.

Documentation Contributions
---------------------------

- Minion documentation is built with **Sphinx** and **Doxygen**.
- To build the documentation locally:
  ```
  cd docs
  make html
  ```
- Edit `.rst` files in `docs/source/` for documentation changes.

Thank you for contributing to Minion!