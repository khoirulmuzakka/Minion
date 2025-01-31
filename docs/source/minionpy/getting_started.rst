Getting Started
===============

This guide will help you set up and run optimization algorithms using **minionpy**.

1. **Importing the Library**
----------------------------

After installing **minionpy**, import it into your Python code:

.. code-block:: python

    import minionpy as mpy

2. **Defining the Objective Function and Bounds**
-------------------------------------------------

You need to define your optimization problem by specifying:

- The **objective function** (which should be vectorized).
- The **bounds** for the decision variables.

In **minionpy**, the objective function should have the following signature:

.. code-block:: python

    func(X) -> list[float]

where `X` is either:
- A list of lists (`list[list[float]]`), or  
- A 2D NumPy array (`np.ndarray`).  

Since **minionpy** expects vectorized functions, if your function takes a single input and returns a scalar, you can vectorize it as follows:

.. code-block:: python

    def func(X):
        return [fun(x) for x in X]

For example, let's define a simple **Sphere function** (sum of squares) to minimize:

.. code-block:: python

    def objective_function(X):
        return [sum(x**2) for x in X]

3. **Creating an Optimizer**
----------------------------

Choose an optimization algorithm from the available list:

- **"DE"**, **"GWO_DE"**, **"ARRDE"**, **"LSHADE"**, **"JADE"**, **"jSO"**, **"NLSHADE_RSP"**, **"LSRTDE"**, **"j2020"**, **"NelderMead"**.

Suppose we use the **ARRDE** algorithm. We can instantiate it using:

.. code-block:: python

    optimizer = mpy.ARRDE(
        func=objective_function,
        x0=None,
        bounds=[(-10, 10)] * dimension,
        relTol=0.0,
        maxevals=10000,
        callback=None,
        seed=None,
        options=None
    )

Alternatively, you can use a **generic interface**:

.. code-block:: python

    optimizer = mpy.Minimizer(
        func=objective_function,
        x0=None,
        bounds=[(-10, 10)] * dimension,
        algo="ARRDE",
        relTol=0.0,
        maxevals=10000,
        callback=None,
        seed=None,
        options=None
    )

These two approaches are **equivalent**.  

Parameter Explanation:
- `x0`: Initial guess (list or 1D NumPy array).
- `bounds`: Search space boundaries (list of tuples).
- `relTol`: Relative tolerance for convergence.
- `maxevals`: Maximum number of function evaluations.
- `callback`: A function that receives the current optimization result.
- `seed`: Random seed for reproducibility.
- `options`: Additional configuration options (see **API (Python)** section).

**Note:** All algorithms in **minionpy** share the same constructor, so the instantiation process is identical for each one.

4. **Running the Optimization**
-------------------------------

To execute the optimization process, call the `optimize` method:

.. code-block:: python

    result = optimizer.optimize()
    print(result)

5. **Interpreting the Results**
-------------------------------

The **MinionResult** object contains key information about the optimization process:

- **`x`**: The optimal solution vector.
- **`fun`**: Function value at the optimum.
- **`nit`**: Number of iterations.
- **`nfev`**: Number of function evaluations.
- **`success`**: `True` if optimization was successful, else `False`.
- **`message`**: A summary message about the optimization result.

Example:

.. code-block:: python

    print(f"Solution: {result.x}")
    print(f"Function value: {result.fun}")

For more details on available algorithms and advanced configuration, refer to the **API** section.  
For additional examples, check the **Examples** section.
