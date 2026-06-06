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

where `X` is :
- A list of lists (`list[list[float]]`)

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

- **"DE"**, **"GWO_DE"**, **"ARRDE"**, **"AGSK"**, **"LSHADE"**, **"JADE"**, **"jSO"**, **"NLSHADE_RSP"**, **"LSRTDE"**, **"j2020"**, **"NelderMead"**.

Suppose we use the **ARRDE** algorithm. We can instantiate it using:

.. code-block:: python

    optimizer = mpy.ARRDE(
        func=objective_function,
        x0=None,
        bounds=[(-10, 10)] * dimension,
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
        maxevals=10000,
        callback=None,
        seed=None,
        options=None
    )

These two approaches are **equivalent**.  

``x0`` should be given as ``list[list[float]]`` when you want to provide initial guesses.
Each inner list is one candidate starting point, so Minion supports **multiple initial guesses**.

.. code-block:: python

    x0 = [
        [0.0] * dimension,
        [1.0] * dimension,
        [-0.5] * dimension,
    ]

    optimizer = mpy.Minimizer(
        func=objective_function,
        x0=x0,
        bounds=[(-10, 10)] * dimension,
        algo="ARRDE",
        maxevals=10000,
        callback=None,
        seed=None,
        options=None
    )

.. note::

   Multiple initial guesses are **not** the same as manually defining the algorithm's full population.
   For population-based algorithms, Minion first initializes the population according to the algorithm's own rules, then replaces some individuals with the supplied guesses.
   For non-population-based algorithms, Minion evaluates the provided guesses and starts from the best one.

Parameter Explanation:
- `x0`: Initial guesses (`list[list[float]]`). Each entry is one candidate start. In population-based algorithms, some initialized individuals are replaced by these guesses; other algorithms evaluate them and keep the best one as the actual starting point.
- `bounds`: Search space boundaries (list of tuples).
- `maxevals`: Maximum number of function evaluations.
- `callback`: A function that receives the current optimization result.
- `seed`: Random seed for reproducibility.
- `options`: Additional algorithm-specific configuration options (see **API (Python)** section). For algorithms that support tolerance-based stopping, set `options["convergence_tol"]`.

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
