Notes Regarding Vectorization Support
====================================

Minion and MinionPy expect the objective function to be **vectorized**. In other words, the objective should accept a batch of candidate points and return one objective value per point.

For **most algorithms**, this vectorized interface is used **natively**. That means the algorithm can submit batches with size greater than 1, so any multithreading or multiprocessing inside the objective can be used effectively.

Algorithms that do **not** support native batch evaluation
==========================================================

The following algorithms do not use native batch evaluation. Even if the objective function is vectorized, the effective batch size is still ``1``:

- ``j2020``
- ``Nelder-Mead``

For these algorithms, Minion still calls the objective through the vectorized interface, but one candidate point is evaluated at a time.


Algorithms with partial batch support
=====================================

- ``Dual Annealing``

``Dual Annealing`` is only partially batch-oriented. Its local-search stage can still benefit from batch evaluation because it uses derivative-based evaluations internally.


L-BFGS-B and L-BFGS
===================

``L-BFGS-B`` and ``L-BFGS`` benefit from vectorization because function and finite-difference derivative evaluations can be grouped into batches. This lets Minion exploit parallel objective evaluation even though these are not population-based methods.
