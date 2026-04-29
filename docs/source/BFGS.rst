L-BFGS-B and L-BFGS (Unconstrained) Algorithms
===============================================

The implementation of the limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm (L-BFGS) and its bound-constrained counterpart (L-BFGS-B) in Minion is designed to comply with the library's vectorization requirements. Consequently, function evaluations and their derivatives are computed in batches, enabling the algorithm to achieve parallelization speeds comparable to other population-based algorithms. The L-BFGS-B and L-BFGS algorithms use the LBFGSpp library as a backend, with enhancements for improved vectorization and noise handling.

Robustness in Noisy Optimization Problems
-----------------------------------------

In real-world problems, objective functions are rarely smooth. Generally, quasi-Newton methods such as L-BFGS-B and L-BFGS are not recommended for noisy functions, as gradient calculations can become inaccurate. However, the Minuit optimization library, which implements the quasi-Newton algorithm Migrad, has been reliably used in high-energy physics for decades despite handling noisy objective functions. Inspired by this, Minion aims to achieve a similar level of robustness in its L-BFGS-B and L-BFGS implementations.

To enhance robustness under noisy conditions, the derivative must be computed as accurately as possible. For a noisy function using forward differences, the optimal step size for derivative estimation is given by:

.. math::
   h = 2\sqrt{\frac{\epsilon_f}{|f^{(2)}|}}

where :math:`f^{(2)}` is the second derivative. Here, the noise is assumed to be multiplicative: :math:`\epsilon_f(x) = \epsilon_r f(x)`, where :math:`\epsilon_r \ll 1`. The second derivative is estimated from the L-BFGS updates and computed recursively using results from previous iterations, eliminating the need for additional function evaluations. Similarly, the function values used to calculate :math:`\epsilon_f` are obtained from previous iterations.

Noise-Robust Derivative Estimation
-----------------------------------

To further improve robustness, Minion employs the Lanczos noise-robust derivative estimation:

.. math::
   f'(x) \approx \frac{3}{h} \sum_{k=1}^m \frac{k}{m(m+1)(2m+1)} (f(x+kh)-f(x-kh)), \quad m=\frac{N-1}{2}

This formula is derived by fitting a quadratic function to :math:`N` sample points. Notably, when :math:`N=3`, it reduces to the standard central difference formula. In Minion, setting :math:`N=1` corresponds to forward differences.

Performance Comparison
----------------------

The following notebook presents a comparison of Minion's L-BFGS-B and L-BFGS algorithms for solving CEC2017 benchmark problems. The results are compared against Minuit's Migrad and SciPy's L-BFGS-B implementation.

.. toctree::
   :maxdepth: 2
   :caption: Comparison Notebook:

   l_bfgs_b
