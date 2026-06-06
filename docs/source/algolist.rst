Implemented Algorithms
======================

This page provides a compact list of the optimization algorithms currently implemented in Minion and MinionPy, together with their canonical literature references. For implementation-specific hybrids, the cited paper is the closest parent or representative method.

For detailed descriptions, parameter documentation, and default options, see :doc:`Algorithms`.


Differential Evolution Family
=============================

``DE``
    Storn, R and Price, K, *Differential Evolution - a Simple and Efficient Heuristic for Global Optimization over Continuous Spaces*, Journal of Global Optimization, 1997, 11, 341-359.

``JADE``
    J. Zhang and A. C. Sanderson, *JADE: Adaptive Differential Evolution With Optional External Archive*, IEEE Transactions on Evolutionary Computation, 13(5), 945-958, 2009.

``LSHADE``
    R. Tanabe and A. S. Fukunaga, *Improving the search performance of SHADE using linear population size reduction*, IEEE CEC, 2014.

``LSHADE_cnEpSin``
    N. H. Awad, M. Z. Ali and P. N. Suganthan, *Ensemble sinusoidal differential covariance matrix adaptation with Euclidean neighborhood for solving CEC2017 benchmark problems*, IEEE CEC, 2017.

``jSO``
    J. Brest, M. S. Maučec and B. Bošković, *Single objective real-parameter optimization: Algorithm jSO*, IEEE CEC, 2017.

``j2020``
    J. Brest, M. S. Maučec and B. Bošković, *Differential Evolution Algorithm for Single Objective Bound-Constrained Optimization: Algorithm j2020*, IEEE CEC, 2020.

``NLSHADE_RSP``
    V. Stanovov, S. Akhmedova and E. Semenkin, *NL-SHADE-RSP Algorithm with Adaptive Archive and Selective Pressure for CEC 2021 Numerical Optimization*, IEEE CEC, 2021.

``LSRTDE``
    V. Stanovov and E. Semenkin, *Success Rate-based Adaptive Differential Evolution L-SRTDE for CEC 2024 Competition*, IEEE CEC, 2024.

``ARRDE``
    Khoirul Faiq Muzakka, Ahsani Hafizhu Shali, Haris Suhendar, Sören Möller, Martin Finsterbusch, *Robust Differential Evolution via Nonlinear Population Size Reduction and Adaptive Restart: The ARRDE Algorithm*, arXiv, 2025. https://arxiv.org/abs/2511.18429

``IMODE``
    Karam M. Sallam, Saber M. Elsayed, Ripon K. Chakrabortty, and Michael J. Ryan, *Improved Multi-operator Differential Evolution Algorithm for Solving Unconstrained Problems*, IEEE CEC, 2020.

``AGSK``
    A. W. Mohamed, A. A. Hadi, A. K. Mohamed and N. H. Awad, *Evaluating the Performance of Adaptive Gaining-Sharing Knowledge Based Algorithm on CEC 2020 Benchmark Problems*, IEEE CEC, 2020.

``GWO_DE``
    S. Mirjalili, S. M. Mirjalili, and A. Lewis, *Grey Wolf Optimizer*, Advances in Engineering Software, 69, 46-61, 2014. Minion implements a Grey Wolf / Differential Evolution hybrid based on this family of methods.


Particle Swarm and Swarm-Based Methods
======================================

``PSO``
    J. Kennedy and R. Eberhart, *Particle Swarm Optimization*, Proc. IEEE International Conference on Neural Networks, 1995.

``SPSO2011``
    M. Zambrano-Bigiarini, M. Clerc and R. Rojas, *Standard Particle Swarm Optimisation 2011 at CEC-2013: A baseline for future PSO improvements*, IEEE CEC, 2013.

``DMSPSO``
    Jing J. Liang and Ponnuthurai N. Suganthan, *Dynamic multi-swarm particle swarm optimizer with local search*, IEEE Congress on Evolutionary Computation, 2005.

``ABC``
    D. Karaboga, *An Idea Based on Honey Bee Swarm for Numerical Optimization*, Technical Report-TR06, Erciyes University, 2005.


Evolution Strategies
====================

``CMAES``
    N. Hansen and A. Ostermeier, *Adapting arbitrary normal mutation distributions in evolution strategies: the covariance matrix adaptation*, Proceedings of IEEE International Conference on Evolutionary Computation, 1996.

``BIPOP_aCMAES``
    Nikolaus Hansen, *Benchmarking a BI-population CMA-ES on the BBOB-2009 function testbed*, GECCO '09 Companion, 2009.

``RCMAES``
    Anne Auger and Nikolaus Hansen, *A Restart CMA Evolution Strategy With Increasing Population Size*, IEEE Congress on Evolutionary Computation, 2005.


Classical and Local Search Methods
==================================

``NelderMead``
    Nelder, John A.; R. Mead, *A simplex method for function minimization*, Computer Journal, 7(4), 308-313, 1965.

``DA``
    Tsallis C, Stariolo DA, *Generalized Simulated Annealing*, Physica A, 233, 395-406, 1996.

``L_BFGS_B``
    Byrd, R. H.; Lu, P.; Nocedal, J.; Zhu, C., *A Limited Memory Algorithm for Bound Constrained Optimization*, SIAM Journal on Scientific Computing, 16(5), 1190-1208, 1995.

``L_BFGS``
    Liu, D. C.; Nocedal, J., *On the Limited Memory Method for Large Scale Optimization*, Mathematical Programming B, 45(3), 503-528, 1989.
