
Notes Regarding Vectorization Support
=========================================

As mentioned, Minion requires the objective function to be vectorized. This ensures that algorithms capable of batch function calls can 
fully utilize the parallelization implemented by the user in the vectorized function. However, some algorithms do not support batch function 
calls natively. Population-based algorithms are generally known for their support of batch function calls, while sequential ones, 
such as Nelder-Mead, do not.

Here is a list of algorithms implemented in Minion/MinionPy that **support** batch function calls, and therefore can fully take advantage 
of parallelization:

- DE  
- LSHADE  
- AGSK  
- JADE  
- jSO  
- ARRDE  
- NL-SHADE-RSP  
- LSRTDE  
- GWO-DE  
- ABC  
- PSO  
- SPSO2011  
- DMSPSO  
- CMA-ES  
- BIPOP-aCMAES
- LSHADE-cnEpSin  
- L-BFGS-B  

Algorithms that **do not** support batch function calls:

1) j2020  
2) Nelder-Mead  

Algorithms that **partially** support batch function calls:

1) Dual Annealing  

The L-BFGS-B algorithm in Minion supports batch function calls because, in every function evaluation, 
a derivative is also computed. Evaluations of the objective function at the evaluation point and the shifted point (for the derivative) 
can be performed in parallel. Note that this parallelization feature is not available in other optimization library that implement L-BFGS-B such as SciPy. 
Since Dual Annealing uses L-BFGS-B for local search, it can also benefit from this feature.

The j2020 algorithm, although population-based, is not written in a way that supports batch function calls. 
This is because an update is performed after every function call.
