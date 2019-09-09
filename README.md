# Minion
Minion is an optimization program written in c++ focusing on derivative-free methods.
We plan to include the following alghorithm :
  - Particle Swarm Optimization (PSO)
  - Genetic Algorithm (GA)
  - Baysesian Optimization (BO)
  - Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm
  - Levenberg–Marquardt (LM) algorithm
  - Nelder-Mead Simplex 
  
and provide a universal application programming interface (API) that can be use by anyone easily. 
 
 How to compile  : 
 
    - cd to the Minion folder
    
    - cmake .
    
    - make -j8
    
    
Minion support multithreading computation. But multithreading does not always translate to a faster operation. If you want to minimze a function which is cheap to evaluate, do not use Multithreading. By default, multithreading is set to off. To enable it, when creating an instance of a minimizer, say, PSO, then specify the second argument of the constructor :
 PSO pso(20, true) ---> means pso object with 20 number of swarms with multithreading-enabled is created.
