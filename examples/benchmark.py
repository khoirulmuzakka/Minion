import sys
sys.path.append("../")
import minionpy as mpy




if __name__ == "__main__":
    # Run the benchmark driver on BBOB2009/CEC.
    bbob_benchmark = mpy.benchmark(
        mode='cec', #"cec"
        num_runs=51,
        dimension=10,
        algo='rcmaes',
        popsize=0,
        year=2017,
        max_evals=2000,
        nthreads=128,
        acc=8,
        dump_results=True,
        results_folder="./results"
    )
