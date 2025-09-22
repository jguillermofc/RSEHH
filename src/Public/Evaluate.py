"""
Evaluate.
"""

import numpy as np
import multiprocessing as mp

from Public.FastIterativeGreedyRemovalAlgorithm import fastIterativeGreedyRemovalAlgorithm
from Public.ObtainMinAndMax import obtainMinAndMax
from Indicators.SPD import SPD
from Indicators.MMD import MMD

def evaluate(decision, training_problems, training_sets, distances_list, ppf, subset_size, iterations, indicator, runs):
    """Evaluates a population"""
    N, n = np.shape(decision)
    NA = len(training_problems)
    evaluation = np.zeros((N, NA))
    distances_list = [distances_list for i in range(NA)]
    ppf = [ppf for i in range(NA)]
    subset_size = [subset_size for i in range(NA)]
    iterations = [iterations for i in range(NA)]
    indicator = [indicator for i in range(NA)]
    runs = [runs for i in range(NA)]
    cpus = mp.cpu_count() if mp.cpu_count() < NA else NA
    cpus = 1
    for i in range(N):
        print('Evaluating individual', i+1)
        individual = [decision[i] for j in range(NA)]
        print("1")
        with mp.Pool(cpus) as pool:
            results = pool.starmap(parallelFunction, zip(training_problems, training_sets, distances_list, ppf, subset_size, iterations, individual, indicator, runs))
        print("2")
        for j in range(NA):
            evaluation[i,j] = results[j]
            
        print("End of evaluation of", i +1)
    return evaluation

def parallelFunction(problem, A, distances_list, ppf, subset_size, iterations, individual, indicator, runs):
    """Helper function to parallelize the PPF-based subset selection and its evaluation"""
    m = len(A[0])
    zmin, zmax = obtainMinAndMax(problem, m)
    zmin = np.tile(zmin, (subset_size, 1))
    zmax = np.tile(zmax, (subset_size, 1))
    evaluation = []
    for run in range(runs):
        S = fastIterativeGreedyRemovalAlgorithm(A, distances_list, ppf, subset_size, iterations, individual)
        Sprime = (S-zmin)/(zmax-zmin)
        if indicator == 'SPD':
            evaluation.append(SPD(Sprime))
        elif indicator == 'MMD':
            evaluation.append(MMD(Sprime))
    return np.mean(evaluation)
