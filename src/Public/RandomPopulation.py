"""
Random population.
"""

import numpy as np

from Public.Population import population
from Public.Evaluate import evaluate

def randomPopulation(N, n, lb, ub, training_problems, training_sets, distances_list, ppf, subset_size, iterations, indicator, runs):
    """Generates a random population"""
    decision = np.random.randint(lb, ub+np.full(n, 1.0), (N, n))
    evaluation = evaluate(decision, training_problems, training_sets, distances_list, ppf, subset_size, iterations, indicator, runs)
    return population(decision, evaluation)
