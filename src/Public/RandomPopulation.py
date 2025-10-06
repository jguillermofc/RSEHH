"""
Random population.
"""

import numpy as np

from Public.Population import population
from Public.Evaluate import evaluate

# def randomPopulation(N, n, lb, ub, training_problems, training_sets, distances_list, ppf, subset_size, iterations, indicator, runs):
#     """Generates a random population"""
#     decision = np.random.randint(lb, ub+np.full(n, 1.0), (N, n))
#     evaluation = evaluate(decision, training_problems, training_sets, distances_list, ppf, subset_size, iterations, indicator, runs)
#     return population(decision, evaluation)

# DESCOMENTAR SI SE QUIEREN INYECTAR secuencias [i, i, i, ..., i] para i=0..8
def randomPopulation(N, n, lb, ub, training_problems, training_sets, distances_list, ppf, subset_size, iterations, indicator, runs):
    """Generates a random population"""
    # First 9 rows, each row filled with the same number 0 to 8
    first_rows = np.array([np.full(n, i) for i in range(0, 9)])
    
    # Remaining rows with random integers between 0 and 8 inclusive
    if N > 9:
        #random_rows = np.random.randint(0, 9, size=(N - 9, n))
        decision = np.vstack((first_rows, np.random.randint(lb, ub+np.full(n, 1.0), (N - 9, n))))
    else:
        # If N <= 9, just take the first N rows from first_rows
        decision = first_rows[:N]
    evaluation = evaluate(decision, training_problems, training_sets, distances_list, ppf, subset_size, iterations, indicator, runs)
    return population(decision, evaluation)
