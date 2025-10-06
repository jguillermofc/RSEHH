"""
Generate offspring.
"""

import numpy as np

from Public.Population import population
from Public.Evaluate import evaluate

def generateOffspring(M, N, lb, ub, pc, pm, training_problems, training_sets, distances_list, ppf, subset_size, iterations, indicator, runs):
    """Generates offspring population from mating pool"""
    MA, MB = M[:len(M)//2], M[len(M)//2:]
    decision = crossover(MA, MB, N, pc)
    decision = mutation(decision, lb, ub, pm)
    evaluation = evaluate(decision, training_problems, training_sets, distances_list, ppf, subset_size, iterations, indicator, runs)    
    return population(decision, evaluation)

def crossover(MA, MB, N, pc):
    """Generates an offspring population"""
    O, n = np.shape(MA)
    mu = np.random.rand(O, n)
    mu[np.tile(np.random.rand(O, 1) > pc, (1, n))] = 1
    QA = np.copy(MA)
    QB = np.copy(MA)
    QA[mu<=0.5] = MB[mu<=0.5]
    QB[mu>0.5] = MB[mu>0.5]
    Q = np.vstack((QA, QB))
    if N % 2 == 1:
        Q = np.delete(Q, len(Q)-1, axis=0) if np.random.rand() <= 0.5 else np.delete(Q, len(Q)//2-1, axis=0)
    return Q

def mutation(Q, lb, ub, pm):
    """Mutates an offspring population"""
    N, n = np.shape(Q)
    mutate = np.random.rand(N, n) <= pm
    new_variables = np.random.randint(lb, ub+np.full(n, 1.0), (N, n))
    Q[mutate] = new_variables[mutate]
    return Q
