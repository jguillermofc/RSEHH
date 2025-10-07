"""
Survival selection.
"""

import numpy as np
import scipy.stats as stats

from Public.Population import population

def survivalSelection(R, N, fitness_type):
    """Returns population with best individuals"""
    if fitness_type == 'Mean':
        fitness = meanFitness(R)
    elif fitness_type == 'Median':
        fitness = medianFitness(R)
    elif fitness_type == 'Rank':
        fitness = rankFitness(R)
    elif fitness_type == 'SDD':
        fitness = SDDFitness(R)
    # Subset selection
    if fitness_type in ['Mean','Median']:
        selected = np.argsort(-fitness)[:N]
    elif fitness_type in ['Rank','SDD']:
        selected = np.argsort(fitness)[:N]
    return population(R.decision[selected], R.evaluation[selected])

def meanFitness(evaluation):
    """Calculates fitness of individuals based on mean"""
    return np.mean(evaluation, axis=1)

def medianFitness(evaluation):
    """Calculates fitness of individuals based on median"""
    return np.median(evaluation, axis=1)

def rankFitness(evaluation):
    """Calculates fitness of individuals based on ranking"""
    rankings = stats.rankdata(-evaluation, method='average', axis=0)
    return np.mean(rankings, axis=1)

def SDDFitness(evaluation, best_value=None):
    N, NA = np.shape(evaluation)
    """Calculates fitness of individuals based on standard deviation of differences (Pillay & Qu (2020))"""
    # N: number of individuals. 
    # NA: number of point set instances (size of the training set).
    if best_value is not None:
        best = best_value * np.ones(NA)
    else:
        best = np.max(evaluation,axis=0)
    best = np.ones(NA)*105
    SDD = np.zeros(N)
    for i in range(N):
        x = np.zeros(NA)
        for j in range(NA):
            if best[j] != 0 and evaluation[i,j] != 0:
                x[j] = ( (np.abs(evaluation[i,j]-best[j])) / ((evaluation[i,j]+best[j])/2) ) * 100  
        x_mean = np.mean(x)
        SDD[i] = np.sqrt(np.sum((x - x_mean)**2)/(N-1))
    return SDD

