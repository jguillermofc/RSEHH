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
        fitness = rankFitness(R)

    if fitness_type in ['Mean','Median']:
        selected = np.argsort(-fitness)[:N]
    elif fitness_type in ['Rank','SDD']:
        selected = np.argsort(fitness)[:N]
        
    return population(R.decision[selected], R.evaluation[selected])

def meanFitness(R):
    """Calculates fitness of individuals based on mean"""
    return np.mean(R.evaluation, axis=1)

def medianFitness(R):
    """Calculates fitness of individuals based on median"""
    return np.median(R.evaluation, axis=1)

def rankFitness(R):
    """Calculates fitness of individuals based on ranking"""
    rankings = stats.rankdata(-R.evaluation, method='average', axis=0)
    return np.mean(rankings, axis=1)
    
def SDDFitness(R): 
    """Calculates fitness of individuals based on standard deviation of differences (Pillay & Qu (2020))"""
    N, NA = np.shape(R.evaluation)
    best = np.max(R.evaluation,axis=0)    
    stdDevDiff = np.zeros(N)
    for i in range(N):
    	x = np.zeros(NA)
    	for j in range(NA):
    		if best[j] != 0 and R.evaluation[i,j] != 0:
    			x[j] = ( (np.abs(R.evaluation[i]-best)) / (np.mean(np.vstack((R.evaluation[i],best)),axis=0)) ) * 100
    	x_mean = np.mean(x)
    	stdDevDiff[i] = np.sqrt(((x - x_mean)**2)/(N-1))
    return stdDevDiff
