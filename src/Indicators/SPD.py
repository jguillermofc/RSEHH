"""
Solow Polasky Diversity.

A. R. Solow, S. Polasky, "Measuring biological diversity," in Environmental 
and Ecological Statistics, vol. 1, no. 2, pp. 95–103, 1994.
"""

import numpy as np
from scipy.spatial import distance

#from Public import efficientNonDominatedSort

def SPD(A, theta=10):
    """Calculates SPD indicator"""
    #Fronts = efficientNonDominatedSort(A)
    #A = A[Fronts[0]]
    unique = np.unique(np.around(A, 6), return_index=True, axis=0)[1]
    A = A[unique]
    d = distance.pdist(A, 'euclidean')
    C = np.exp(-theta*distance.squareform(d))
    try:
        M = np.linalg.inv(C)
        return np.sum(M)
    except np.linalg.LinAlgError:
        return 0
