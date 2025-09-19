"""
Max-Min Diversity.

D. C. Porumbel, J. K. Hao, F. Glover, "A simple and effective algorithm for the 
MaxMin diversity problem," in Annuals of Operations Research, vol. 186, no. 1, 
pp. 275-293, 2011.
"""

import numpy as np
from scipy.spatial import distance

#from Public import efficientNonDominatedSort

def MMD(A):
    """Calculates MMD indicator"""
    #Fronts = efficientNonDominatedSort(A)
    #A = A[Fronts[0]]
    d = distance.pdist(A, 'euclidean')
    D = distance.squareform(d)
    np.fill_diagonal(D, float('inf'))
    return np.min(D)
