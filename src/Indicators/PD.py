"""
Pure Diversity.

H. Wang, Y. Jin, and X. Yao, "Diversity Assessment in Many-Objective 
Optimization," in IEEE Transactions on Cybernetics, vol. 47, no. 6, 
pp. 1510-1522, 2017.
"""

import numpy as np
from scipy.spatial import distance



def PD(A):
    """Calculates PD indicator"""

    C = np.eye(len(A), dtype=bool)
    d = distance.pdist(A, 'minkowski', p=0.1)
    D = distance.squareform(d)
    np.fill_diagonal(D, np.inf)
    pd = 0
    for k in range(0, len(A)-1):
        while True:
            d = np.min(D, axis=1)
            neighbors = np.argmin(D, axis=1)
            i = np.argmax(d)
            j = neighbors[i]
            if D[i,j] != -np.inf:
                D[i,j] = np.inf
            if D[j,i] != -np.inf:
                D[j,i] = np.inf
            P = C[i,:]
            while not P[j]:
                newP = np.any(C[P,:], axis=0)
                if np.all(P == newP):
                    break
                else:
                    P = newP
            if not P[j]:
                break
        C[i,j] = True
        C[j,i] = True
        D[i,:] = -np.inf
        pd += d[i]
    return pd