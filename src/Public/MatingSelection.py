"""
Mating selection.
"""

import numpy as np

def matingSelection(P, N):
    """Selects random parent population"""
    O = N+1 if N%2 == 1 else N
    if len(P.decision) > 1:
        indexes = np.array([np.random.choice(len(P.decision), 2, replace=False) for i in range(O//2)])
    else:
        indexes = np.zeros((O//2, 2), dtype=int)
    M = P.decision[np.hstack((indexes[:,0], indexes[:,1]))]
    return M
