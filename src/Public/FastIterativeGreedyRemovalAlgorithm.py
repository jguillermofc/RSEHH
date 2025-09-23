"""
Reduce cardinality.
"""

import numpy as np

from Public.PairPotentialEnergyKernel import pairPotentialEnergyKernel

def fastIterativeGreedyRemovalAlgorithm(A, distances_list, ppf, subset_size, iterations, individual):
    """Selects a set with the desired subset size from A using the fast iterative greedy removal algorithm"""
    if len(A) <= subset_size:
        S = np.copy(A)
    else:
        np.random.shuffle(A)
        zmin = np.min(A, axis=0)
        zmax = np.max(A, axis=0)
        denom = zmax-zmin
        denom[denom == 0] = 1e-12
        Aprime = (A-zmin)/denom
        selected = np.arange(0, subset_size)
        candidates = np.arange(subset_size, len(A))
        # Define the number of time windows
        window_cycles = iterations//len(individual)
        # Main cycle
        for var in individual:
            # For each distance (var represents an index in [0, 8]), and then decode it with distances_list.
            distance = distances_list[var]
            if 'minkowski' in distance:
                distance, p_minkowski = distance.split()
                p_minkowski = float(p_minkowski)
            else:
                p_minkowski = None            
            Diss = pairPotentialEnergyKernel(Aprime[selected], ppf, distance_type=distance, p_minkowski=p_minkowski)
            memo = np.sum(Diss, axis=1)
            for i in range(window_cycles):
                idx = np.random.randint(len(candidates))
                candidate = candidates[idx]
                diss = pairPotentialEnergyKernel(Aprime[selected], ppf, Aprime[candidate], distance_type=distance, p_minkowski=p_minkowski)
                memo = memo+diss
                cnew = np.sum(diss)
                worst = np.argmax(np.append(memo, cnew))
                if worst == len(memo):
                    memo = memo-diss
                else:
                    candidates[idx] = selected[worst]
                    memo = memo-Diss[:,worst]
                    selected[worst] = candidate
                    memo[worst] = cnew-diss[worst]
                    diss[worst] = 0
                    Diss[:,worst] = diss
                    Diss[worst,:] = diss
        # Define subset.
        S = A[selected]
    return S
