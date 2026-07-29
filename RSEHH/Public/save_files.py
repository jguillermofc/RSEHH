"""
Save approximation set.
"""

import numpy as np
import os

DIR_subset = "output/"

def saveApproximationSet(A, ppf, instance, ss_size, dim, run):
    """Draws and saves a given approximation set"""
    fname_prefix = f"{ppf:s}_{instance}_ss{ss_size:d}_R{run:02d}.pof"
    fname_pof = os.path.join(DIR_subset, fname_prefix)
    print(f"Saving approximation set to {fname_pof:s}")
    np.savetxt(fname_pof, A, fmt='%.6e', header=str(ss_size)+' '+str(dim))
   