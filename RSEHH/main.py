import sys
import time
import os
import numpy as np
from Public.subset_selection_algs import Iterative
from Public.save_files import saveApproximationSet
import argparse

DIR_distances = "input/distances/"
DIR_sequences = "input/sequences/"
DIR_pfas = "ParetoFronts/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pair-potential energy-based subset selection algorithms.')
    parser.add_argument('--ppf', required=True, type=str, default='RSE', choices=['RSE', 'COU', 'MPT', 'PTP', 'KRA', 'GAE'], help='Pair-potential energy.')
    parser.add_argument('--t_max', required=True, type=int, help='Maximum number of iterations for Iterative.')
    parser.add_argument('--instance', required=True, type=str, help='File to be processed.')
    parser.add_argument('--subset_size', required=True, type=int, default=10, help='Desired subset size.')
    parser.add_argument('--nobj', required=True, type=int, help="Number of objectives.")
    parser.add_argument('--executions', type=int, default=1, help='Number of executions.')
    parser.add_argument('--dist_list', required=True, type=str, help='File with the baseline distance functions to use')
    parser.add_argument('--sequence', required=True, type=str, help='File with the sequence to be used in RSEIterative.')     
    args = parser.parse_args()
    
    if args.t_max < 1:
        parser.error('The --t_max argument must be a positive integer.')
        sys.exit(1)
    if args.nobj < 2:
        parser.error('The number of objectives should be two or more.')
        sys.exit(1)
    if args.executions is None:
        args.executions = 1
    else:
        # and it should be larger than 0.
        if args.executions < 1:
            parser.error('The --executions argument must be a positive integer.')
            sys.exit(1)
    # Subset size should be a positive integer.
    if args.subset_size < 1:
        parser.error('The --subset_size argument must be a positive integer.')
        sys.exit(1)

    ################################### END PARAM VALIDATION #####################################
    
    # Read data from file
    instance_noext, _ = os.path.splitext(args.instance)
    A = np.genfromtxt(os.path.join(f"{DIR_pfas:s}/{args.nobj:02d}D/", args.instance))  
    M, dim = A.shape
    if args.subset_size > M:
        parser.error('The --subset_size argument must be smaller than or equal to the number of points in the input file.')
        sys.exit(1)
    # Read distance list from file
    distances_list = []    
    with open(os.path.join(DIR_distances, args.dist_list), 'r') as f:
        for line in f:
            distances_list.append(line.strip())
    # Read sequence from file
    seq = []
    with open(os.path.join(DIR_sequences, args.sequence), 'r') as f:
        for line in f:
            if line.strip() in distances_list:
                seq.append(distances_list.index(line.strip()))
            else:
                seq.append(int(line.strip()))  
    # LAUNCH SUBSET SELECTION BASED ON THE GIVEN PARAMETERS
    elapsed = []
    for run in range(1, args.executions + 1):  
        print('PPF:', args.ppf, '| Cycles:', args.t_max, '| Problem:', 
              args.instance, '| Objectives:', dim, '| Cardinality:', args.subset_size, 
              '| Run:', run)             
       
        S = Iterative(A, distances_list, args.ppf, args.subset_size, args.t_max, seq)        
        # Save approximation set 
        saveApproximationSet(S, args.ppf, instance_noext, args.subset_size, dim, run)


 
