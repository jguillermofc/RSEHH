import sys
import os
import numpy as np
import argparse
import hashlib

from Public.FastIterativeGreedyRemovalAlgorithm import fastIterativeGreedyRemovalAlgorithm
from Public.ObtainMinAndMax import obtainMinAndMax
from Public.SaveApproximationSet import saveApproximationSet
from Indicators.SPD import SPD
from Indicators.MMD import MMD

DIR = "Results/Sequences/"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tester of distance sequence for RSEIterative.")
    parser.add_argument("--problem", required=True, type=str, help="Problem's name.")
    parser.add_argument("--m", required=True, type=int, default=2, help="Number of objective functions of the problem.")
    parser.add_argument("--ppf", required=True, type=str, default="RSE", help="Pair-potential kernel name", choices=['RSE', 'COU', 'MPT', 'PTP', 'KRA', 'GAE'])
    parser.add_argument("--subset_size", required=True, type=int, default=100, help="Desired subset size.")
    parser.add_argument("--iterations", required=True, type=int, default=100, help="Number of iterations for RSEIterative.")
    parser.add_argument("--QI", required=True, type=str, default="SPD", choices=["SPD", "MMD"], help="Quality indicator for fitness evaluation of the resulting subset.")
    parser.add_argument("--runs", required=True, type=int, default=1, help="Number of independent runs required to evaluate the best solution on the problem.")
    parser.add_argument("--file", required=True, type=str, help="File that contains the resulting sequences.")
    parser.add_argument("--seq", type=str, default="best", choices=["best", "worst", "median"], help="Sequence of distances to analyze.")
    args = parser.parse_args()
    
    if args.m < 2:
        args.error("The number of objective functions (m) should be larger or equal than 2!")
        sys.exit(-1)
    if args.subset_size < 1:
        args.error("The subset size should be larger or equal than 1!")
        sys.exit(-1)
    if args.iterations < 1:
        args.error("The number of iterations should be larger or equal than 1!")
        sys.exit(-1)
    if args.runs < 1:
        args.error("The number of runs should be at least one!")
        exit(-1)

    # Load file to be processed.
    A = np.genfromtxt('ParetoFronts/'+'Test/'+'{0:0=2d}D/'.format(args.m)+args.problem+'_{0:0=2d}D'.format(args.m)+'.pof')
    # Define set of distances.
    distances_list = ['euclidean', 'seuclidean', 'cityblock', 'chebyshev', 'braycurtis', 'mahalanobis', 'correlation', 'canberra', 'cosine']
    # Load file with sequences of distances.
    seq_file = os.path.join(DIR, args.file)
    P = np.genfromtxt(seq_file, dtype='int')
    
    if args.seq == "best":
        sequence = P[0]
    elif args.seq == "worst":
        sequence = P[len(P) - 1]
    elif args.seq == "median":
        sequence = P[int(np.floor(len(P) / 2)) + 1]
    

    # Load reference points
    zmin, zmax = obtainMinAndMax(args.problem, args.m)
    zmin = np.tile(zmin, (args.subset_size, 1))
    zmax = np.tile(zmax, (args.subset_size, 1))
    evaluation = []
    for run in range(1, args.runs+1):
        print(f'{args.seq} solution | Problem:', args.problem, '| Objectives:', args.m, '| Indicator:', args.QI, '| Run:', run)
        # Execute RSEIterative
        S = fastIterativeGreedyRemovalAlgorithm(A, distances_list, args.ppf, args.subset_size, args.iterations, sequence)
        Sprime = (S-zmin)/(zmax-zmin)
        if args.QI == 'SPD':
            evaluation.append(SPD(Sprime))
        elif args.QI == 'MMD':
            evaluation.append(MMD(Sprime))
        saveApproximationSet(S, args.seq, args.problem, args.subset_size, run, args.file, 'save_all')
        
    name, _ = os.path.splitext(args.file)
    md5_hash = hashlib.md5(name.encode()).hexdigest()
    np.savetxt(f'Results/Performance/{args.seq}_'+args.problem+'_ss{0:0=d}_{1:0=2d}D'.format(args.subset_size, args.m)+'_'+md5_hash+'.'+args.QI.lower(), evaluation, fmt='%.18e', header=str(len(evaluation))+' 1')
