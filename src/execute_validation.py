import sys
import os
import numpy as np
import argparse
import hashlib
from itertools import product

from Public.FastIterativeGreedyRemovalAlgorithm import fastIterativeGreedyRemovalAlgorithm
from Public.ObtainMinAndMax import obtainMinAndMax
from Public.SaveApproximationSet import saveApproximationSet
from Indicators.SPD import SPD
from Indicators.MMD import MMD



PPF = "RSE"
ITERS = 10000
QI = "SPD"
RUNS = 21
CARD_GROUND_SET = [10000,]
SEQ = ["best", "median", "worst"]
PROBLEMS = {2: ["DTLZ1", "DTLZ2", "DTLZ5", "DTLZ7", "IMOP1", "IMOP2", "IMOP3", "WFG1", "WFG2", "WFG3", "WFG4", "ZDT1", "ZDT2", "ZDT3", "ZDT6"], 
            3: ["DTLZ1", "DTLZ2", "DTLZ5", "DTLZ7", "IMOP4", "IMOP5", "IMOP6", "IMOP7", "IMOP8", "VNT1", "VNT2", "VNT3", "WFG1", "WFG2", "WFG3", "WFG4"],
            4: ["DTLZ1", "DTLZ2", "DTLZ5", "DTLZ7", "WFG1", "WFG2", "WFG3", "WFG4"],
            5: ["DTLZ1", "DTLZ2", "DTLZ5", "DTLZ7", "WFG1", "WFG2", "WFG3", "WFG4"],
            6: ["DTLZ1", "DTLZ2", "DTLZ5", "DTLZ7", "WFG1", "WFG2", "WFG3", "WFG4"],
            7: ["DTLZ1", "DTLZ2", "DTLZ5", "DTLZ7", "WFG1", "WFG2", "WFG3", "WFG4"],
            8: ["DTLZ1", "DTLZ2", "DTLZ5", "DTLZ7", "WFG1", "WFG2", "WFG3", "WFG4"],
            9: ["DTLZ1", "DTLZ2", "DTLZ5", "DTLZ7", "WFG1", "WFG2", "WFG3", "WFG4"],
            10: ["DTLZ1", "DTLZ2", "DTLZ5", "DTLZ7", "WFG1", "WFG2", "WFG3", "WFG4"]
            }

SS_SIZE = {
    2: [10, 25, 50, 100, 150, 200],
    3: [28,  55, 105, 153, 210],
    5: [15, 70, 126, 210],
    8: [36, 120],
    10: [10, 55, 220]
}

def _run_validation(type_seq, problem, m, ppf, subset_size, iterations, QI, runs, seq_file):
    # Load file to be processed.
    A = np.genfromtxt('ParetoFronts/'+'Test/'+'{0:0=2d}D/'.format(m)+problem+'_{0:0=2d}D'.format(m)+'.pof')
    # Define set of distances.
    distances_list = ['euclidean', 'seuclidean', 'cityblock', 'chebyshev', 'braycurtis', 'mahalanobis', 'correlation', 'canberra', 'cosine']
    # Load file with sequences of distances.
    P = np.genfromtxt(seq_file, dtype='int')
    
    if type_seq == "best":
        sequence = P[0]
    elif type_seq == "worst":
        sequence = P[len(P) - 1]
    elif type_seq == "median":
        sequence = P[int(np.floor(len(P) / 2)) + 1]
    # Load reference points
    zmin, zmax = obtainMinAndMax(problem, m)
    zmin = np.tile(zmin, (subset_size, 1))
    zmax = np.tile(zmax, (subset_size, 1))
    evaluation = []
    for run in range(1, runs+1):
        print(f'{type_seq} solution | Problem:', problem, '| Objectives:', m, '| Indicator:', QI, '| Run:', run)
        # Execute RSEIterative
        S = fastIterativeGreedyRemovalAlgorithm(A, distances_list, ppf, subset_size, iterations, sequence)
        Sprime = (S-zmin)/(zmax-zmin)
        if QI == 'SPD':
            evaluation.append(SPD(Sprime))
        elif QI == 'MMD':
            evaluation.append(MMD(Sprime))
        saveApproximationSet(S, ppf, type_seq, m, problem, subset_size, run, seq_file, 'save_all')
        
    name, _ = os.path.splitext(seq_file)
    md5_hash = hashlib.md5(name.encode()).hexdigest()
    np.savetxt(f'Results/Performance/{type_seq}_'+problem+'_ss{0:0=d}_{1:0=2d}D'.format(subset_size, m)+'_'+md5_hash+'.'+QI.lower(), evaluation, fmt='%.18e', header=str(len(evaluation))+' 1')



def validate(sequence_file, nobj):
    problems = PROBLEMS[nobj]    
    for problem, card, type_seq, ss_size in product(
        problems,
        CARD_GROUND_SET,
        SEQ,
        SS_SIZE[nobj]
    ):
        instance = f"{problem}_{card}"
        _run_validation(type_seq, instance, nobj, PPF, ss_size, ITERS, QI, RUNS, sequence_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a sequence obtained by the hyper-heuristic.")
    parser.add_argument('--sequence_file', required=True, type=str, help="File containing the sequence of distances.")
    parser.add_argument('--nobj', required=True, type=int, help="Number of objectives of the problems to be solved.")
    args = parser.parse_args()
    if args.nobj < 2:
        args.error("nobj should be larger or equal than 2!")
        sys.exit(-1)
    validate(args.sequence_file, args.nobj)