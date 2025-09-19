import sys
import numpy as np

from Public.FastIterativeGreedyRemovalAlgorithm import fastIterativeGreedyRemovalAlgorithm
from Public.ObtainMinAndMax import obtainMinAndMax
from Public.SaveApproximationSet import saveApproximationSet
from Indicators.SPD import SPD
from Indicators.MMD import MMD

if __name__ == '__main__':
    if (str(sys.argv[1]) == '--help'):
        f = open('../README.txt', 'r')
        contents = f.read()
        f.close()
        print(contents)
    else:
        if (len(sys.argv) != 8):
            sys.exit('Incorrect number of arguments. Use: test.py --help')
        
        problem = str(sys.argv[1])
        m = int(sys.argv[2])
        ppf = str(sys.argv[3])
        subset_size = int(sys.argv[4])
        iterations = int(sys.argv[5])
        indicator = str(sys.argv[6])
        runs = int(sys.argv[7])
    
        A = np.genfromtxt('ParetoFronts/'+'Test/'+'{0:0=2d}D/'.format(m)+problem+'_{0:0=2d}D'.format(m)+'.pof')
        
        distances_list = ['euclidean', 'seuclidean', 'cityblock', 'chebyshev', 'braycurtis', 'mahalanobis', 'correlation', 'canberra', 'cosine']
        
        P = np.genfromtxt('Results/FinalPopulation.txt', dtype='int')
        best = P[0]
        
        zmin, zmax = obtainMinAndMax(problem, m)
        zmin = np.tile(zmin, (subset_size, 1))
        zmax = np.tile(zmax, (subset_size, 1))
        evaluation = []
        for run in range(1, runs+1):
            print('Best solution | Problem:', problem, '| Objectives:', m, '| Indicator:', indicator, '| Run:', run)
            S = fastIterativeGreedyRemovalAlgorithm(A, distances_list, ppf, subset_size, iterations, best)
            Sprime = (S-zmin)/(zmax-zmin)
            if indicator == 'SPD':
                evaluation.append(SPD(Sprime))
            elif indicator == 'MMD':
                evaluation.append(MMD(Sprime))
            saveApproximationSet(S, 'Best_solution', problem, run, 'save_all')
        np.savetxt('Results/Performance/Best_solution_'+problem+'_{0:0=2d}D'.format(m)+'.'+indicator.lower(), evaluation, fmt='%.18e', header=str(len(evaluation))+' 1')
