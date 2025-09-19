import sys
import numpy as np

from Public.Population import population
from Public.RandomPopulation import randomPopulation
from Public.MatingSelection import matingSelection
from Public.GenerateOffspring import generateOffspring
from Public.SurvivalSelection import survivalSelection

def GA(N, n, max_generations, training_problems, training_sets, distances_list, ppf, subset_size, iterations, indicator, runs, fitness_type):
    """Runs main framework of GA"""
    print('Initialization')
    lb, ub = np.zeros(n), np.full(n, len(distances_list)-1.0)
    pc, pm = 0.9, 1/n
    P = randomPopulation(N, n, lb, ub, training_problems, training_sets, distances_list, ppf, subset_size, iterations, indicator, runs)
    generations = 0
    
    while generations < max_generations:
        print('Generation', generations+1)
        M = matingSelection(P, N)
        Q = generateOffspring(M, N, lb, ub, pc, pm, training_problems, training_sets, distances_list, ppf, subset_size, iterations, indicator, runs)
        R = population(np.vstack((P.decision, Q.decision)), np.vstack((P.evaluation, Q.evaluation)))
        P = survivalSelection(R, N, fitness_type)
        generations += 1
    return P

if __name__ == '__main__':
    if (str(sys.argv[1]) == '--help'):
        f = open('../README.txt', 'r')
        contents = f.read()
        f.close()
        print(contents)
    else:
        if (len(sys.argv) != 12):
            sys.exit('Incorrect number of arguments. Use: hyperheuristic.py --help')
        
        N = int(sys.argv[1])
        n = int(sys.argv[2])
        max_generations = int(sys.argv[3])
        M = int(sys.argv[4])
        m = int(sys.argv[5])
        ppf = str(sys.argv[6])
        subset_size = int(sys.argv[7])
        iterations = int(sys.argv[8])
        indicator = str(sys.argv[9])
        runs = int(sys.argv[10])
        fitness_type = str(sys.argv[11])
        
        training_problems = ['ZCAT1', 'ZCAT2', 'ZCAT3', 'ZCAT4', 'ZCAT5',
                             'ZCAT6', 'ZCAT7', 'ZCAT8', 'ZCAT9', 'ZCAT10',
                             'ZCAT11', 'ZCAT12', 'ZCAT13', 'ZCAT14', 'ZCAT15',
                             'ZCAT16', 'ZCAT17', 'ZCAT18', 'ZCAT19', 'ZCAT20']
        
        file_list = [problem+'_'+str(M)+'_{0:0=2d}D.pof'.format(m) for problem in training_problems]
        
        training_sets = [np.genfromtxt('ParetoFronts/'+str(M)+'/{0:0=2d}D/'.format(m)+file) for file in file_list]
        
        distances_list = ['euclidean', 'seuclidean', 'cityblock', 'chebyshev', 'braycurtis', 'mahalanobis', 'correlation', 'canberra', 'cosine']
        
        print('Population size:', N, '| Windows:', n, '| Generations:', max_generations, 
              '| Set size:', M, '| Objectives:', m, '| PPF:', ppf, '| Subset size:', subset_size, '| Iterations:', iterations, 
              '| Indicator:', indicator, '| Runs per evaluation:', runs, '| Fitness:', fitness_type)
        
        P = GA(N, n, max_generations, training_problems, training_sets, distances_list, ppf, subset_size, iterations, indicator, runs, fitness_type)
        N, n = np.shape(P.decision)
        NA = len(P.evaluation[0])
        np.savetxt('Results/FinalPopulation.txt', P.decision, fmt='%d', header=str(N)+' '+str(n))
        np.savetxt('Results/FinalEvaluation.txt', P.evaluation, fmt='%.6e', header=str(N)+' '+str(NA))
