import sys
import numpy as np
import argparse


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyper-heuristic to select distance metric for Riesz s-kernel during subset selection (RSEIterative).")
    parser.add_argument('--N', required=True, type=int, default=10, help="Population size.")
    parser.add_argument('--n', required=True, type=int, default=2, help="Number of decision variables (i.e., number of time windows).")
    parser.add_argument('--Gmax', required=True, type=int, default=100, help="Maximum number of generations.")
    parser.add_argument("--M", required=True, type=int, help="Cardinality of the training sets.")
    parser.add_argument("--m", required=True, type=int, default=2, help="number of objectives of the training sets.")
    parser.add_argument("--ppf", required=True, type=str, default="RSE", help="Pair-potential kernel name", choices=['RSE', 'COU', 'MPT', 'PTP', 'KRA', 'GAE'])
    parser.add_argument("--subset_size", required=True, type=int, default=100, help="Desired subset size.")
    parser.add_argument("--iterations", required=True, type=int, default=100, help="Number of iterations for RSEIterative.")
    parser.add_argument("--QI", required=True, type=str, default="SPD", choices=["SPD", "MMD"], help="Quality indicator for fitness evaluation of the resulting subset.")
    parser.add_argument("--runs", required=True, type=int, default=1, help="Number of independent runs required to evaluate the best solution on the problem.")
    parser.add_argument("--fitness", required=True, type=str, default="SDD", choices=["Mean", "Median", "Rank", "SDD"], help="Fitness type used to assess the performance of the sequences of metrics for RSEIterative.")
    args = parser.parse_args()
    
    if args.N < 0:
        args.error("N must be a positive integer!")
        sys.exit(-1)
    if args.n < 0:
        args.error("n must be a positive integer!")
        sys.exit(-1)
    if args.Gmax < 0:
        args.error("Gmax must be a positive integer!")
        sys.exit(-1)
    if args.M < 0:
        args.error("M must be a positive integer!")
        sys.exit(-1)
    if args.m < 2:
        args.error("m should be larger or equal than 2!")
        sys.exit(-1)
    if args.subset_size < 0 or args.subset_size > args.M:
        args.error(f"The subset size should be in [1, {args.M}]!")
        sys.exit(-1)
    if args.iterations < 0:
        args.error("The number of iterations for RSEIterative should be a positive integer!")
        sys.exit(-1)
    if args.runs < 0:
        args.error("runs should be a positive integer!")
        sys.exit(-1)

        
    training_problems = ['ZCAT1', 'ZCAT2', 'ZCAT3', 'ZCAT4', 'ZCAT5',
                            'ZCAT6', 'ZCAT7', 'ZCAT8', 'ZCAT9', 'ZCAT10',
                            'ZCAT11', 'ZCAT12', 'ZCAT13', 'ZCAT14', 'ZCAT15',
                            'ZCAT16', 'ZCAT17', 'ZCAT18', 'ZCAT19', 'ZCAT20']
    
    file_list = [problem+'_'+str(args.M)+'_{0:0=2d}D.pof'.format(args.m) for problem in training_problems]
    
    training_sets = [np.genfromtxt('ParetoFronts/'+str(args.M)+'/{0:0=2d}D/'.format(args.m)+file) for file in file_list]
    
    distances_list = ['euclidean', 'seuclidean', 'cityblock', 'chebyshev', 'braycurtis', 'mahalanobis', 'correlation', 'canberra', 'cosine']
    
    print('Population size:', args.N, '| Windows:', args.n, '| Generations:', args.Gmax, 
            '| Set size:', args.M, '| Objectives:', args.m, '| PPF:', args.ppf, '| Subset size:', args.subset_size, '| Iterations:', args.iterations, 
            '| Indicator:', args.QI, '| Runs per evaluation:', args.runs, '| Fitness:', args.fitness)
    
    P = GA(args.N, 
           args.n, 
           args.Gmax, 
           training_problems, 
           training_sets, 
           distances_list, 
           args.ppf, 
           args.subset_size, 
           args.iterations, 
           args.QI, 
           args.runs, 
           args.fitness)
    N, n = np.shape(P.decision)
    NA = len(P.evaluation[0])
    np.savetxt('Results/FinalPopulation.txt', P.decision, fmt='%d', header=str(N)+' '+str(n))
    np.savetxt('Results/FinalEvaluation.txt', P.evaluation, fmt='%.6e', header=str(N)+' '+str(NA))
