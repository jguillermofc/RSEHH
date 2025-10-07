import numpy as np
import os
import glob

from Public.SurvivalSelection import SDDFitness, meanFitness, rankFitness, medianFitness
OUTPUT_DIR = "Results/Sequences/"


fitness_dict = {"Mean": meanFitness, "Median": medianFitness, "Rank": rankFitness, "SDD": SDDFitness}
      


def join(params):   
    prefix = f"{params.ppf}_N{params.N}_n{params.n}_G{params.Gmax}_M{params.M}_m{params.m}_ss{params.subset_size}_it{params.iterations}_runsSS{params.runs_ss}_QI{params.QI}_fit{params.fitness}"
    eval_prefix = f"Evaluation_{prefix}"
    pop_prefix = f"Population_{prefix}" 
    # Gather all evaluation and population files      
    eval_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, f"{eval_prefix}_r*.dat")))
    pop_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, f"{pop_prefix}_r*.dat")))  
    # Stack the data from all runs     
    evaluation = np.vstack([np.loadtxt(f, skiprows=1, delimiter=' ') for f in eval_files])
    sequences = np.vstack([np.loadtxt(f, skiprows=1, delimiter=' ') for f in pop_files])
    # Compute the fitness values
    if params.fitness == "SDD":        
        fitness = fitness_dict[params.fitness](evaluation, best_value = params.subset_size)
    else:
        fitness = fitness_dict[params.fitness](evaluation)
    # Sort the sequences and evaluation  based on fitness
    sorted_indices = np.argsort(fitness)
    fitness = fitness[sorted_indices]
    sequences = sequences[sorted_indices]
    evaluation = evaluation[sorted_indices]         
    # Save the combined results        
    eval_fname = os.path.join(OUTPUT_DIR, f"{eval_prefix}.dat")
    pop_fname = os.path.join(OUTPUT_DIR, f"{pop_prefix}.dat")
    fit_fname = os.path.join(OUTPUT_DIR, f"Fitness_{prefix}.dat")        
    N, NInstances = evaluation.shape
    _, NTWindows = sequences.shape
    np.savetxt(pop_fname, sequences, fmt='%d', header=str(N)+' '+str(NTWindows))
    np.savetxt(eval_fname, evaluation, fmt='%.6e', header=str(N)+' '+str(NInstances))
    np.savetxt(fit_fname, fitness, fmt='%.6e', header=str(N))
    return pop_fname, eval_fname, fit_fname
    

        
