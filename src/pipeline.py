from Experiments.params import Parameters
from hyperheuristic import execute_hyperheuristic
from join_sequences import join
from execute_validation import validate

param_set = [# Changing the core indicator of SDD
            #Parameters("RSE", N=10, n=100, Gmax=100, M=10000, m=2, subset_size=100, iterations=10000, QI="PD", runs_ss=11, fitness="SDD", runs_ga=5),
             #Parameters("RSE", N=10, n=100, Gmax=100, M=10000, m=2, subset_size=100, iterations=10000, QI="MMD", runs_ss=11, fitness="SDD", runs_ga=5),
             
            #Parameters("RSE", N=10, n=100, Gmax=100, M=10000, m=3, subset_size=100, iterations=10000, QI="PD", runs_ss=11, fitness="SDD", runs_ga=5),
            #Parameters("RSE", N=10, n=100, Gmax=100, M=10000, m=3, subset_size=100, iterations=10000, QI="MMD", runs_ss=11, fitness="SDD", runs_ga=5),

            #Parameters("RSE", N=10, n=100, Gmax=100, M=10000, m=5, subset_size=100, iterations=10000, QI="PD", runs_ss=11, fitness="SDD", runs_ga=5),
            #Parameters("RSE", N=10, n=100, Gmax=100, M=10000, m=5, subset_size=100, iterations=10000, QI="MMD", runs_ss=11, fitness="SDD", runs_ga=5),
             
            #Parameters("RSE", N=10, n=100, Gmax=100, M=10000, m=8, subset_size=100, iterations=10000, QI="PD", runs_ss=11, fitness="SDD", runs_ga=5),
            #Parameters("RSE", N=10, n=100, Gmax=100, M=10000, m=8, subset_size=100, iterations=10000, QI="MMD", runs_ss=11, fitness="SDD", runs_ga=5),
            
            #Parameters("RSE", N=10, n=100, Gmax=100, M=10000, m=10, subset_size=100, iterations=10000, QI="PD", runs_ss=11, fitness="SDD", runs_ga=5),
            #Parameters("RSE", N=10, n=100, Gmax=100, M=10000, m=10, subset_size=100, iterations=10000, QI="MMD", runs_ss=11, fitness="SDD", runs_ga=5)
            
            # Caso 1: Varying the maximum number of generations (Gmax).
            #Parameters("RSE", N=10, n=100, Gmax=10, M=10000, m=3, subset_size=100, iterations=10000, QI="SPD", runs_ss=11, fitness="SDD", runs_ga=5),
            #Parameters("RSE", N=10, n=100, Gmax=100, M=10000, m=3, subset_size=100, iterations=10000, QI="SPD", runs_ss=11, fitness="SDD", runs_ga=5),
            #Parameters("RSE", N=10, n=100, Gmax=1000, M=10000, m=3, subset_size=100, iterations=10000, QI="SPD", runs_ss=11, fitness="SDD", runs_ga=5),
            #NO EJECUTAR #Parameters("RSE", N=10, n=100, Gmax=10000, M=10000, m=3, subset_size=100, iterations=10000, QI="SPD", runs_ss=11, fitness="SDD", runs_ga=5),
            
            # Caso 3: Varying the length of the time window.
            #Parameters("RSE", N=10, n=10, Gmax=100, M=10000, m=3, subset_size=100, iterations=10000, QI="SPD", runs_ss=11, fitness="SDD", runs_ga=5),
            #Parameters("RSE", N=10, n=20, Gmax=100, M=10000, m=3, subset_size=100, iterations=10000, QI="SPD", runs_ss=11, fitness="SDD", runs_ga=5),
            #Parameters("RSE", N=10, n=50, Gmax=100, M=10000, m=3, subset_size=100, iterations=10000, QI="SPD", runs_ss=11, fitness="SDD", runs_ga=5),
            #Parameters("RSE", N=10, n=100, Gmax=100, M=10000, m=3, subset_size=100, iterations=10000, QI="SPD", runs_ss=11, fitness="SDD", runs_ga=5),
            #NO EJECUTAR #Parameters("RSE", N=10, n=200, Gmax=100, M=10000, m=3, subset_size=100, iterations=10000, QI="SPD", runs_ss=11, fitness="SDD", runs_ga=5),
            #NO EJECUTAR #Parameters("RSE", N=10, n=500, Gmax=100, M=10000, m=3, subset_size=100, iterations=10000, QI="SPD", runs_ss=11, fitness="SDD", runs_ga=5),
            #Parameters("RSE", N=10, n=1000, Gmax=100, M=10000, m=3, subset_size=100, iterations=10000, QI="SPD", runs_ss=11, fitness="SDD", runs_ga=5),
            #NO EJECUTAR #Parameters("RSE", N=10, n=5000, Gmax=100, M=10000, m=3, subset_size=100, iterations=10000, QI="SPD", runs_ss=11, fitness="SDD", runs_ga=5),
            Parameters("RSE", N=10, n=10000, Gmax=100, M=10000, m=3, subset_size=100, iterations=10000, QI="SPD", runs_ss=11, fitness="SDD", runs_ga=5),
            
            # Caso 2 (Faltante) -- Mucho más costoso computacionalmente, por lo que se deja para el final. Varying the population size (N).
            Parameters("RSE", N=100, n=100, Gmax=1000, M=10000, m=3, subset_size=100, iterations=10000, QI="SPD", runs_ss=11, fitness="SDD", runs_ga=5)

]

if __name__ == "__main__":
    for params in param_set:
        execute_hyperheuristic(params)
        pop_fname, eval_fname, fit_fname = join(params)
        validate(pop_fname, params.m, params.QI)
