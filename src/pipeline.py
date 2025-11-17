from Experiments.params import Parameters
from hyperheuristic import execute_hyperheuristic
from join_sequences import join
from execute_validation import validate

param_set = [Parameters("RSE", N=10, n=10, Gmax=100, M=10000, m=2, subset_size=100, iterations=10000, QI="SPD", runs_ss=11, fitness="SDD", runs_ga=5),
            Parameters("RSE", N=10, n=10, Gmax=100, M=10000, m=3, subset_size=100, iterations=10000, QI="SPD", runs_ss=11, fitness="SDD", runs_ga=5),
            Parameters("RSE", N=10, n=10, Gmax=100, M=10000, m=5, subset_size=100, iterations=10000, QI="SPD", runs_ss=11, fitness="SDD", runs_ga=5),
            Parameters("RSE", N=10, n=10, Gmax=100, M=10000, m=8, subset_size=100, iterations=10000, QI="SPD", runs_ss=11, fitness="SDD", runs_ga=5),
            Parameters("RSE", N=10, n=10, Gmax=100, M=10000, m=10, subset_size=100, iterations=10000, QI="SPD", runs_ss=11, fitness="SDD", runs_ga=5)
]

if __name__ == "__main__":
    for params in param_set:
        #execute_hyperheuristic(params)
        pop_fname, eval_fname, fit_fname = join(params)
        validate(pop_fname, params.m)
