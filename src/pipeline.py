import numpy as np
import os
from Experiments.params import Parameters

from hyperheuristic import execute_hyperheuristic
from join_sequences import join
from execute_validation import validate

param_set = [#Parameters("RSE", 10, 100, 100, 10000, 2, 105, 10000, "SPD", 11, "SDD", 5),
            Parameters("RSE", 10, 100, 100, 10000, 3, 105, 10000, "SPD", 11, "SDD", 5),
            Parameters("RSE", 10, 100, 100, 10000, 4, 105, 10000, "SPD", 11, "SDD", 5),
            Parameters("RSE", 10, 100, 100, 10000, 5, 105, 10000, "SPD", 11, "SDD", 5),
            Parameters("RSE", 10, 100, 100, 10000, 6, 105, 10000, "SPD", 11, "SDD", 5),
            Parameters("RSE", 10, 100, 100, 10000, 7, 105, 10000, "SPD", 11, "SDD", 5),
            Parameters("RSE", 10, 100, 100, 10000, 8, 105, 10000, "SPD", 11, "SDD", 5),
            Parameters("RSE", 10, 100, 100, 10000, 9, 105, 10000, "SPD", 11, "SDD", 5),
            Parameters("RSE", 10, 100, 100, 10000, 10, 105, 10000, "SPD", 11, "SDD", 5)]

if __name__ == "__main__":
    for params in param_set:
        #execute_hyperheuristic(params)
        pop_fname, eval_fname, fit_fname = join(params)
        validate(pop_fname, params.m)