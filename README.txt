###############################################################################

Documentation of the module hyperheuristic.py.

NAME
       hyperheuristic.py - execute the hyperheuristic

SYNOPSIS
       hyperheuristic.py N n max_generations M m ppf subset_size iterations, indicator, runs, fitness_type
       hyperheuristic.py OPTION

DESCRIPTION
       This module is used to execute the hyperheuristic based on different
       distance metrics on certain iteration windows using a genetic algorithm.
       The required arguments are described as follows:

       N
              It must be an integer greater than zero. It represents the 
              population size.

       n
              It must be an integer greater than zero. It represents the 
              number of decision variables (i.e., the number of windows).

       max_generations
              It must be an integer greater than or equal to zero. It
              represents the maximum number of generations.

       M
              It must be an integer greater than zero. It represents the
              cardinality of the training sets.

       m
              It must be an integer greater than zero. It represents the
              number of objectives of the training sets.

       ppf
              It must be a valid pair-potential kernel name. The valid names
              are: RSE, GAE, COU, PTP, MPT, and KRA.

       subset_size
              It must be an integer greater than zero. It represents the
              desired subset size.

       iterations
              It must be an integer greater than or equal to zero. It
              represents the number of iterations for the fast iterative greedy
              removal algorithm.

       indicator
              It must be a valid indicator name. The valid names are SPD for
              the Solow Polasky Diversity and MMD for the Max-Min Diversity.

       runs
              It must be an integer greater than zero. It represents the number
              of runs required to evaluate an individual on each problem.

       fitness_type
              It must be a valid fitness type name. The valid names are Mean, 
              Median, Rank and SDD.

       The following option can be used:

       --help 
              Display this help and exit.

REQUIREMENTS
       A computer with the installation of Python 3.8 or superior is needed.
       The modules numpy, matplotlib, and scipy are required.

EXAMPLE
       For running the module hyperheuristic.py, go to Hyperheuristic/ and
       write:

       IPython console users:
              %run hyperheuristic.py 10 5 3 10000 3 RSE 100 10000 SPD 1 Mean

       Windows users:
              python hyperheuristic.py 10 5 3 10000 3 RSE 100 10000 SPD 1 Mean

       Linux users:
              python3 hyperheuristic.py 10 5 3 10000 3 RSE 100 10000 SPD 1 Mean

       The previous line executes the module hyperheuristic.py to execute the
       hyperheuristic using a genetic algorithm with a population size of 10
       and using individuals with 5 iteration windows. The maximum number of 
       generations is set to 3. It employs training sets with cardinality equal
       to 10000 and 3 objective functions. It uses the Riesz s-energy as
       pair-potential kernel on the fast iterative greedy removal algorithm to
       select a subset with cardinality of 100 using 10000 iterations. Finally,
       it uses the Solow Polasky Diversity to evaluate the individual using 1
       run and using the mean as fitness type.

RESULTS
       On success, the output files containing the final population are
       generated in Hyperheuristic/Results/.

###############################################################################

Documentation of the module test.py.

NAME
       test.py - test the best solution

SYNOPSIS
       test.py problem m ppf subset_size iterations indicator runs
       test.py OPTION

DESCRIPTION
       This module is used to test the best solution obtained by the
       hyperheuristic on different Pareto fronts. The required arguments are
       described as follows:

       problem
              It must be a valid problem name. The valid problem names are:
              DTLZ1, DTLZ2, DTLZ7, DTLZ1_MINUS, DTLZ2_MINUS, WFG1, and
              WFG1_MINUS.

       m
              It must be an integer greater than one. It represents the number 
              of objective functions of the problem.

       ppf
              It must be a valid pair-potential kernel name. The valid names
              are: RSE, GAE, COU, PTP, MPT, and KRA.

       subset_size
              It must be an integer greater than zero. It represents the
              desired subset size.

       iterations
              It must be an integer greater than or equal to zero. It
              represents the number of iterations for the fast iterative greedy
              removal algorithm.

       indicator
              It must be a valid indicator name. The valid names are SPD for
              the Solow Polasky Diversity and MMD for the Max-Min Diversity.

       runs
              It must be an integer greater than zero. It represents the number
              of runs required to evaluate the best solution on the problem.

       The following option can be used:

       --help 
              Display this help and exit.

REQUIREMENTS
       A computer with the installation of Python 3.8 or superior is needed.
       The modules numpy, matplotlib, and scipy are required.

EXAMPLE
       For running the module test.py, go to Hyperheuristic/ and write:

       IPython console users:
              %run test.py DTLZ7 3 RSE 100 10000 SPD 10

       Windows users:
              python test.py DTLZ7 3 RSE 100 10000 SPD 10

       Linux users:
              python3 test.py DTLZ7 3 RSE 100 10000 SPD 10

       The previous line executes the module test.py to test the best solution
       obtained by the hyperheuristic on the DTLZ7 with 3 objective functions.
       It uses the Riesz s-energy as pair-potential kernel on the fast
       iterative greedy removal algorithm to select a subset with cardinality
       of 100 using 10000 iterations. Finally, it uses the Solow Polasky
       Diversity to evaluate the best solution using 10 runs.

RESULTS
       On success, the output files containing the selected
       subsets are generated in Hyperheuristic/Results/Approximations and the
       output files containing the evaluation of each subset are generated in
       Hyperheuristic/Results/Performance.

###############################################################################
