import subprocess
from itertools import product

def run_experiment(params):
    cmd = ["python3", "test.py"]
    for k, v in params.items():
        cmd.append(f"--{k}")
        cmd.append(str(v))
    
    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("Command:", " ".join(cmd))
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)


CARD_GROUND_SET = [10000,]
SEQ = ["best", "median", "worst"]
PROBLEMS = {#2: ["DTLZ1", "DTLZ2", "DTLZ5", "DTLZ7", "IMOP1", "IMOP2", "IMOP3", "WFG1", "WFG2", "WFG3", "WFG4", "ZDT1", "ZDT2", "ZDT3", "ZDT6"], 
            3: ["DTLZ1", "DTLZ2", "DTLZ5", "DTLZ7", "IMOP4", "IMOP5", "IMOP6", "IMOP7", "IMOP8", "VNT1", "VNT2", "VNT3", "WFG1", "WFG2", "WFG3", "WFG4"],
            #4: ["DTLZ1", "DTLZ2", "DTLZ5", "DTLZ7", "WFG1", "WFG2", "WFG3", "WFG4"],
            #5: ["DTLZ1", "DTLZ2", "DTLZ5", "DTLZ7", "WFG1", "WFG2", "WFG3", "WFG4"],
            #6: ["DTLZ1", "DTLZ2", "DTLZ5", "DTLZ7", "WFG1", "WFG2", "WFG3", "WFG4"],
            #7: ["DTLZ1", "DTLZ2", "DTLZ5", "DTLZ7", "WFG1", "WFG2", "WFG3", "WFG4"],
            #8: ["DTLZ1", "DTLZ2", "DTLZ5", "DTLZ7", "WFG1", "WFG2", "WFG3", "WFG4"],
            #9: ["DTLZ1", "DTLZ2", "DTLZ5", "DTLZ7", "WFG1", "WFG2", "WFG3", "WFG4"],
            #10: ["DTLZ1", "DTLZ2", "DTLZ5", "DTLZ7", "WFG1", "WFG2", "WFG3", "WFG4"]
            }


PPF = "RSE"
SS_SIZE = {
    2: [10, 25, 50, 100, 150, 200],
    3: [28,  55, 105, 153, 210],
    5: [15, 70, 126, 210],
    8: [36, 120],
    10: [10, 55, 220]
}
ITERS = 10000
QI = "SPD"
RUNS = 21
FILE = {#2: "Population_RSE_N10_n100_G101_M10000_m2_ss105_it10000_runsSS11_QISPD_fitSDD_r1.dat",
        3: "Population_RSE_N10_n10_G20_M10000_m3_ss105_it10000_runsSS1_QISPD_fitSDD_r1.dat"}

for nobj, problems in PROBLEMS.items():
    file = FILE[nobj]
    for problem, card, type_seq, ss_size in product(
        problems,
        CARD_GROUND_SET,
        SEQ,
        SS_SIZE[nobj]
    ):
        instance = f"{problem}_{card}"
        exp = {
            "seq": type_seq,
            "problem": instance,
            "m": nobj,
            "ppf": PPF,
            "subset_size": ss_size,
            "iterations": ITERS,
            "QI": QI,
            "runs": RUNS,
            "file": file
        }
        run_experiment(exp)
