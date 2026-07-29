import subprocess
from itertools import product

INSTANCES = {
    2: {"mops": ["DTLZ1", "DTLZ2", "DTLZ7", "IMOP1", "IMOP2", "IMOP3", "WFG1", "WFG2"], 
        "size": [200]},
    3: {"mops": ["DTLZ1", "DTLZ2", "DTLZ7", "IMOP4", "IMOP5", "IMOP6", "IMOP7", "IMOP8", "WFG1", "WFG2", "WFG3", "VNT1", "VNT2", "VNT3"],
        "size": [210]}    
}

M = 10000
ppf = "RSE"
TMAX = 10000
BASELINE_DISTs = "distances.txt"

def execute(params):
    cmd = ["python3", "main.py"]
    for k, v in params.items():
        cmd.append(f"--{k}")
        cmd.append(str(v))
    
    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("Command:", " ".join(cmd))
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)

if __name__ == "__main__":
    for nobj in INSTANCES.keys():
        for mop, size in product(INSTANCES[nobj]["mops"], INSTANCES[nobj]["size"]):
            params = {
                "ppf": ppf,
                "t_max": TMAX,
                "instance": f"{mop}_{M}_{nobj:02d}D.pof",
                "subset_size": size,
                "nobj": nobj,
                "executions": 1,
                "dist_list": BASELINE_DISTs,
                "sequence": f"best_{nobj:02}D.txt"
            }
            execute(params)