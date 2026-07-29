import numpy as np
import sys

if __name__ == "__main__":
    sequence = input("Enter the sequence of distance function indices (space-separated): ")
    filename = input("Enter the output filename to save the sequence: ")
    seq = [int(x) for x in sequence.split()]
    np.savetxt(filename, seq, fmt='%d')