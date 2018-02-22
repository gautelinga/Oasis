import numpy as np
import os
import argparse
import cPickle as cp


def parse_args():
    parser = argparse.ArgumentParser(description="List")
    parser.add_argument("-folder", type=str, default="./",
                        help="Folder to search in")
    parser.add_argument("string", type=str, help="String to match")
    return parser.parse_args()


def main():
    args = parse_args()
    
    folders = [x for x in os.listdir(args.folder) if args.string in x and os.path.exists(os.path.join(x, "data"))]

    for folder in folders:
        subfolders = [x for x in os.listdir(
            os.path.join(folder, "data")) if os.path.exists(
                os.path.join(folder, "data", x, "Timeseries", "params.dat"))]

        for subfolder in subfolders:
            params = cp.load(open(os.path.join(folder, "data", subfolder, "Timeseries", "params.dat"), "r"))
            print folder, subfolder, params["mesh_suffix"]
            


if __name__ == "__main__":
    main()
