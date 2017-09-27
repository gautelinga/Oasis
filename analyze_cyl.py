import argparse
import os
import numpy as np
import glob
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze")
    parser.add_argument("folder", type=str, help="Folder")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    data_dir = os.path.join(args.folder,
                            "Interpolated",
                            "Data")

    new_vars = dict()
    for f in glob.glob(os.path.join(data_dir, "*.dat")):
        var = os.path.splitext(os.path.basename(f))[0]
        new_vars[var] = np.loadtxt(f)

    vars().update(new_vars)

    print uz_z_t.shape

    plt.figure()
    plt.plot(z, uz_z_t_center[0, :]/uz_z_t_center[0, :].mean(), "b")
    plt.plot(z, turb_z_t[0, :]/turb_z_t[0, :].mean(), "r")
    plt.show()
