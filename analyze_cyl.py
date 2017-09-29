import argparse
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import cPickle


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze")
    parser.add_argument("folder", type=str, help="Folder")
    args = parser.parse_args()
    return args


def runs_of_ones_list(bits):
    pbits = np.hstack(([bits[-1]], bits, [bits[0]]))
    pbits = np.array(pbits, dtype=int)
    diffs = np.diff(pbits)
    run_starts = np.where(diffs > 0)[0]
    run_ends = np.where(diffs < 0)[0]
    if run_starts[0] > run_ends[0]:
        run_ends = np.hstack((run_ends[1:], [len(bits)+run_ends[0]]))
    return zip(run_starts, run_ends)


def biggest_cluster(flag):
    clusters = np.array(runs_of_ones_list(flag))
    clusters_size = clusters[:, 1]-clusters[:, 0]
    i_max = np.argmax(clusters_size)
    cluster = clusters[i_max, :]
    return cluster


def correct(z, L):
    for step in range(len(z)-1):
        if z[step+1] < z[step] - L/2.:
            z[step+1:] += L


if __name__ == "__main__":
    args = parse_args()

    data_dir = os.path.join(args.folder,
                            "Interpolated",
                            "Data")

    stats_file = os.path.join(args.folder,
                              "Stats", "dump_flux.dat")

    params_file = os.path.join(args.folder,
                               "Timeseries", "params.dat")

    with open(params_file) as p:
        params = cPickle.load(p)

    save_step = params["save_step"]
    dt = params["dt"]
    check_flux = params["check_flux"]
    stats = np.loadtxt(stats_file)

    step_stats = stats[:, 0]
    t_stats = step_stats*dt
    u_stats = stats[:, 5]/stats[:, 7]

    new_vars = dict()
    for f in glob.glob(os.path.join(data_dir, "*.dat")):
        var = os.path.splitext(os.path.basename(f))[0]
        new_vars[var] = np.loadtxt(f)

    vars().update(new_vars)

    t = t*save_step*dt
    
    q_z_t = np.sqrt(turb_z_t)

    dz = z[1]-z[0]
    L = dz*len(z)
    print dz, dt

    z0_stats = np.cumsum(u_stats*check_flux*dt)

    q_thresh = 0.02
    step = 20

    z_l = np.zeros_like(t)
    z_r = np.zeros_like(t)
    for step in range(len(t)):
        time = step*save_step*dt
        uz_mean = u_stats[t_stats == time]

        flag = q_z_t[step, :]/uz_mean > q_thresh
        # flag2 = np.hstack((flag, flag, [True], flag))
        cluster = biggest_cluster(flag)
        z_l[step] = z[cluster[0] % len(z)]
        z_r[step] = z[cluster[1] % len(z)]

    correct(z_l, L)
    correct(z_r, L)

    z0 = np.zeros_like(t)
    for i in range(len(z0)):
        z0[i] = z0_stats[t_stats == t[i]]

    plt.figure()
    plt.plot(t, z0, 'b.-')
    plt.plot(t, z_l, 'r.-')
    plt.plot(t, z_r, 'y.-')
    plt.show()

    plt.figure()
    plt.plot(t, z_l-z0, 'r.-')
    plt.plot(t, z_r-z0,  'y.-')
    plt.plot(t, z_r-z_l, 'g.-')
    plt.show()
        
    #plt.figure()
    #plt.plot(z, uz_z_t_center[step, :]/uz_mean, "b")
    #plt.plot(z, q_z_t[step, :]/uz_mean, "r")
    #plt.plot(z, flag, "y")
    #plt.show()
