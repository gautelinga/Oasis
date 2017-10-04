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
	elif z[step+1] > z[step] + L/2:
	    z[step+1:] -= L

def derivative(z_l, t):
    dt = t[1]-t[0]
    u_l = np.zeros_like(t)
    u_l[1:-1] = (z_l[2:]-z_l[:-2])/(2*dt)
    u_l[0] = (z_l[1]-z_l[0])/dt
    u_l[-1] = (z_l[-1]-z_l[-2])/dt
    return u_l

def get_first_index(l):
    return next((i for i, x in enumerate(l) if x), None)


def get_first_crossing(Re0):
    temp = Re0 - Re0.mean()
    temp = temp*temp[0]
    i_first = get_first_index(temp < 0.)
    return i_first


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
    Re_stats = stats[:, 8]
    q_stats = np.sqrt(stats[:, 9])
    F_stats = stats[:, 10]

    new_vars = dict()
    for f in glob.glob(os.path.join(data_dir, "*.dat")):
        var = os.path.splitext(os.path.basename(f))[0]
        new_vars[var] = np.loadtxt(f)

    vars().update(new_vars)

    t = t*save_step*dt
    t_adv = t*np.mean(u_stats)
    
    q_z_t = np.sqrt(turb_z_t)

    dz = z[1]-z[0]
    L = dz*len(z)

    print L
	
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
    u0 = np.zeros_like(t)
    Re0 = np.zeros_like(t)
    F0 = np.zeros_like(t)
    q0 = np.zeros_like(t)
    for i in range(len(t)):
        z0[i] = z0_stats[t_stats == t[i]]
        u0[i] = u_stats[t_stats == t[i]]
        Re0[i] = Re_stats[t_stats == t[i]]
        F0[i] = F_stats[t_stats == t[i]]
        q0[i] = q_stats[t_stats == t[i]]

    u_l = derivative(z_l, t)
    u_r = derivative(z_r, t)

    i_Re0 = get_first_crossing(Re0)
    i_F0 = get_first_crossing(F0)
    i_q0 = get_first_crossing(q0)
    i_first = np.max([i_Re0, i_F0, i_q0])

    t_adv = t_adv[i_first:]
    Re0 = Re0[i_first:]
    F0 = F0[i_first:]
    q0 = q0[i_first:]

    u0 = u0[i_first:]
    u_l = u_l[i_first:]
    u_r = u_r[i_first:]
    z0 = z0[i_first:]
    z_l = z_l[i_first:]
    z_r = z_r[i_first:]
    
    plt.figure(1)
    plt.plot(t_adv, u0, 'r.-')
    plt.plot(t_adv, u_l, 'b.-')
    plt.plot(t_adv, u_r, 'y.-')
    plt.show()

    z0_fit = np.polyfit(t_adv, z0, 1)
    z_l_fit = np.polyfit(t_adv, z_l, 1)
    z_r_fit = np.polyfit(t_adv, z_r, 1)
    print z0_fit
    print z_l_fit
    print z_r_fit
    z0_f = np.poly1d(z0_fit)
    z_l_f = np.poly1d(z_l_fit)
    z_r_f = np.poly1d(z_r_fit)
    
    plt.figure(2)
    plt.plot(t_adv, z0, 'b.-')
    plt.plot(t_adv, z0_f(t_adv), 'k')
    plt.plot(t_adv, z_l, 'r.-')
    plt.plot(t_adv, z_l_f(t_adv), 'k')
    plt.plot(t_adv, z_r, 'y.-')
    plt.plot(t_adv, z_r_f(t_adv), 'k')
    plt.show()

    plt.figure(3)
    plt.plot(t_adv, z_l-z0, 'r.-')
    plt.plot(t_adv, z_r-z0,  'y.-')
    plt.plot(t_adv, z_r-z_l, 'g.-')
    plt.show()
        
    #plt.figure()
    #plt.plot(z, uz_z_t_center[step, :]/uz_mean, "b")
    #plt.plot(z, q_z_t[step, :]/uz_mean, "r")
    #plt.plot(z, flag, "y")
    #plt.show()
