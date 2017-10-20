import argparse
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import cPickle


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze")
    parser.add_argument("folder", type=str, help="Folder")
    parser.add_argument("-s", "--start_criterion", type=str,
                        default="crossing", help="Start criterion")
    parser.add_argument("-p", "--plot", type=str,
                        default="1,2,3,4,5,6,7", help="Which plots to plot.")
    parser.add_argument("--mode", type=str, default="trace",
                        help="Mode")
    args = parser.parse_args()
    return args


def runs_of_ones_list(bits):
    pbits = np.hstack(([bits[-1]], bits, [bits[0]]))
    pbits = np.array(pbits, dtype=int)
    diffs = np.diff(pbits)
    run_starts = np.where(diffs > 0)[0]
    run_ends = np.where(diffs < 0)[0]
    if not len(run_starts) or not len(run_ends):
        return False
    elif run_starts[0] > run_ends[0]:
        run_ends = np.hstack((run_ends[1:], [len(bits)+run_ends[0]]))
    return zip(run_starts, run_ends)


def biggest_cluster(flag):
    runs = runs_of_ones_list(flag)
    if runs == False:
        return False
    clusters = np.array(runs)
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
    ekin_stats = stats[:, 6]/stats[:, 7]
    u_stats = stats[:, 5]/stats[:, 7]
    Re_stats = stats[:, 8]
    q_stats = np.sqrt((stats[:, 9]*stats[:, 5])**2/stats[:, 7])
    F_stats = stats[:, 10]

    new_vars = dict()
    for f in glob.glob(os.path.join(data_dir, "*.dat")):
        var = os.path.splitext(os.path.basename(f))[0]
        new_vars[var] = np.loadtxt(f)
    vars().update(**new_vars)
        
    if args.mode == "balance":
        U = u_stats
        F = F_stats
        dUdt = derivative(u_stats, t_stats)
        dEdt = derivative(ekin_stats, t_stats)
        F_fric = F - dUdt
        P = F*U
        P_diss = P - dEdt

        plt.figure()
        plt.plot(t_stats, dUdt)
        plt.plot(t_stats, F_stats)
        plt.plot(t_stats, F_fric)

        plt.figure()
        plt.plot(t_stats, dEdt)
        plt.plot(t_stats, P)
        plt.plot(t_stats, P_diss)

        plt.figure()
        plt.plot(U, F_fric)

        plt.figure()
        plt.plot(U, dUdt)

        plt.figure()
        plt.plot(dUdt, F_fric)

        plt.figure()
        plt.plot(F_fric, q_stats)

        plt.show()

        exit()

    u_mean = u_stats.mean()

    t = t*save_step*dt
    t_adv = t*u_mean

    dz = z[1]-z[0]
    nz = len(z)
    L = dz*nz

    print L

    z0_stats = np.cumsum(u_stats*check_flux*dt)

    q_z_t = np.sqrt(turb_z_t)

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
        if isinstance(cluster, bool) and cluster == False:
            break
        z_l[step] = z[cluster[0] % len(z)]
        z_r[step] = z[cluster[1] % len(z)]

    t = t[:step]
    z_l = z_l[:step]
    z_r = z_r[:step]
    t_adv = t_adv[:step]

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

    if args.start_criterion == "crossing":
        i_Re0 = get_first_crossing(Re0)
        i_F0 = get_first_crossing(F0)
        i_q0 = get_first_crossing(q0)
        i_first = np.max([i_Re0, i_F0, i_q0])
    else:
        i_first = 0

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

    plots = [int(q) for q in args.plot.split(",")]
    
    if 1 in plots:
        plt.figure(1)
        plt.plot(t_adv, u0, 'r.-')
        plt.plot(t_adv, u_l, 'b.-')
        plt.plot(t_adv, u_r, 'y.-')
        plt.show()

    z0_fit = np.polyfit(t_adv, z0, 1)
    z_l_fit = np.polyfit(t_adv, z_l, 1)
    z_r_fit = np.polyfit(t_adv, z_r, 1)
    print "Mean flow: ", z0_fit
    print "Left edge: ", z_l_fit
    print "Right edge:", z_r_fit
    z0_f = np.poly1d(z0_fit)
    z_l_f = np.poly1d(z_l_fit)
    z_r_f = np.poly1d(z_r_fit)

    if 2 in plots:
        plt.figure(2)
        plt.plot(t_adv, z0, 'b.-')
        plt.plot(t_adv, z0_f(t_adv), 'k')
        plt.plot(t_adv, z_l, 'r.-')
        plt.plot(t_adv, z_l_f(t_adv), 'k')
        plt.plot(t_adv, z_r, 'y.-')
        plt.plot(t_adv, z_r_f(t_adv), 'k')
        plt.show()

    if 3 in plots:
        plt.figure(3)
        plt.plot(t_adv, z0-z0_f(t_adv), 'b.-')
        plt.plot(t_adv, z_l-z0_f(t_adv), 'r.-')
        plt.plot(t_adv, z_r-z0_f(t_adv),  'y.-')
        plt.plot(t_adv, z_r-z_l, 'g.-')
        plt.show()
    
    q_shift = np.zeros_like(q_z_t)
    uz_center_shift = np.zeros_like(q_z_t)
    
    dt_adv = t_adv[1]-t_adv[0]
    for it in range(len(q_shift[:, 0])):
        iz = int((z0_f(dt_adv*it)-z_l[0])/dz) % nz
        q_shift[it, nz-iz:] = q_z_t[it, :iz]
        q_shift[it, :nz-iz] = q_z_t[it, iz:]
        uz_center_shift[it, nz-iz:] = uz_z_t_center[it, :iz]
        uz_center_shift[it, :nz-iz] = uz_z_t_center[it, iz:]

    if 4 in plots:
        plt.figure(4)
        plt.imshow(q_z_t/u_mean, origin='lower', cmap="viridis",
                   extent=[0,L,0,dt_adv*len(t_adv)])
        plt.show()

    if 5 in plots:
        fig = plt.figure(5)
        cax = plt.imshow(q_shift/u_mean, origin='lower', cmap="viridis",
                   extent=[0,L,0,dt_adv*len(t_adv)])
        fig.colorbar(cax)
        plt.show()

    if 6 in plots:
        plt.figure(6)
        plt.imshow(uz_z_t_center/u_mean, origin='lower', cmap="viridis",
                   extent=[0,L,0,dt_adv*len(t_adv)])
        plt.show()

    if 7 in plots:
        fig = plt.figure(7)
        cax = plt.imshow(uz_center_shift/u_mean, origin='lower', cmap="viridis",
                   extent=[0,L,0,dt_adv*len(t_adv)])
        fig.colorbar(cax)
        plt.show()
