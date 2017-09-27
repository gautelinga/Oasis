import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import mpi4py.MPI as MPI
import h5py

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot cylinder.")
    parser.add_argument("folder", type=str, help="Folder")
    args = parser.parse_args()
    return args


def integrate_radially(r, theta, u_z):
    theta_loc = np.hstack((theta, theta[0]+np.pi))
    r_loc = r

    Ntheta, Nr, Nz = u_z.shape
    uz_z_loc = np.zeros((Ntheta+1, Nr, Nz))
    uz_z_loc[:Ntheta, :, :] = u_z
    uz_z_loc[Ntheta, :, :] = u_z[0, :, :]
    uz_z = np.zeros(Nz)
    for i in range(1, Nr):
        for j in range(1, Ntheta+1):
            uz_z[:] += abs(theta_loc[j]-theta_loc[j-1])*abs(
                r_loc[i]-r_loc[i-1])/(3.*4.)*(
                    (2*r_loc[i]+r_loc[i-1])*(
                        uz_z_loc[j, i, :]+uz_z_loc[j-1, i, :])
                    + (2*r_loc[i-1]+r_loc[i])*(
                        uz_z_loc[j, i-1, :]+uz_z_loc[j-1, i-1, :]))
    return uz_z


def plot_rz(rv, zv, u_z):
    # rz plane, averaged over theta
    r_loc = rv[0, :, :]
    z_loc = zv[0, :, :]
    uz_loc = u_z.mean(0)

    plt.figure()
    plt.pcolor(r_loc, z_loc, uz_loc)


def plot_rtheta(rc, thetav, u_z):
    r_loc = np.vstack((rv[:, :, 0], rv[0, :, 0]))
    theta_loc = np.vstack((thetav[:, :, 0], thetav[0, :, 0]))
    x_loc = r_loc*np.cos(theta_loc)
    y_loc = r_loc*np.sin(theta_loc)
    uz_loc = u_z.mean(2)
    uz_loc = np.vstack((uz_loc[:, :], uz_loc[0, :]))

    plt.figure()
    plt.pcolormesh(x_loc, y_loc, uz_loc)


def plot_thetaz(thetav, zv, u_z):
    theta_loc = thetav[:, 0, :]
    z_loc = zv[:, 0, :]
    uz_loc = u_z.mean(1)

    plt.figure()
    plt.pcolormesh(theta_loc, z_loc, uz_loc)


if __name__ == "__main__":
    args = parse_args()

    h5filename = os.path.join(args.folder,
                              "Interpolated", "snapshot.h5")

    if not os.path.exists(h5filename):
        if rank == 0:
            print "Could not find snapshot in {}".format(args.folder)

    with h5py.File(h5filename, "r") as h5f:
        nodes_xyz = np.array(h5f["Mesh/coordinates"])
        elems = np.array(h5f["Mesh/elements"], dtype=int)
        nodes_cyl = np.array(h5f["Mesh/cylinder_coordinates"])

        r = np.unique(nodes_cyl[:, 0])
        theta = np.unique(nodes_cyl[:, 1])
        z = np.unique(nodes_cyl[:, 2])

        Nr = len(r)
        Ntheta = len(theta)
        Nz = len(z)

        rv = nodes_cyl[:, 0].reshape((Ntheta, Nr, Nz))
        thetav = nodes_cyl[:, 1].reshape((Ntheta, Nr, Nz))
        zv = nodes_cyl[:, 2].reshape((Ntheta, Nr, Nz))

        xv = nodes_xyz[:, 0].reshape((Ntheta, Nr, Nz))
        yv = nodes_xyz[:, 1].reshape((Ntheta, Nr, Nz))

        sintheta = np.sin(thetav)
        costheta = np.cos(thetav)

        dset_ids = sorted([int(key) for key in h5f["Data"]])
        fields = [str(key) for key in h5f["Data/{}".format(dset_ids[0])]]

        fig_dir = os.path.join(args.folder,
                               "Interpolated",
                               "Plots",
                               "uz_xz")
        data_dir = os.path.join(args.folder,
                                "Interpolated",
                                "Data")
        
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        np.savetxt(os.path.join(data_dir, "z.dat"), z)
        np.savetxt(os.path.join(data_dir, "t.dat"), np.array(dset_ids))
        
        plt.figure(frameon=False,
                   figsize=(zv.max()-zv.min(), xv.max()-xv.min()))
        plt.axes().set_aspect("equal")
        plt.tick_params(
            axis="both",
            which="both",
            bottom="off",
            left="off",
            labelbottom="off",
            labelleft="off")
        plt.gca().set_axis_off()

        turb_z_t = np.zeros((len(dset_ids), len(z)))
        uz_z_t = np.zeros_like(turb_z_t)
        turb_z_t_center = np.zeros_like(turb_z_t)
        uz_z_t_center = np.zeros_like(turb_z_t)

        for i, dset_id in enumerate(dset_ids[:]):
            print "dset = {}".format(dset_id)
            
            u_data = np.array(h5f["Data/{}/{}".format(
                dset_id, "u")])

            u_x = u_data[:, 0].reshape((Ntheta, Nr, Nz))
            u_y = u_data[:, 1].reshape((Ntheta, Nr, Nz))
            u_z = u_data[:, 2].reshape((Ntheta, Nr, Nz))

            u_r = u_x*costheta + u_y*sintheta
            u_theta = - u_x*sintheta + u_y*costheta
            
            # Integrals
            uz_z = integrate_radially(r, theta, u_z)
            uz_z_t[i, :] = uz_z
            
            turb = u_x**2 + u_y**2
            turb_z = integrate_radially(r, theta, turb)
            turb_z_t[i, :] = turb_z

            # Centerline
            turb_z_t_center[i, :] = turb[0, 0, :]
            uz_z_t_center[i, :] = u_z[0, 0, :]

            # Cross
            x_loc = np.vstack((xv[Ntheta/2, ::-1, :], xv[0, 1:, :]))
            z_loc = np.vstack((zv[Ntheta/2, ::-1, :], zv[0, 1:, :]))
            u_z_loc = np.vstack((u_z[Ntheta/2, ::-1, :], u_z[0, 1:, :]))

            figname = os.path.join(fig_dir,
                                   "uz_xz_{}.png".format(dset_id))

            if rank == 0:
                plt.pcolormesh(z_loc, x_loc, u_z_loc)
                plt.savefig(figname,
                            bbox_inches="tight",
                            transparent=True,
                            pad_inches=0)

                os.system("convert {figname} -trim {figname}".format(
                    figname=figname))

        if rank == 0:
            np.savetxt(os.path.join(data_dir, "turb_z_t.dat"),
                       turb_z_t)
            np.savetxt(os.path.join(data_dir, "uz_z_t.dat"),
                       uz_z_t)
            np.savetxt(os.path.join(data_dir, "turb_z_t_center.dat"),
                       turb_z_t_center)
            np.savetxt(os.path.join(data_dir, "uz_z_t_center.dat"),
                       uz_z_t_center)

    print "yessda"
