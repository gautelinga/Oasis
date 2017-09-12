import argparse
import dolfin as df
import os
import glob
import h5py
import numpy as np
from common.ductutils import numpy_to_dolfin, make_dof_coords, make_xdict, set_val
from common import info_red
from fenicstools import Probes


def get_args():
    parser = argparse.ArgumentParser(description="Take a snapshot.")
    parser.add_argument("folder", type=str, help="Folder.")
    parser.add_argument("-step", type=int, default=None, help="Step.")
    parser.add_argument("-Nz", type=int, default=10, help="Nz")
    parser.add_argument("-Nr", type=int, default=10, help="Nr")
    parser.add_argument("-Ntheta", type=int, default=10, help="Ntheta")
    args = parser.parse_args()
    return args


def get_step(init_step, h5f):
    if init_step is not None and isinstance(init_step, int):
        step = init_step
    else:
        step = str(max([int(step) for step
                        in h5f["VisualisationVector"]]))
    return step


def initialize(nodes, elems):
    mesh = numpy_to_dolfin(nodes, elems)
    S = df.FunctionSpace(mesh, "CG", 1)
    V = df.VectorFunctionSpace(mesh, "CG", 1)

    x = make_dof_coords(S)
    xdict = make_xdict(nodes)

    return mesh, S, V, x, xdict


def load_data(init_step, h5f_str):
    with h5py.File(h5f_str, "r") as h5fu:
        step = get_step(init_step, h5fu)
        data = np.array(h5fu.get("VisualisationVector/" + step))
    return data, step


if __name__ == "__main__":
    args = get_args()
    init_folder = args.folder
    init_step = args.step

    init_tstep = max([int(os.path.basename(string)[13:-3])
                      for string in glob.glob(
                              os.path.join(init_folder,
                                           "Timeseries/u_from_tstep_*.h5"))])

    h5fu_str = os.path.join(init_folder,
                            "Timeseries/u_from_tstep_{}.h5".format(init_tstep))
    h5fp_str = os.path.join(init_folder,
                            "Timeseries/p_from_tstep_{}.h5".format(init_tstep))

    for h5f_str in [h5fu_str, h5fp_str]:
        if not os.path.exists(h5fu_str):
            info_red("Could not find file: " + h5f_str)
            exit()

    with h5py.File(h5fu_str, "r") as h5fu:
        nodes = np.array(h5fu.get("Mesh/0/mesh/geometry"))
        elems = np.array(h5fu.get("Mesh/0/mesh/topology"))

    mesh, S, V, x, xdict = initialize(nodes, elems)

    x_max = nodes.max(0)
    x_min = nodes.min(0)
    x_mid = (x_max + x_min)/2
    R = ((x_max - x_min)/2)[:2].max()
    Lz = x_max[2]-x_min[2]

    print x_mid, R, Lz

    i_z = np.arange(args.Nz)
    i_r = np.arange(args.Nr)
    i_theta = np.arange(args.Ntheta)

    r = np.sqrt((2*i_r+1)/(2*args.Nr))*R
    z = Lz*(2.*i_z+1.)/(2*args.Nz)
    theta = 2.*np.pi*(2.*i_theta+1.)/(2.*args.Ntheta)

    rv, thetav, zv = np.meshgrid(r, theta, z)

    pts_cyl = np.array(zip(rv.flatten(),
                           thetav.flatten(),
                           zv.flatten()))

    xv = rv*np.cos(thetav)
    yv = rv*np.sin(thetav)

    pts_xyz = np.array(zip(xv.flatten(), yv.flatten(), zv.flatten()))

    probes = Probes(pts_xyz.flatten(), V)

    u_x = [df.Function(S) for _ in range(3)]
    u = df.Function(V)
    p = df.Function(S)

    u_data, u_step = load_data(init_step, h5fu_str)
    p_data, p_step = load_data(init_step, h5fp_str)
    assert(u_step == p_step)

    for i in range(3):
        set_val(u_x[i], u_data[:, i], x, xdict)
        u.sub(i).assign(u_x[i])
    set_val(p, p_data[:], x, xdict)

    probes(u)
    u_probes = probes.array(0)

    print u_probes.max()
