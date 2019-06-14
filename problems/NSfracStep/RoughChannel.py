__author__ = "Gaute Linga <linga@nbi.dk>"
__date__ = "2017"
__copyright__ = "Copyright (C) 2017 " + __author__

from ..NSfracStep import *
import pickle as cPickle # to load parameters
from fenicstools import interpolate_nonmatching_mesh, StructuredGrid
import mpi4py
import h5py
from os import getcwd, makedirs, path
import numpy as np
import random
import itertools
from common.ductutils import *

# folder = "RoughChannel_A2e-1_results"
# folder = "SmoothChannel_results"
folder = "RoughChannel_A1e-1_results"
# folder = "RoughChannel_A1_results"
# folder = "RoughChannel_A8e-1_results"
# folder = "RoughChannel_results"
# restart_folder = folder + "/data/4/Checkpoint"
restart_folder = None

comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# This script loads from a given snapshot at a given timestep (locally
# in hdf5 timeseries file) instead of the pseudo-laminar scaling that
# Mads implemented.

start_from_scratch = False
# mesh_file = "mesh/rough_channel_H8e-1_A2e-1_10x10x1.h5"
# mesh_file = "mesh/rough_channel_H8e-1_A1e-1_10x10x1.h5"
mesh_file = "mount_lizard/mesh/rough_channel_H8e-1_A1e-1_10x10x1.h5"
# mesh_file = "mesh/smooth_channel_10x10x1.h5"
# mesh_file = "mesh/rough_channel_H8e-1_A1_10x10x1.h5"
# mesh_file = "mesh/rough_channel_H8e-1_A8e-1_10x10x1.h5"
# mesh_file = "mesh/rough_channel_H8e-1_10x10x1_2.h5"
if path.isfile(mesh_file):
    info_red("Mesh: " + mesh_file)
else:
    exit("Couldn't find mesh: " + mesh_file)

# Kept for later...
# init_folder = "RoughChannel_results/data/2/Timeseries/"
# init_folder = "init_states/rough_channel_H8e-1_10x10x1_2/Re860/"
# init_folder = "SmoothChannel_results/data/1/Timeseries/"
# init_folder = folder + "/data/6/Timeseries/"
init_folder = "mount_lizard/" + folder + "/data/6/Timeseries/"
h5fu_str = init_folder + "u_from_tstep_0.h5"
h5fp_str = init_folder + "p_from_tstep_0.h5"

step = "4999"  # which timestep within the timeseries do we initialize from?

# Viscosity
nu = 9.e-6

# Body force
# F_0 = 1e-8
# F_0 = 3.75e-8
# F_0 = 1e-7  # Re=42
# F_0 = 3.75e-7  # Re=125
# F_0 = 7.5e-7  # Re=195 # Smooth
F_0 = 1.0e-6
# F_0 = 1.3e-6
# F_0 = 1.5e-6  # Re=290
# F_0 = 3e-6  # Re=420
# F_0 = 6e-6  # Re=600
# F_0 = 1.2e-5  # Re=860
# F_0 = 2.4e-5

# Dimensions
Lx = 10.
Ly = 10.


def build_wall_coords():
    with h5py.File(mesh_file, "r") as h5f:
        nodes = np.array(h5f["mesh/coordinates"])
        elems = np.array(h5f["mesh/topology"])
    inlet_wall_coords_x = tabulate_wall_nodes(nodes, elems, 0)
    inlet_wall_coords_y = tabulate_wall_nodes(nodes, elems, 1)
    nodes_side = nodes[nodes[:, 0] == 0., :]
    nodes_edge = nodes_side[nodes_side[:, 1] == 0., :]
    z_min = nodes_edge[:, 2].min()
    z_max = nodes_edge[:, 2].max()
    corners_z = set([z_min, z_max])
    return inlet_wall_coords_x, inlet_wall_coords_y, corners_z

inlet_wall_coords_x, inlet_wall_coords_y, corners_z = build_wall_coords()

# Create a mesh here
def mesh(refine=1, **params):
    # Reads the mesh from the timeseries file
    mesh = Mesh()
    h5fmesh = HDF5File(mesh.mpi_comm(), mesh_file, "r")
    h5fmesh.read(mesh, "/mesh", False)
    h5fmesh.close()
    return mesh


class PeriodicDomain(SubDomain):
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two slave edges
        return bool((near(x[0], 0.) or near(x[1], 0.)) and 
                    (not (near(x[0], Lx) or near(x[1], Ly))) and on_boundary)

    def map(self, x, y):
        if near(x[0], Lx) and near(x[1], Ly):
            y[0] = x[0] - Lx
            y[1] = x[1] - Ly
            y[2] = x[2]
        elif near(x[0], Lx):
            y[0] = x[0] - Lx
            y[1] = x[1]
            y[2] = x[2]
        else:  # near(x[1], Ly):
            y[0] = x[0]
            y[1] = x[1] - Ly
            y[2] = x[2]


constrained_domain = PeriodicDomain()

# If restarting from previous solution then read in parameters
if restart_folder:
    restart_folder = path.join(getcwd(), restart_folder)
    f = open(path.join(restart_folder, 'params.dat'), 'r')
    NS_parameters.update(cPickle.load(f))
    NS_parameters['dt'] = 0.2 * 5
    NS_parameters['checkpoint'] = 1000
    NS_parameters['save_step'] = 150
    NS_parameters['T'] += 500 * NS_parameters['save_step'] * NS_parameters['dt']
    NS_parameters['restart_folder'] = restart_folder
    NS_parameters['update_statistics'] = 10000 # new
    NS_parameters['save_statistics'] = 10000 # new
    NS_parameters['check_flux'] = 50 # new
    globals().update(NS_parameters)
else:
    # Override some problem specific parameters
    NS_parameters.update(dict(
        T=500.0 * 0.2 * 10 * 300,
        dt=0.2 * 5,  # crank up? Usual: 0.2
        nu=nu,
        # Re = Re,
        # Re_tau = Re_tau,
        checkpoint=10000000,
        plot_interval=1,
        save_step=150,
        folder=folder,
        max_iter=1,
        velocity_degree=1,
        use_krylov_solvers=True,
        check_flux=50,
        update_statistics=10000000,
        save_statistics=10000000
    )
    )


def walls(x, on_boundary):
    tup_x = tuple(x[[1, 2]])
    tup_y = tuple(x[[0, 2]])
    in_bulk_x = x[0] > 0. and x[0] < Lx
    in_bulk_y = x[1] > 0. and x[1] < Ly
    return on_boundary and ((in_bulk_x and in_bulk_y) or
                            (tup_x in inlet_wall_coords_x and in_bulk_y) or
                            (tup_y in inlet_wall_coords_y and in_bulk_x) or
                            (not in_bulk_x and not in_bulk_y and
                             x[2] in corners_z))


def create_bcs(V, sys_comp, **NS_namespace):
    bcs = dict((ui, []) for ui in sys_comp)
    bc0  = DirichletBC(V, 0., walls)
    bcs['u0'] = [bc0]
    bcs['u1'] = [bc0]
    bcs['u2'] = [bc0]
    return bcs


def body_force(**NS_namespace):
    return Constant((F_0, 0., 0.))


def initialize(V, q_, q_1, q_2, bcs, restart_folder, **NS_namespace):
    """ Initialize from timeseries file """
    if start_from_scratch:
        # initialize vectors at two timesteps
        q_1['u0'].vector()[:] = 0.  # u0x.vector()[:]
        q_1['u1'].vector()[:] = 0.  # u1x.vector()[:]
        q_1['u2'].vector()[:] = 0.  # u2x.vector()[:]
        q_2['u0'].vector()[:] = q_1['u0'].vector()[:]
        q_2['u1'].vector()[:] = q_1['u1'].vector()[:]
        q_2['u2'].vector()[:] = q_1['u2'].vector()[:]
    elif restart_folder is None:
        S = FunctionSpace(V.mesh(),
                          V.ufl_element().family(),
                          V.ufl_element().degree())

        with h5py.File(h5fu_str, "r") as h5fu:
            u_data = np.array(h5fu.get("VisualisationVector/"+step))
            x_data = np.array(h5fu.get("Mesh/0/mesh/geometry"))
        with h5py.File(h5fp_str, "r") as h5fp:
            p_data = np.array(h5fp.get("VisualisationVector/"+step))

        x = make_dof_coords(S)
        xdict = make_xdict(x_data)

        u0x = Function(S)
        u0y = Function(S)
        u0z = Function(S)
        p0 = Function(S)

        set_val(u0x, u_data[:, 0], x, xdict)
        set_val(u0y, u_data[:, 1], x, xdict)
        set_val(u0z, u_data[:, 2], x, xdict)
        set_val(p0, p_data[:], x, xdict)

        info_green("Projecting u0x...")
        u0x = interpolate_nonmatching_mesh(u0x, V)
        info_green("Projecting u0y...")
        u0y = interpolate_nonmatching_mesh(u0y, V)
        info_green("Projecting u0z...")
        u0z = interpolate_nonmatching_mesh(u0z, V)
        info_green("Projecting p0...")
        p0 = interpolate_nonmatching_mesh(p0, V)

        # initialize vectors at two timesteps
        q_['u0'].vector()[:] = u0x.vector()[:]
        q_['u1'].vector()[:] = u0y.vector()[:]
        q_['u2'].vector()[:] = u0z.vector()[:]
        q_['p'].vector()[:] = p0.vector()[:]
        q_1['u0'].vector()[:] = q_['u0'].vector()[:]
        q_1['u1'].vector()[:] = q_['u1'].vector()[:]
        q_1['u2'].vector()[:] = q_['u2'].vector()[:]
        q_1['p'].vector()[:] = q_['p'].vector()[:]
        q_2['u0'].vector()[:] = q_['u0'].vector()[:]
        q_2['u1'].vector()[:] = q_['u1'].vector()[:]
        q_2['u2'].vector()[:] = q_['u2'].vector()[:]


def inlet(x, on_boundary):
    return on_boundary and near(x[0], 0.)


def pre_solve_hook(V, u_, mesh, AssignedVectorFunction, newfolder, MPI,
                   **NS_namespace):
    """Called prior to time loop"""
    statsfolder = path.join(newfolder, "Stats")
    if MPI.rank(MPI.comm_world) == 0:
        try:
            makedirs(statsfolder)
        except:
            pass
    uv = AssignedVectorFunction(u_)

    Nx = 40
    Ny = 40
    N = [Nx, Ny]
    origin = [0.5*Lx/Nx, 0.5*Ly/Ny, 0.5]
    vectors = [[1., 0., 0.], [0., 1., 0.]]
    dL = [Lx*(1.-1./Nx), Ly*(1.-1./Ny)]

    stats = StructuredGrid(V, N, origin, vectors, dL, statistics=True)

    # Create FacetFunction to compute flux
    Inlet = AutoSubDomain(inlet)
    facets = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    Inlet.mark(facets, 1)
    normal = FacetNormal(mesh)

    return dict(uv=uv, stats=stats, facets=facets, normal=normal)


def temporal_hook(q_, u_, V, tstep, uv, stats, update_statistics,
                  newfolder, folder, check_flux, save_statistics, mesh,
                  facets, normal, check_if_reset_statistics,
                  **NS_namespace):
    # print timestep
    info_red("tstep = {}".format(tstep))       
    if check_if_reset_statistics(folder):
        info_red("Resetting statistics")
        stats.probes.clear()

    if tstep % update_statistics == 0:
        stats(q_['u0'], q_['u1'], q_['u2'])

    if tstep % save_statistics == 0:
        statsfolder = path.join(newfolder, "Stats")
        stats.toh5(1, tstep,
                   filename=statsfolder+"/dump_vz_{}.h5".format(tstep))

    if tstep % check_flux == 0:
        statsfolder = path.join(newfolder, "Stats")
        u1 = assemble(dot(u_, normal) *
                      ds(1, domain=mesh, subdomain_data=facets))
        if MPI.rank(MPI.comm_world) == 0:
            with open(statsfolder + "/dump_flux.dat", "a") as fluxfile:
                fluxfile.write("%d %.8f \n" % (tstep, u1))

    return dict()


def theend(newfolder, tstep, stats, **NS_namespace):
    """Store statistics before exiting"""
    statsfolder = path.join(newfolder, "Stats")
    stats.toh5(1, tstep, filename=statsfolder+"/dump_vz_{}.h5".format(tstep))
