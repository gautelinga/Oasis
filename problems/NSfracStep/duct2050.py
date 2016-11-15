__author__ = "Mads Holst Aagaard Madsen <bkl886@alumni.ku.dk>;" + \
             "Gaute Linga <linga@nbi.dk"
__date__ = "2016-05-04"
__copyright__ = "Copyright (C) 2016 " + __author__

from ..NSfracStep import *
import cPickle # to load parameters
from fenicstools import interpolate_nonmatching_mesh, StructuredGrid
import mpi4py
import h5py
from os import getcwd, makedirs
import numpy as np

folder = "duct2050_results"
# restart_folder = folder + "/data/4/Checkpoint"
restart_folder = None


comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# This script loads from a given snapshot at a given timestep (locally
# in hdf5 timeseries file) instead of the pseudo-laminar scaling that
# Mads implemented.

init_folder = "duct2050_results/data/2/Timeseries/"
h5fu_str = init_folder + "u_from_tstep_0.h5"
h5fp_str = init_folder + "p_from_tstep_0.h5"
step = "999"  # which timestep within the timeseries do we initialize from?

# Body force
F_0 = 6.5e-6  # approx Re=2100

# Viscosity
nu = 9.e-6


def make_dof_coords(S):
    dofmap = S.dofmap()
    my_first, my_last = dofmap.ownership_range()
    x = S.tabulate_dof_coordinates().reshape((-1, 3))
    unowned = dofmap.local_to_global_unowned()
    dofs = filter(lambda dof: dofmap.local_to_global_index(dof)
                  not in unowned,
                  xrange(my_last-my_first))
    x = x[dofs]
    return x


def make_xdict(x_data):
    if rank == 0:
        xdict = dict([(tuple(x_list), i) for i, x_list in
                      enumerate(x_data.tolist())])
    else:
        xdict = None
    xdict = comm.bcast(xdict, root=0)
    return xdict


def set_val(f, f_data, x, xdict):
    vec = f.vector()
    values = vec.get_local()
    values[:] = [f_data[xdict[tuple(x_val)]] for x_val in x.tolist()]
    vec.set_local(values)
    vec.apply('insert')


# Create a mesh here
def mesh(refine=1, **params):
    # Reads the mesh from the timeseries file
    mesh = Mesh()
    h5fmesh = HDF5File(mesh.mpi_comm(), h5fu_str, "r")
    h5fmesh.read(mesh, "/Mesh/0", False)
    h5fmesh.close()
    return mesh


class PeriodicDomain(SubDomain):
    def inside(self, x, on_boundary):
        return bool(near(x[2], 0) and on_boundary)
    def map(self, x, y):
        y[0] = x[0]
        y[1] = x[1]
        y[2] = x[2] - 40.0


constrained_domain = PeriodicDomain()

# If restarting from previous solution then read in parameters
if restart_folder:
    restart_folder = path.join(getcwd(), restart_folder)
    f = open(path.join(restart_folder, 'params.dat'), 'r')
    NS_parameters.update(cPickle.load(f))
    NS_parameters['dt'] = 0.2
    NS_parameters['checkpoint'] = 1000
    NS_parameters['save_step'] = 150
    NS_parameters['T'] += 10000 * NS_parameters['save_step'] * NS_parameters['dt']
    NS_parameters['restart_folder'] = restart_folder
    NS_parameters['update_statistics'] = 10000 # new
    NS_parameters['save_statistics'] = 10000 # new
    NS_parameters['check_flux'] = 100 # new
    globals().update(NS_parameters)
else:
    # Override some problem specific parameters
    NS_parameters.update(dict(
        T=10000 * 0.2 * 150,
        dt=0.2,
        nu=nu,
        # Re = Re,
        # Re_tau = Re_tau,
        checkpoint=1000,
        plot_interval=1,
        save_step=150,
        folder=folder,
        max_iter=1,
        velocity_degree=1,
        use_krylov_solvers=True,
        check_flux=50,
        update_statistics=10000,
        save_statistics=10000
    )
    )


def walls(x, on_boundary):
    return on_boundary and ((near(x[1], 0.0) or near(x[1], 1.0) or
                             near(x[0], 0.0) or near(x[0], 1.0)) or
                            (x[2] > 0 and x[2] < 40.0))


def create_bcs(V, sys_comp, **NS_namespace):
    bcs = dict((ui, []) for ui in sys_comp)    
    bc0  = DirichletBC(V, 0., walls)
    bcs['u0'] = [bc0]
    bcs['u1'] = [bc0]
    bcs['u2'] = [bc0]
    return bcs


def body_force(**NS_namespace):
    return Constant((0.0, 0.0, F_0))


def initialize(V, q_, q_1, q_2, bcs, restart_folder, **NS_namespace):
    """ Initialize from timeseries file """
    if restart_folder is None:
        S = FunctionSpace(V.mesh(),
                          V.ufl_element().family(),
                          V.ufl_element().degree())

        with h5py.File(h5fu_str, "r") as h5fu:
            u_data = np.array(h5fu.get("VisualisationVector/"+step))
            x_data = np.array(h5fu.get("Mesh/0/coordinates"))
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
        

def inlet(x, on_bnd):
    return on_bnd and near(x[2], 0.)


def pre_solve_hook(V, u_, mesh, AssignedVectorFunction, newfolder, MPI,
                   mpi_comm_world, **NS_namespace):    
    """Called prior to time loop"""
    statsfolder = path.join(newfolder, "Stats")
    if MPI.rank(mpi_comm_world()) == 0:
        try:
            makedirs(statsfolder)
        except:
            pass
    uv = AssignedVectorFunction(u_)

    Nx = 10
    Ny = 10
    N = [Nx, Ny]
    origin = [0.5/Nx, 0.5/Ny, 40./2]
    vectors = [[1., 0., 0.], [0., 1., 0.]]
    dL = [1.-1./Nx, 1.-1./Ny]

    stats = StructuredGrid(V, N, origin, vectors, dL, statistics=True)

    # Create FacetFunction to compute flux
    Inlet = AutoSubDomain(inlet)
    facets = FacetFunction('size_t', mesh, 0)
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
        if MPI.rank(mpi_comm_world()) == 0:
            with open(statsfolder + "/dump_flux.dat", "a") as fluxfile:
                fluxfile.write("%d %.8f \n" % (tstep, u1))


def theend(newfolder, tstep, stats, **NS_namespace):
    """Store statistics before exiting"""
    statsfolder = path.join(newfolder, "Stats")
    stats.toh5(1, tstep, filename=statsfolder+"/dump_vz_{}.h5".format(tstep))
