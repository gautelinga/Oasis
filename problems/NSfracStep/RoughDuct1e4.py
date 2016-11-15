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
import random
import itertools

folder = "RoughDuct1e4_results"
# restart_folder = folder + "/data/4/Checkpoint"
restart_folder = None

comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# This script loads from a given snapshot at a given timestep (locally
# in hdf5 timeseries file) instead of the pseudo-laminar scaling that
# Mads implemented.

start_from_scratch = False
mesh_file = "mesh/rough_duct.h5"
info_red("Mesh: " + mesh_file)

# Kept for later...
init_folder = "RoughDuct3800_results/data/1/Timeseries/"
h5fu_str = init_folder + "u_from_tstep_0.h5"
h5fp_str = init_folder + "p_from_tstep_0.h5"
step = "655"  # which timestep within the timeseries do we initialize from?

# Body force
F_0 = 4.0e-5  #

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


def tabulate_inlet_wall_nodes(nodes, elems):
    info_green("Building inlet wall nodes")
    node_ids = set(np.where(nodes[:, 2] == 0.)[0])
    faces = []
    for i in xrange(rank, len(elems), size):
        row = elems[i, :]
        nodes_in_node_ids = []
        for node in row:
            if node in node_ids:
                nodes_in_node_ids.append(node)
        if len(nodes_in_node_ids) == 3:
            nodes_in_node_ids.sort()
            faces.append(tuple(nodes_in_node_ids))
    data = comm.gather(faces, root=0)
    if rank == 0:
        faces = np.array(list(itertools.chain.from_iterable(data)))
    faces = comm.bcast(faces, root=0)
    edges = set()
    for face in faces:
        for i in xrange(3):
            seg = [face[i], face[(i+1) % 3]]
            seg.sort()
            seg = tuple(seg)
            if seg in edges:
                edges.remove(seg)
            else:
                edges.add(seg)
    edges = np.array(list(edges))
    wall_nodes = np.unique(edges.flatten())

    wall_coords = set()
    for node in wall_nodes:
        wall_coords.add(tuple(nodes[node, 0:2]))
    return wall_coords

if start_from_scratch:
    with h5py.File(mesh_file, "r") as h5f:
        nodes = np.array(h5f["mesh/coordinates"])
        elems = np.array(h5f["mesh/topology"])
else:
    with h5py.File(h5fu_str, "r") as h5f:
        nodes = np.array(h5f["Mesh/0/coordinates"])
        elems = np.array(h5f["Mesh/0/topology"])
inlet_wall_coords = tabulate_inlet_wall_nodes(nodes, elems)


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
    if start_from_scratch:
        h5fmesh = HDF5File(mesh.mpi_comm(), mesh_file, "r")
        h5fmesh.read(mesh, "/mesh", False)
    else:
        h5fmesh = HDF5File(mesh.mpi_comm(), h5fu_str, "r")
        h5fmesh.read(mesh, "/Mesh/0", False)
    h5fmesh.close()
    return mesh


class PeriodicDomain(SubDomain):
    def inside(self, x, on_boundary):
        return bool(near(x[2], 0.) and on_boundary)

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
    NS_parameters['T'] += 500 * NS_parameters['save_step'] * NS_parameters['dt']
    NS_parameters['restart_folder'] = restart_folder
    NS_parameters['update_statistics'] = 10000 # new
    NS_parameters['save_statistics'] = 10000 # new
    NS_parameters['check_flux'] = 100 # new
    globals().update(NS_parameters)
else:
    # Override some problem specific parameters
    NS_parameters.update(dict(
        T=500.0 * 0.2 * 300,
        dt=0.2,  # crank up? Usual: 0.2
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
    return on_boundary and (tuple(x[0:2]) in inlet_wall_coords or
                            (x[2] > 0.0 and x[2] < 40.0))


def create_bcs(V, sys_comp, **NS_namespace):
    bcs = dict((ui, []) for ui in sys_comp)
    bc0  = DirichletBC(V, 0., walls)
    bcs['u0'] = [bc0]
    bcs['u1'] = [bc0]
    bcs['u2'] = [bc0]
    return bcs


class RandomStreamVector(Expression):
    def __init__(self):
        random.seed(2 + MPI.rank(mpi_comm_world()))
    def eval(self, values, x):
        values[0] = 0.0005*random.random()
        values[1] = 0.0005*random.random()
        values[2] = 0.0005*random.random()
    def value_shape(self):
        return (3,)


def body_force(**NS_namespace):
    return Constant((0.0, 0.0, F_0))


def initialize(V, q_, q_1, q_2, bcs, restart_folder, **NS_namespace):
    """ Initialize from timeseries file """
    if start_from_scratch:
        # Initialize using a perturbed flow. Create random streamfunction
        # info_red("Creating Vv")
        # Vv = VectorFunctionSpace(V.mesh(), V.ufl_element().family(), V.ufl_element().degree())
        # info_red("         psi")
        # psi = interpolate(RandomStreamVector(), Vv)
        # info_red("         u0")
        # u0 = project(curl(psi), Vv)
        # info_red("         u0x")
        # # u0x = project(u0[0], V, bcs=bcs['u0'])
        # u0x = interpolate_nonmatching_mesh(u0[0], V)
        # info_red("         u0y")
        # # u1x = project(u0[1], V, bcs=bcs['u0'])
        # u1x = interpolate_nonmatching_mesh(u0[1], V)
        # info_red("         u0z")
        # # u2x = project(u0[2], V, bcs=bcs['u0'])
        # u2x = interpolate_nonmatching_mesh(u0[2], V)

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
