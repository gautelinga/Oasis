__author__ = "Mads Holst Aagaard Madsen <bkl886@alumni.ku.dk>;" + \
             "Gaute Linga <linga@nbi.dk>"
__date__ = "2016-05-04"
__copyright__ = "Copyright (C) 2016 " + __author__

from ..NSfracStep import *
import cPickle # to load parameters
from fenicstools import interpolate_nonmatching_mesh, StructuredGrid
import mpi4py
import h5py
from os import getcwd, makedirs, path
import numpy as np
import random
import itertools
from common.ductutils import tabulate_inlet_wall_nodes, \
   make_dof_coords, make_xdict, set_val, generate_puff_spark, \
   numpy_to_dolfin, get_mesh_properties, build_wall_coords
import os
import glob


# Some default values
folder = "RoughPipe_results/"
restart_folder = None
mesh_file = "mesh/rough_pipe_N20.h5"

comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Viscosity
nu = 9.e-6

# Body force
F = 6e-6

# Timestep
dt = 0.2
Nt = 500
save_step = 150
check_flux = 50  # 50

# Regulatory time scale
kreg = 8.*nu/0.5

# Create a mesh here
def mesh(mesh_file, **params):
    mesh = Mesh()
    h5fmesh = HDF5File(mesh.mpi_comm(), mesh_file, "r")
    h5fmesh.read(mesh, "/mesh", False)
    h5fmesh.close()
    return mesh


class PeriodicDomain(SubDomain):
    def __init__(self, Lz):
        self.Lz = Lz
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return bool(near(x[2], 0.) and on_boundary)

    def map(self, x, y):
        y[0] = x[0]
        y[1] = x[1]
        y[2] = x[2] - self.Lz


# If restarting from previous solution then read in parameters
if restart_folder:
    restart_folder = path.join(getcwd(), restart_folder)
    f = open(path.join(restart_folder, 'params.dat'), 'r')
    NS_parameters.update(cPickle.load(f))
    NS_parameters['dt'] = dt
    NS_parameters['checkpoint'] = 1000
    NS_parameters['save_step'] = save_step
    NS_parameters['T'] += Nt * save_step * dt
    NS_parameters['restart_folder'] = restart_folder
    NS_parameters['update_statistics'] = 10000 # new
    NS_parameters['save_statistics'] = 10000 # new
    NS_parameters['check_flux'] = check_flux # new
    globals().update(NS_parameters)
else:
    # Override some problem specific parameters
    NS_parameters.update(
        dict(
            T=Nt * dt * save_step,
            tstep=-1,
            dt=dt,
            nu=nu,
            checkpoint=1000,
            plot_interval=1,
            save_step=save_step,
            folder=folder,
            max_iter=1,
            velocity_degree=1,
            use_krylov_solvers=True,
            check_flux=check_flux,
            update_statistics=10000,
            save_statistics=10000,
            mesh_file=mesh_file,
            F=F,
            init_folder=None,
            init_step=None,
            spark_puff=False,
            N=None,
            scale=1.0,
            mesh_suffix="fine",
            puff_magnitude=2.,
            control="Re",
            Re_target=2000.,
            Kp=10.*kreg,
            Ki=20.*kreg**2,
            Kd=0.05
        )
    )


def create_bcs(V, sys_comp, inlet_wall_coords, Lz, **NS_namespace):
    bcs = dict((ui, []) for ui in sys_comp)

    def walls(x, on_boundary):
        return on_boundary and (tuple(x[0:2]) in inlet_wall_coords or
                                (x[2] > 0.0 and x[2] < Lz))

    bc0 = DirichletBC(V, 0., walls)
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


def body_force(F, **NS_namespace):
    return Constant((0.0, 0.0, F))


def initialize(V, q_, q_1, q_2, bcs, restart_folder, init_folder, init_step,
               spark_puff, puff_center, puff_magnitude, puff_radius, scale,
               mesh,
               **NS_namespace):
    """ Initialize from timeseries file """
    if init_folder is None and restart_folder is None:
        # Start from scratch
        # initialize vectors at two timesteps
        q_1['u0'].vector()[:] = 0.  # u0x.vector()[:]
        q_1['u1'].vector()[:] = 0.  # u1x.vector()[:]
        q_1['u2'].vector()[:] = 0.  # u2x.vector()[:]
        q_2['u0'].vector()[:] = q_1['u0'].vector()[:]
        q_2['u1'].vector()[:] = q_1['u1'].vector()[:]
        q_2['u2'].vector()[:] = q_1['u2'].vector()[:]
    elif restart_folder is None:
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
            if init_step is not None and isinstance(init_step, int):
                step_u = str(init_step)
            else:
                step_u = str(max([int(step) for step
                                  in h5fu["VisualisationVector"]]))
            u_data = scale*np.array(h5fu.get("VisualisationVector/" + step_u))
            nodes = np.array(h5fu.get("Mesh/0/mesh/geometry"))
            elems = np.array(h5fu.get("Mesh/0/mesh/topology"))
        with h5py.File(h5fp_str, "r") as h5fp:
            if init_step is not None and isinstance(init_step, int):
                step_p = str(init_step)
            else:
                step_p = str(max([int(step) for step
                                  in h5fp["VisualisationVector"]]))
            p_data = scale*np.array(h5fp.get("VisualisationVector/" + step_p))
        assert(step_u == step_p)

        other_mesh = numpy_to_dolfin(nodes, elems)
        S = FunctionSpace(other_mesh, "CG", 1)

        x = make_dof_coords(S)
        xdict = make_xdict(nodes)

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
        info_green("Done with projecting (for now).")

        if spark_puff:
            info_green("Making puff!")

            u0z_mean = (assemble(u0z*dx(domain=mesh))/
                        assemble(Constant(1.)*dx(domain=mesh)))

            info_green("Mean velocity: {}".format(u0z_mean))

            with h5py.File("puff/puff.h5", "r") as h5f:
                u_data_puff = np.array(h5f.get("VisualisationVector/0"))
                nodes_puff = np.array(h5f.get("Mesh/0/mesh/geometry"))
                elems_puff = np.array(h5f.get("Mesh/0/mesh/topology"))

            info_green("The puff velocity is scaled by a factor {}.".format(
                puff_magnitude))
            info_green(("The puff radius is set to {} and the puff is"
                        " shifted by ({}, {}, {}).").format(
                            puff_radius, puff_center[0],
                            puff_center[1], puff_center[2]))
            u_data_puff = puff_magnitude*u_data_puff*u0z_mean
            nodes_puff = nodes_puff*puff_radius/0.5
            nodes_puff[:, 0] += puff_center[0]
            nodes_puff[:, 1] += puff_center[1]
            nodes_puff[:, 2] += puff_center[2]

            info_green("Recreating the puff mesh...")
            mesh_puff = numpy_to_dolfin(nodes_puff, elems_puff)
            S_puff = FunctionSpace(mesh_puff, "CG", 1)
            x_puff = make_dof_coords(S_puff)
            xdict_puff = make_xdict(nodes_puff)

            u_x_puff = Function(S_puff)
            u_y_puff = Function(S_puff)
            u_z_puff = Function(S_puff)

            info_green("Setting puff values...")
            set_val(u_x_puff, u_data_puff[:, 0], x_puff, xdict_puff)
            set_val(u_y_puff, u_data_puff[:, 1], x_puff, xdict_puff)
            set_val(u_z_puff, u_data_puff[:, 2], x_puff, xdict_puff)

            info_green("Projecting u0x_puff...")
            u0x_puff = interpolate_nonmatching_mesh(u_x_puff, V)
            info_green("Projecting u0y_puff...")
            u0y_puff = interpolate_nonmatching_mesh(u_y_puff, V)
            info_green("Projecting u0z_puff...")
            u0z_puff = interpolate_nonmatching_mesh(u_z_puff, V)

            u0x.vector()[:] = u0x.vector()[:] + u0x_puff.vector()[:]
            u0y.vector()[:] = u0y.vector()[:] + u0y_puff.vector()[:]
            u0z.vector()[:] = u0z.vector()[:] + u0z_puff.vector()[:]

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
    return on_boundary and near(x[2], 0.)


def pre_solve_hook(V, u_, mesh, AssignedVectorFunction, newfolder, MPI,
                   mpi_comm_world, Lz, **NS_namespace):
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
    origin = [0., 0., Lz/2]
    vectors = [[1., 0., 0.], [0., 1., 0.]]
    dL = [1.-1./Nx, 1.-1./Ny]

    stats = StructuredGrid(V, N, origin, vectors, dL, statistics=True)

    # Create FacetFunction to compute flux
    Inlet = AutoSubDomain(inlet)
    facets = FacetFunction('size_t', mesh, 0)
    Inlet.mark(facets, 1)
    normal = FacetNormal(mesh)

    area = assemble(Constant(1.)*ds(1, domain=mesh, subdomain_data=facets))
    info("Cross sectional area at inlet: " + str(area))

    volume = assemble(Constant(1.)*dx(domain=mesh))
    info("Volume: " + str(volume))
    info("Mean cross sectional area: " + str(volume/Lz))

    return dict(uv=uv, stats=stats, facets=facets, normal=normal,
                area=area, volume=volume)


def temporal_hook(q_, u_, V, tstep, t, uv, stats, update_statistics,
                  newfolder, folder, check_flux, save_statistics, mesh,
                  facets, normal, check_if_reset_statistics, area, volume,
                  Lz, nu, F, u_err_integral, Kp, Ki, Kd, dt, u_err, Re_target,
                  control,
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
        u_axial = -assemble(dot(u_, normal) *
                            ds(1, domain=mesh, subdomain_data=facets))
        e_kin = 0.5*assemble(dot(u_, u_) *
                             ds(1, domain=mesh, subdomain_data=facets))

        n_z = Constant((0., 0., 1.))
        u_z = dot(u_, n_z)
        u_n = u_ - u_z*n_z

        u_axial_vol = assemble(u_z*dx(domain=mesh))
        e_kin_vol = 0.5*assemble(dot(u_, u_)*dx(domain=mesh))
        u_normal_vol = np.sqrt(assemble(dot(u_n, u_n)*dx(domain=mesh)))

        rad_avg = np.sqrt(volume/(Lz*np.pi))
        u_axial_mean = u_axial_vol/volume
        Re = u_axial_mean*2*rad_avg/nu

        turb = u_normal_vol/u_axial_vol

        if control == "Re":
            F_arr = np.zeros(1)
            if MPI.rank(mpi_comm_world()) == 0:
                u_target = float(Re_target)*nu/(2.*rad_avg)
                u_err_prev = u_err
                u_err = u_target - u_axial_mean

                u_err_integral += check_flux*dt*u_err
                u_err_derivative = (u_err-u_err_prev)/(check_flux*dt)

                F_arr[0] = Kp*u_err + Ki*u_err_integral + Kd*u_err_derivative
            comm.Bcast(F_arr, root=0)
            F = F_arr[0]

        if MPI.rank(mpi_comm_world()) == 0:
            with open(statsfolder + "/dump_flux.dat", "a") as fluxfile:
                fluxfile.write(
                   "{:d} {:e} {:e} {:e} {:e} {:e} {:e} {:e} {:e} {:e} {:e}\n".format(
                      tstep, t, u_axial, e_kin, area,
                       u_axial_vol, e_kin_vol, volume, Re, turb, F))


    return dict(F=F, u_err=u_err, u_err_integral=u_err_integral)


def theend(newfolder, tstep, stats, spark_puff, **NS_namespace):
    """Store statistics before exiting"""
    statsfolder = path.join(newfolder, "Stats")
    stats.toh5(1, tstep, filename=statsfolder+"/dump_vz_{}.h5".format(tstep))


def early_hook(mesh, mesh_file, folder, spark_puff, N, F,
               mesh_suffix,
               **NS_namespace):
    """ Do stuff before anything else. """
    if N is not None and isinstance(N, int):
        mesh_file = "mesh/rough_pipe_N{}{}.h5".format(
            N, "_" + mesh_suffix if bool(isinstance(mesh_suffix, str)
                                         and mesh_suffix is not "") else "")
        folder = "RoughPipe_N{}_F{}_results/".format(N, F)

    if path.isfile(mesh_file):
        info_red("Mesh: " + mesh_file)
    else:
        info_red("Couldn't find mesh: " + mesh_file)
        exit()

    # Create the mesh here.
    mesh = mesh(mesh_file, **NS_namespace)

    nodes_max, nodes_min, num_nodes, num_elems = get_mesh_properties(mesh_file)
    inlet_wall_coords = build_wall_coords(mesh_file)

    Lz = nodes_max[2]
    info_red("Some useful info")
    info("Number of nodes:    {}".format(num_nodes))
    info("Number of elements: {}".format(num_elems))
    info("Length of mesh:     {}".format(Lz))

    puff_center = 0.5*(nodes_max+nodes_min)
    puff_radius = 0.5*(nodes_max[0:2]-nodes_min[0:2]).min()

    constrained_domain = PeriodicDomain(Lz)

    return dict(mesh=mesh, mesh_file=mesh_file, folder=folder, Lz=Lz,
                inlet_wall_coords=inlet_wall_coords,
                puff_center=puff_center, puff_radius=puff_radius,
                constrained_domain=constrained_domain,
                u_err_integral=0., u_err=0.)
