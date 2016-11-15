from mpi4py.MPI import COMM_WORLD
import numpy as np
from itertools import chain
from fenicstools import interpolate_nonmatching_mesh
# from dolfin import info_red, info_blue, info_green, Function, norm, \
#     VectorFunctionSpace, curl, as_vector, project
from problems.NSfracStep import info_red, info_blue, info_green, \
    Function, norm, \
    VectorFunctionSpace, curl, as_vector, project


comm = COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


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
        faces = np.array(list(chain.from_iterable(data)))
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


def set_val(f, f_data, x, xdict):
    vec = f.vector()
    values = vec.get_local()
    values[:] = [f_data[xdict[tuple(x_val)]] for x_val in x.tolist()]
    vec.set_local(values)
    vec.apply('insert')


def generate_puff_spark(puff_center, puff_radius, puff_magnitude,
                        u_target,
                        x, x_data, xdict, S, V):
    dist_to_center = np.sqrt((x_data[:, 0]-puff_center[0])**2 +
                             (x_data[:, 1]-puff_center[1])**2 +
                             (x_data[:, 2]-puff_center[2])**2)
    length = len(dist_to_center)
    envelope = np.zeros(length)
    ids_within = dist_to_center <= puff_radius
    envelope[ids_within] = (
        np.exp(-(2*dist_to_center[ids_within]/puff_radius)**2) -
        np.exp(-2.**2))

    info_red("Making streamfunction")
    phi_x = Function(S)
    phi_y = Function(S)
    phi_z = Function(S)

    set_val(phi_x,
            envelope*np.random.normal(size=length),
            x, xdict)
    set_val(phi_y,
            envelope*np.random.normal(size=length),
            x, xdict)
    set_val(phi_z,
            envelope*np.random.normal(size=length),
            x, xdict)

    info_red("Creating Sv")
    Sv = VectorFunctionSpace(S.mesh(),
                             S.ufl_element().family(),
                             S.ufl_element().degree())
    info_blue("Projecting u_puff")
    u_puff = project(curl(as_vector((phi_x, phi_y, phi_z))),
                     Sv,
                     solver_type="gmres",
                     preconditioner_type="amg")

    u_puff_norm = norm(u_puff)
    info_blue("u_puff_norm = {}".format(u_puff_norm))
    u_puff.vector()[:] = (
        # u_target = Re_target*nu
        puff_magnitude*u_target*u_puff.vector()[:]/u_puff_norm)

    info_green("Interpolating u_puff_x")
    u_puff_x = interpolate_nonmatching_mesh(u_puff.sub(0), V)
    info_green("Interpolating u_puff_y")
    u_puff_y = interpolate_nonmatching_mesh(u_puff.sub(1), V)
    info_green("Interpolating u_puff_z")
    u_puff_z = interpolate_nonmatching_mesh(u_puff.sub(2), V)

    return u_puff_x, u_puff_y, u_puff_z
