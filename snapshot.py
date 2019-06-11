from __future__ import print_function
import argparse
import dolfin as df
import os
import glob
import h5py
import numpy as np
from common.ductutils import numpy_to_dolfin, make_dof_coords, \
    make_xdict, set_val
from common import info_red
from fenicstools import Probes
import mpi4py
import pandas as pd
from xml.etree import cElementTree as ET

comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def get_args():
    parser = argparse.ArgumentParser(description="Take a snapshot.")
    parser.add_argument("folder", type=str, help="Folder.")
    parser.add_argument("-Nz", type=int, default=200, help="Nz")
    parser.add_argument("-Nr", type=int, default=32, help="Nr")
    parser.add_argument("-Ntheta", type=int, default=64, help="Ntheta")
    parser.add_argument("-s", "--steps", type=str, default="m1",
                        help="Timesteps to get")
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


def cyl_coords(Nr, Ntheta, Nz):
    i_z = np.arange(Nz).astype(float)
    i_r = np.arange(Nr).astype(float)
    i_theta = np.arange(Ntheta).astype(float)

    r = i_r/args.Nr*R
    z = Lz*(2.*i_z+1.)/(2.*args.Nz)
    theta = (2.*np.pi*i_theta)/args.Ntheta

    rv, thetav, zv = np.meshgrid(r, theta, z)

    Nrv, Nthetav, Nzv = np.meshgrid(range(Nr), range(Ntheta), range(Nz))
    elem_pos = zip(Nrv.flatten(), Nthetav.flatten(), Nzv.flatten())
    pos2id_dict = dict(zip(elem_pos, range(len(elem_pos))))

    def pos2id(i, j, k):
        key = (i % Nr, j % Ntheta, k % Nz)
        return pos2id_dict[key]

    elems = []
    for i in range(Nr-1):
        for j in range(Ntheta):
            for k in range(Nz-1):
                elems.append([pos2id(i, j, k), pos2id(i, j, k+1),
                              pos2id(i, j+1, k+1), pos2id(i, j+1, k),
                              pos2id(i+1, j, k), pos2id(i+1, j, k+1),
                              pos2id(i+1, j+1, k+1), pos2id(i+1, j+1, k)])
    elems = np.array(elems, dtype=int)

    pts_cyl = np.array(zip(rv.flatten(),
                           thetav.flatten(),
                           zv.flatten()))
    return pts_cyl, elems


def load_data(init_step, h5f_str):
    with h5py.File(h5f_str, "r") as h5fu:
        step = get_step(init_step, h5fu)
        data = np.array(h5fu.get("VisualisationVector/" + step))
    return data, step


def cyl2xyz(pts_cyl):
    xv = pts_cyl[:, 0]*np.cos(pts_cyl[:, 1])
    yv = pts_cyl[:, 0]*np.sin(pts_cyl[:, 1])

    pts_xyz = np.array(zip(xv, yv, pts_cyl[:, 2]))
    return pts_xyz


def get_cyl_dim(nodes, xy_mid):
    xy_rel = (nodes[:, :2]-np.outer(np.ones(len(nodes)),
                                    xy_mid))
    rad = np.linalg.norm(xy_rel, axis=1)
    R = rad.max()
    Lz = nodes[:, 2].max()-nodes[:, 2].min()
    return R, Lz


def get_tsteps(field):
    return sorted([int(os.path.basename(string)[13:-5])
                   for string in glob.glob(
                           os.path.join(
                               init_folder,
                               "Timeseries/{}_from_tstep_*.xdmf".format(
                                   field)))])


def parse_xdmf(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    dsets = []
    timestamps = []
    for i, step in enumerate(root[0][0]):
        if step.tag == "Time":
            # Support for earlier dolfin formats
            timestamps = [float(time) for time in
                          step[0].text.strip().split(" ")]
        elif step.tag == "Grid":
            timestamp = None
            dset_address = None
            for prop in step:
                if prop.tag == "Time":
                    timestamp = float(prop.attrib["Value"])
                elif prop.tag == "Attribute":
                    address = prop[0].text.split(":")
                    file_address = address[0]
                    dset_address = address[1]
            if timestamp is None:
                timestamp = timestamps[i-1]
            dsets.append((timestamp, os.path.join(os.path.dirname(xml_file),
                                                  file_address), dset_address))
    return dsets


def get_dsets(field, init_tsteps):
    dsets = []
    for init_tstep in init_tsteps:
        dsets.extend(parse_xdmf(os.path.join(
            init_folder,
            "Timeseries/{}_from_tstep_{}.xdmf".format(field, init_tstep))))
    return dsets


def control_dsets(dsets):
    for tstep, (time, file_address, dset_address) in enumerate(dsets):
        if not os.path.exists(file_address):
            info_red("Could not find file: {}".format(file_address))
            exit()
        with h5py.File(file_address, "r") as h5f:
            if dset_address not in h5f:
                info_red("Could not find dset: {}:{}".format(
                    file_address, dset_address))
                exit()


def probe(snapshot, probes, u, u_x, p, dsets_u, dsets_p, x, xdict):
    for tstep in range(len(dsets_u)):
        time_u, file_address_u, dset_address_u = dsets_u[tstep]
        time_p, file_address_p, dset_address_p = dsets_p[tstep]
        assert(time_u == time_p)
        time = time_u
        if rank == 0:
            print("Time = {}".format(time))
        with h5py.File(file_address_u, "r") as h5f:
            u_data = np.array(h5f[dset_address_u])
        with h5py.File(file_address_p, "r") as h5f:
            p_data = np.array(h5f[dset_address_p])
        for i in range(3):
            set_val(u_x[i], u_data[:, i], x, xdict)
            df.assign(u.sub(i), u_x[i])
        set_val(p, p_data, x, xdict)
        probes(u)
        u_probes = np.copy(probes.array(0))
        probes.clear()

        probes(p)
        p_probes = np.copy(probes.array(0))
        probes.clear()

        snapshot.write([u_probes, p_probes], fields=["u", "p"],
                       tstep=tstep, time=time)


class Snapshot:
    def __init__(self, filename, elems, nodes, nodes_cyl):
        self.filename = filename
        self.elems = elems
        self.nodes = nodes

        self.h5filename = filename + ".h5"
        self.xdmf_filename = filename + ".xdmf"
        self.local_h5filename = os.path.basename(self.h5filename)

        if rank == 0:
            self.h5f = h5py.File(self.h5filename, "w")

            self.h5f.create_dataset("Mesh/elements", data=elems)
            self.h5f.create_dataset("Mesh/coordinates", data=nodes)
            self.h5f.create_dataset("Mesh/cylinder_coordinates",
                                    data=nodes_cyl)

            self.xdmf_str = """<?xml version="1.0"?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">
  <Domain>
    <Grid Name="TimeSeries_Velocity function" GridType="Collection" CollectionType="Temporal">
      <Grid Name="mesh" GridType="Uniform">
        <Topology TopologyType="Hexahedron" NumberOfElements="{n_elem}" NodesPerElement="8">
          <DataItem Format="HDF" DataType="UInt" Dimensions="{n_elem} 8">
            {h5filename}:/Mesh/elements
          </DataItem>
        </Topology>
        <Geometry GeometryType="XYZ">
          <DataItem Format="HDF" Dimensions="{n_node} 3">
            {h5filename}:/Mesh/coordinates
          </DataItem>
        </Geometry>""".format(
                h5filename=self.local_h5filename,
                n_elem=len(self.elems),
                n_node=len(self.nodes))

    def vector_xml(self, tstep, field):
        vector_str = """
        <Attribute Name="{field}" AttributeType="Vector" Center="Node">
          <DataItem Format="HDF" Dimensions="{n_node} 3">
            {h5filename}:/Data/{tstep}/{field}
          </DataItem>
        </Attribute>""".format(h5filename=self.local_h5filename,
                               n_node=len(self.nodes),
                               tstep=tstep, field=field)
        return vector_str

    def scalar_xml(self, tstep, field):
        scalar_str = """
        <Attribute Name="{field}" AttributeType="Scalar" Center="Node">
          <DataItem Format="HDF" Dimensions="{n_node} 1">
            {h5filename}:/Data/{tstep}/{field}
          </DataItem>
        </Attribute>""".format(h5filename=self.local_h5filename,
                               n_node=len(self.nodes),
                               tstep=tstep, field=field)
        return scalar_str

    def write(self, funcs, fields=["u", "p"], tstep=0, time=0):
        if rank == 0:
            extra_str = """
        <Time Value="{time}" />""".format(time=time)
            for func, field in zip(funcs, fields):
                is_vector = (func.ndim > 1)
                if is_vector:
                    self.h5f.create_dataset(
                        "Data/{tstep}/{field}".format(tstep=tstep,
                                                      field=field),
                        data=np.copy(func))
                    extra_str += self.vector_xml(tstep, field)
                else:
                    self.h5f.create_dataset(
                        "Data/{tstep}/{field}".format(tstep=tstep,
                                                      field=field),
                        data=np.copy(func))
                    extra_str += self.scalar_xml(tstep, field)

            extra_str += """
      </Grid>
      <Grid>
        <xi:include xpointer="xpointer(//Grid[@Name=&quot;TimeSeries_Velocity function&quot;]/Grid[1]/*[self::Topology or self::Geometry])" />"""
            self.xdmf_str += extra_str

    def close(self):
        if rank == 0:
            self.xdmf_str = self.xdmf_str[:-156]
            self.xdmf_str += """
    </Grid>
  </Domain>
</Xdmf>"""

            self.h5f.close()
            xdmf_file = open(self.xdmf_filename, "w")
            xdmf_file.write(self.xdmf_str)
            xdmf_file.close()


if __name__ == "__main__":
    args = get_args()
    init_folder = args.folder
    steps = args.steps.replace("m", "-")

    init_tsteps_u = get_tsteps("u")
    init_tsteps_p = get_tsteps("p")
    init_tsteps = sorted(list(set(init_tsteps_u) & set(init_tsteps_p)))

    dsets_u = get_dsets("u", init_tsteps)
    dsets_p = get_dsets("p", init_tsteps)
    assert(len(dsets_u) == len(dsets_p))
    if len(dsets_u) == 0:
        info_red("No datasets found.")
        exit()

    ids = eval("range(len(dsets_u))[{}]".format(steps))
    if not isinstance(ids, list):
        ids = [ids]
    dsets_u = [dsets_u[i] for i in ids]
    dsets_p = [dsets_p[i] for i in ids]

    control_dsets(dsets_u)
    control_dsets(dsets_p)

    with h5py.File(dsets_u[0][1], "r") as h5fu:
        nodes = np.array(h5fu.get("Mesh/0/mesh/geometry"))
        elems = np.array(h5fu.get("Mesh/0/mesh/topology"))

    if rank == 0:
        print("Initializing!")

    mesh, S, V, x, xdict = initialize(nodes, elems)

    xy_mid = np.array([0., 0.])
    R, Lz = get_cyl_dim(nodes, xy_mid)
    if rank == 0:
        print("Dimensions:", R, Lz)

    pts_cyl, elems_cyl = cyl_coords(args.Nr, args.Ntheta, args.Nz)
    pts_xyz = cyl2xyz(pts_cyl)

    if rank == 0:
        print("Placing probes...")
    probes = Probes(pts_xyz.flatten(), V)

    if rank == 0:
        print("Initializing functions...")

    u_x = [df.Function(S) for _ in range(3)]
    u = df.Function(V)
    p = df.Function(S)
    indicator = df.Function(S)
    indicator.vector()[:] = 1.

    interpolated_folder = os.path.join(init_folder,
                                       "Interpolated")
    if rank == 0:
        print("Making folder: " + interpolated_folder)
        if not os.path.exists(interpolated_folder):
            os.makedirs(interpolated_folder)

        print("Creating snapshot.")

    snapshot_filename = os.path.join(interpolated_folder, "snapshot")
    snapshot = Snapshot(snapshot_filename, elems_cyl, pts_xyz, pts_cyl)

    if rank == 0:
        print("Probing indicator.")

    probe(snapshot, probes, u, u_x, p, dsets_u, dsets_p, x, xdict)

    probes(indicator)
    indicator_probes = np.copy(probes.array(0))
    probes.clear()
    if rank == 0:
        snapshot.h5f.create_dataset("Data/0/indicator", data=indicator_probes)

    snapshot.close()
