# coding: utf-8
import h5py
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Mask away a puff")
parser.add_argument("folder", type=str, help="Folder")
parser.add_argument("x_left", type=float, help="x_left")
parser.add_argument("x_right", type=float, help="x_right")
parser.add_argument("snapshot_id", type=int, help="Snapshot ID")
parser.add_argument("--other_snapshot_id", default=None,
                    type=int, help="Snapshot ID 2")
parser.add_argument("--noshift", action="store_true", help="Don't shift")
parser.add_argument("--invert", action="store_true", help="Invert mask")
args = parser.parse_args()

x_left = args.x_left
x_right = args.x_right
snapshot_id = args.snapshot_id

u_file_in = "{}/Timeseries/u_from_tstep_-1.h5".format(args.folder)
p_file_in = "{}/Timeseries/p_from_tstep_-1.h5".format(args.folder)

assert(os.path.exists(u_file_in))
assert(os.path.exists(p_file_in))
assert(x_right > x_left)

h5fu_out = h5py.File("u_from_tstep_-1.h5", "w")
h5fp_out = h5py.File("p_from_tstep_-1.h5", "w")

h5fu_in = h5py.File(u_file_in, "r")
h5fp_in = h5py.File(p_file_in, "r")
geometry = h5fu_in["Mesh/0/mesh/geometry"]

geometry = np.array(h5fu_in["Mesh/0/mesh/geometry"])
topology = np.array(h5fu_in["Mesh/0/mesh/topology"])
u = np.array(h5fu_in["VisualisationVector/{}".format(snapshot_id)])
p = np.array(h5fp_in["VisualisationVector/{}".format(snapshot_id)])

if args.other_snapshot_id is not None:
    u2 = np.array(h5fu_in["VisualisationVector/{}".format(
        args.other_snapshot_id)])
    p2 = np.array(h5fu_in["VisualisationVector/{}".format(
        args.other_snapshot_id)])
else:
    u2 = u
    p2 = p

h5fu_in.close()
h5fp_in.close()

h5fu_out.create_dataset("Mesh/0/mesh/topology", data=topology)
h5fu_out.create_dataset("Mesh/0/mesh/geometry", data=geometry)

num_pts = len(geometry)
num_edge = len(geometry[geometry[:, 2] == 0.0, :])
num_unique = num_pts-num_edge

pts_low = geometry[num_edge:num_edge+num_unique/2, :]
pts_high = geometry[num_edge+num_unique/2:, :]
pts_high - pts_low

print np.max(pts_high - pts_low, 0)

u_shift = np.zeros_like(u2)
p_shift = np.zeros_like(p2)
if args.noshift:
    u_shift[:, :] = u2[:, :]
    p_shift[:, :] = p2[:, :]
else:
    u_shift[num_edge:num_edge+num_unique/2, :] = u2[num_edge+num_unique/2:, :]
    u_shift[num_edge+num_unique/2:, :] = u2[num_edge:num_edge+num_unique/2, :]
    u_shift[:num_edge, :] = u2[:num_edge, :]
    p_shift[num_edge:num_edge+num_unique/2, :] = p2[num_edge+num_unique/2:, :]
    p_shift[num_edge+num_unique/2:, :] = p2[num_edge:num_edge+num_unique/2, :]
    p_shift[:num_edge, :] = p2[:num_edge, :]

mask = 0.5*(np.tanh(geometry[:, 2]-x_left)-np.tanh(geometry[:, 2]-x_right))
if args.invert:
    mask = 1.-mask

u_out = np.zeros_like(u)
u_out[:, 0] = (1.-mask)*u[:, 0] + mask*u_shift[:, 0]
u_out[:, 1] = (1.-mask)*u[:, 1] + mask*u_shift[:, 1]
u_out[:, 2] = (1.-mask)*u[:, 2] + mask*u_shift[:, 2]

p_out = np.zeros_like(p)
p_out[:, 0] = (1.-mask)*p[:, 0] + mask*p_shift[:, 0]

h5fu_out.create_dataset("VisualisationVector/0", data=u_out)
h5fp_out.create_dataset("VisualisationVector/0", data=p_out)
h5fu_out.close()
h5fp_out.close()
