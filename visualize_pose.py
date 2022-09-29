# import necessary module
from mpl_toolkits.mplot3d import axes3d
import os
import matplotlib.pyplot as plt
import numpy as np

# load data from file
# you replace this using with open
gt_path = os.path.join(os.path.dirname(__file__), "splits", "endovis", "curve", "gt_poses.npz")
gt_local_poses = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

our_path = os.path.join(os.path.dirname(__file__), "splits", "endovis", "curve", "pose_our.npz")
our_local_poses = np.load(our_path, fix_imports=True, encoding='latin1')["data"]


def dump(source_to_target_transformations):
    Ms = []
    cam_to_world = np.eye(4)
    Ms.append(cam_to_world)
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(source_to_target_transformation, cam_to_world)
        Ms.append(cam_to_world)
    return Ms


def compute_scale(gtruth, pred):

    # Optimize the scaling factor
    scale = np.sum(gtruth[:, :3, 3] * pred[:, :3, 3]) / np.sum(pred[:, :3, 3] ** 2)

    return scale

dump_gt = np.array(dump(gt_local_poses))
dump_our = np.array(dump(our_local_poses))

scale_our = dump_our * compute_scale(dump_gt, dump_our)

num = gt_local_poses.shape[0]
points_our = []
points_gt = []
origin = np.array([[0], [0], [0], [1]])

for i in range(0, num):
    point_our = np.dot(scale_our[i], origin)
    point_gt = np.dot(dump_gt[i], origin)

    points_our.append(point_our)
    points_gt.append(point_gt)

points_our = np.array(points_our)
points_gt = np.array(points_gt)

# new a figure and set it into 3d
fig = plt.figure()
ax = fig.gca(projection='3d')

# set figure information
# ax.set_title("3D_Curve")
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.set_zlabel("z [mm]")

# draw the figure, the color is r = read
figure1, = ax.plot(points_gt[:, 0, 0], points_gt[:, 1, 0], points_gt[:, 2, 0], c='b', linewidth=1.6)
figure2, = ax.plot(points_our[:, 0, 0], points_our[:, 1, 0], points_our[:, 2, 0], c='g', linewidth=1.6)

plt.savefig('vo.png',dpi=600)
plt.show()
