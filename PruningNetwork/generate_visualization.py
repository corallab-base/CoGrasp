import numpy as np
from contact_graspnet.contact_graspnet import mesh_utils
import mayavi.mlab as mlab
from model import PruningNetwork
import grasps_evaluation
import glob
import open3d as o3d
from contact_graspnet.PoinTr.utils import misc
import torch
import trimesh
import matplotlib.pyplot as plt

def get_gripper_pc(gripper_control_points, gripper_grasp):
    cam_pose = np.eye(4)
    pts = np.matmul(gripper_control_points, gripper_grasp[:3, :3].T)
    # pts -= object_translation
    pts += np.expand_dims(gripper_grasp[:3, 3], 0)
    pts_homog = np.concatenate((pts, np.ones((7, 1))),axis=1)
    pts = np.dot(pts_homog, cam_pose.T)[:,:3]
    return pts

def plot_data(predictions, obj_pc, hand_pc, gripper_control_points_closed):
    all_pts = []
    connections = []
    index = 0
    N = 7
    cam_pose = np.eye(4)

    for i,g in enumerate(predictions):
        pts = get_gripper_pc(gripper_control_points_closed, g)
        all_pts.append(pts)
        connections.append(np.vstack([np.arange(index,   index + N - 1.5),
                                      np.arange(index + 1, index + N - .5)]).T)
        index += N


    mesh = o3d.io.read_triangle_mesh('/home/kaykay/Downloads/030_fork/poisson/textured.obj')
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    # obj_pc = np.asarray(pcd.points)
    fig = mlab.figure('Pred Grasps')
    mlab.view(azimuth=180, elevation=180, distance=0.2)
    mlab.points3d(obj_pc[:, 0], obj_pc[:, 1], obj_pc[:, 2], color=(0.54296875, 0.26953125, 0.07421875))
    mlab.points3d(hand_pc[:, 0], hand_pc[:, 1], hand_pc[:, 2], color=(0.94140625, 0.7578125, 0.48828125))

    all_pts = np.vstack(all_pts)
    connections = np.vstack(connections)
    src = mlab.pipeline.scalar_scatter(all_pts[:,0], all_pts[:,1], all_pts[:,2])
    src.mlab_source.dataset.lines = connections
    src.update()
    lines =mlab.pipeline.tube(src, tube_radius=.0008, tube_sides=12)
    mlab.pipeline.surface(lines, color=(0,1.,0), opacity=1.0)

    # imgmap = mlab.screenshot(figure=fig, mode='rgba', antialiased=True)

    mlab.show()

    # plt.imshow(imgmap, zorder=4)
    # plt.show()

def plot_data_gripper(predictions, obj_pc, hand_pc, gripper_control_points_closed):
    all_pts = []
    connections = []
    index = 0
    N = 7
    cam_pose = np.eye(4)

    for i,g in enumerate(predictions):
        pts = get_gripper_pc(gripper_control_points_closed, g)
        all_pts.append(pts)
        connections.append(np.vstack([np.arange(index,   index + N - 1.5),
                                      np.arange(index + 1, index + N - .5)]).T)
        index += N


    mesh = o3d.io.read_triangle_mesh('/home/kaykay/Downloads/030_fork/poisson/textured.obj')
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    # obj_pc = np.asarray(pcd.points)
    fig = mlab.figure('Pred Grasps')
    mlab.view(azimuth=180, elevation=180, distance=0.2)
    mlab.points3d(obj_pc[:, 0], obj_pc[:, 1], obj_pc[:, 2], color=(0.54296875, 0.26953125, 0.07421875))
    # mlab.points3d(hand_pc[:, 0], hand_pc[:, 1], hand_pc[:, 2], color=(.878, .675, .411))

    all_pts = np.vstack(all_pts)
    connections = np.vstack(connections)
    src = mlab.pipeline.scalar_scatter(all_pts[:,0], all_pts[:,1], all_pts[:,2])
    src.mlab_source.dataset.lines = connections
    src.update()
    lines =mlab.pipeline.tube(src, tube_radius=.0008, tube_sides=12)
    mlab.pipeline.surface(lines, color=(0,1.,0), opacity=1.0)
    mlab.show()

def plot_data_hand(predictions, obj_pc, hand_pc, gripper_control_points_closed):
    all_pts = []
    connections = []
    index = 0
    N = 7
    cam_pose = np.eye(4)

    for i,g in enumerate(predictions):
        pts = get_gripper_pc(gripper_control_points_closed, g)
        all_pts.append(pts)
        connections.append(np.vstack([np.arange(index,   index + N - 1.5),
                                      np.arange(index + 1, index + N - .5)]).T)
        index += N


    mesh = o3d.io.read_triangle_mesh('/home/kaykay/Downloads/030_fork/poisson/textured.obj')
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    # obj_pc = np.asarray(pcd.points)
    fig = mlab.figure('Pred Grasps')
    mlab.view(azimuth=180, elevation=180, distance=0.2)
    mlab.points3d(obj_pc[:, 0], obj_pc[:, 1], obj_pc[:, 2], color=(0.54296875, 0.26953125, 0.07421875))
    mlab.points3d(hand_pc[:, 0], hand_pc[:, 1], hand_pc[:, 2], color=(0.94140625, 0.7578125, 0.48828125))
    mlab.show()

def draw_pc():
    mesh = o3d.io.read_triangle_mesh('/home/kaykay/isaacgym/assets/urdf/ycb/025_mug/google_16k/textured.obj')
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    obj_pc = np.asarray(pcd.points)
    obj_pc = obj_pc[np.random.choice(obj_pc.shape[0], 8192, replace=False),:]
    obj_pc = np.expand_dims(obj_pc, axis=0)
    obj_pc = torch.from_numpy(obj_pc).float()
    obj_pc = obj_pc.cuda()
    partial, _ = misc.seprate_point_cloud(obj_pc, 8192, [int(8192 * 1/4) , int(8192 * 1/4)], fixed_points = None)
    partial = partial.cpu()
    partial = partial.squeeze(0)
    obj_pc = obj_pc.cpu()
    obj_pc = obj_pc.squeeze(0)
    tri_input = trimesh.points.PointCloud(partial)
    tri_input.show()
    tri_pred = trimesh.points.PointCloud(obj_pc)
    tri_pred.show()

def get_collision_score(gripper_pc, hand):
    dist = cdist(hand, gripper_pc)
    return np.mean(dist)


def main():
    # draw_pc()
    file_list = glob.glob('/home/kaykay/isaacgym/python/contact_graspnet/pruning_network_data/train/mug_a53b89110b4848e7bb469de2a848cdd9.npy')
    data_file = '/home/kaykay/isaacgym/python/contact_graspnet/pruning_network_data/train/mug_c6f7295e051648ecbc72df5ec3757c04.npy'
    model = grasps_evaluation.load_model(175)
    model.eval()

    gripper_width=0.08
    gripper = mesh_utils.create_gripper('panda')
    gripper_control_points = gripper.get_control_point_tensor(1, False, convex_hull=False).squeeze()
    mid_point = 0.5*(gripper_control_points[1, :] + gripper_control_points[2, :])
    grasp_line_plot = np.array([np.zeros((3,)), mid_point, gripper_control_points[1], gripper_control_points[3],
                            gripper_control_points[1], gripper_control_points[2], gripper_control_points[4]])
    gripper_control_points_closed = grasp_line_plot.copy()
    gripper_control_points_closed[2:,0] = np.sign(grasp_line_plot[2:,0]) * gripper_width/2

    for file in file_list:
        print(file)
        data = np.load(file, allow_pickle=True)[()]
        obj_pc = np.array(data['obj_pc'])
        hand = np.array(data['hand'])
        gripper_pred = np.array(data['gripper_pred'])
        cograsp_pred = []

        best = 0
        best_grasp = 0
        best_pred = []
        worst = 10000
        worst_grasp = 0
        worst_pred = []
        for grasp in gripper_pred:
            # print(grasp.shape)
            res = grasps_evaluation.evaluate(model, obj_pc, hand, grasp, gripper_control_points_closed)
            # gripper_pc = get_gripper_pc(gripper_control_points_closed, gripper_grasp)
            # score = get_collision_score(gripper_pc, hand)
            if res>best:
                best = res
                best_grasp = grasp
            if worst>res:
                worst = res
                worst_grasp = grasp
            if (res>0.97):
                cograsp_pred.append(grasp)

        cograsp_pred = np.array(cograsp_pred)

        # mesh = o3d.io.read_triangle_mesh('/home/kaykay/isaacgym/assets/urdf/ycb/025_mug/google_16k/textured.obj')
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = mesh.vertices
        # obj_pc = np.asarray(pcd.points)
        # obj_pc = obj_pc[np.random.choice(obj_pc.shape[0], 8192, replace=False),:]

        best_pred.append(best_grasp)
        best_pred.append(worst_grasp)
        # worst_pred.append(worst_grasp)
        plot_data(best_pred, obj_pc, hand, gripper_control_points_closed)
        # plot_data_hand(cograsp_pred, obj_pc, hand, gripper_control_points_closed)
        # plot_data_gripper(gripper_pred, obj_pc, hand, gripper_control_points_closed)
        # plot_data(gripper_pred, obj_pc, hand, gripper_control_points_closed)
        # plot_data(cograsp_pred, obj_pc, hand, gripper_control_points_closed)

if __name__ == '__main__':
    main()
