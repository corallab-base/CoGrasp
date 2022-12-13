import numpy as np
import os
from scipy.spatial.distance import cdist
from contact_graspnet.contact_graspnet import mesh_utils
from model import PruningNetwork
import grasps_evaluation
import open3d as o3d
import torch
import gc

def get_normal_score(gripper_grasp, hand):
    hand_pcd = o3d.geometry.PointCloud()
    hand_pcd.points = o3d.utility.Vector3dVector(np.concatenate((hand[15:25], hand[110:120])))
    hand_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    hand_normals = np.array(hand_pcd.normals)
    hand_avg_normal = np.mean(hand_normals, axis=0)
    gripper_normal = gripper_grasp[:3, 2]
    return np.abs(np.dot(gripper_normal,hand_avg_normal))

def get_collision_score(gripper_pc, hand):
    dist = cdist(hand, gripper_pc)
    return np.mean(dist)

def get_gripper_pc(gripper_control_points, gripper_grasp):
    cam_pose = np.eye(4)
    pts = np.matmul(gripper_control_points, gripper_grasp[:3, :3].T)
    # pts -= object_translation
    pts += np.expand_dims(gripper_grasp[:3, 3], 0)
    pts_homog = np.concatenate((pts, np.ones((7, 1))),axis=1)
    pts = np.dot(pts_homog, cam_pose.T)[:,:3]
    return pts

def get_nearest_score(hand, gripper):
    dist = cdist(hand, gripper)
    return np.min(dist)

def main2():
    root = '/home/kaykay/isaacgym/python/contact_graspnet/pruning_network_data/train/'
    model = grasps_evaluation.load_model(175)
    model.eval()
    S_a_pruning = []
    S_a_graspnet = []
    S_d_pruning = []
    S_d_graspnet = []
    S_n_pruning = []
    S_n_graspnet = []

    gripper_width=0.08
    gripper = mesh_utils.create_gripper('panda')
    gripper_control_points = gripper.get_control_point_tensor(1, False, convex_hull=False).squeeze()
    mid_point = 0.5*(gripper_control_points[1, :] + gripper_control_points[2, :])
    grasp_line_plot = np.array([np.zeros((3,)), mid_point, gripper_control_points[1], gripper_control_points[3],
                            gripper_control_points[1], gripper_control_points[2], gripper_control_points[4]])
    gripper_control_points_closed = grasp_line_plot.copy()
    gripper_control_points_closed[2:,0] = np.sign(grasp_line_plot[2:,0]) * gripper_width/2

    for file in os.listdir(root):
        print(file)
        data = np.load(os.path.join(root, file), allow_pickle=True)[()]
        obj_pc = np.array(data['obj_pc'])
        hand = np.array(data['hand'])
        gripper_pred = np.array(data['gripper_pred'])
        scores = np.array(data['scores'])
        # print(gripper_pred.shape)

        tot = min(gripper_pred.shape[0], 5)
        graspnet_ind = np.argpartition(scores, -tot)[-tot:]
        graspnet_pred = gripper_pred[graspnet_ind]
        cograsp_scores = []

        for grasp in gripper_pred:
            # print(grasp.shape)
            res = grasps_evaluation.evaluate(model, obj_pc, hand, grasp, gripper_control_points_closed)
            # gc.collect()
            # torch.cuda.empty_cache()
            cograsp_scores.append(res.item())

        cograsp_scores = np.array(cograsp_scores)
        cograsp_ind = np.argpartition(cograsp_scores, -tot)[-tot:]
        cograsp_pred = gripper_pred[cograsp_ind]
        # print(f'Cograsp Scores: {res[cograsp_ind]}, Graspnet Scores: {scores[graspnet_ind]}')

        for cograsp, graspnet in zip(cograsp_pred, graspnet_pred):
            S_a_pruning.append(get_normal_score(cograsp, hand))
            S_a_graspnet.append(get_normal_score(graspnet, hand))
            cograsp_gripper_pc = get_gripper_pc(gripper_control_points_closed, cograsp)
            graspnet_gripper_pc = get_gripper_pc(gripper_control_points_closed, graspnet)
            S_d_pruning.append(get_collision_score(cograsp_gripper_pc, hand))
            S_d_graspnet.append(get_collision_score(graspnet_gripper_pc, hand))
            S_n_pruning.append(get_nearest_score(hand, cograsp_gripper_pc))
            S_n_graspnet.append(get_nearest_score(hand, graspnet_gripper_pc))

    S_a_pruning = np.array(S_a_pruning)
    S_a_grapsnet = np.array(S_a_graspnet)
    S_d_pruning = np.array(S_d_pruning)
    S_d_graspnet = np.array(S_d_graspnet)
    S_n_pruning = np.array(S_n_pruning)
    S_n_graspnet = np.array(S_n_graspnet)

    print(f'Min S_a_graspnet: {np.min(S_a_graspnet)} Max S_a_graspnet: {np.max(S_a_grapsnet)} Avg S_a_graspnet: {np.mean(S_a_graspnet)} Std S_a_graspnet: {np.std(S_a_graspnet)}')
    print(f'Min S_a_pruning: {np.min(S_a_pruning)} Max S_a_pruning: {np.max(S_a_pruning)} Avg S_a_pruning: {np.mean(S_a_pruning)} Std S_a_pruning: {np.std(S_a_pruning)}')
    print(f'Min S_d_graspnet: {np.min(S_d_graspnet)} Max S_d_graspnet: {np.max(S_d_graspnet)} Avg S_d_graspnet: {np.mean(S_d_graspnet)} Std S_d_graspnet: {np.std(S_d_graspnet)}')
    print(f'Min S_d_pruning: {np.min(S_d_pruning)} Max S_d_pruning: {np.max(S_d_pruning)} Avg S_d_pruning: {np.mean(S_d_pruning)} Std S_d_pruning: {np.std(S_d_pruning)}')
    print(f'Min S_n_graspnet: {np.min(S_n_graspnet)} Max S_n_graspnet: {np.max(S_n_graspnet)} Avg S_n_graspnet: {np.mean(S_n_graspnet)} Std S_n_graspnet: {np.std(S_n_graspnet)}')
    print(f'Min S_n_pruning: {np.min(S_n_pruning)} Max S_n_pruning: {np.max(S_n_pruning)} Avg S_n_pruning: {np.mean(S_n_pruning)} Std S_n_pruning: {np.std(S_n_pruning)}')

def main():
    root = '/home/kaykay/isaacgym/python/contact_graspnet/PruningNetwork/data/train'

    S_a_pruning = []
    S_a_graspnet = []
    S_d_pruning = []
    S_d_graspnet = []
    S_n_pruning = []
    S_n_graspnet = []

    for file in os.listdir(root):
        data = np.load(os.path.join(root,file), allow_pickle=True)[()]
        obj_pc = np.array(data['obj_pc'])
        hand = np.array(data['hand_pc'])
        gripper = np.array(data['gripper_pc'])
        normal_score = np.array(data['normal_score'])
        distance_score = np.array(data['collision_score'])
        grasp_score = data['grasp_score']
        S_a_graspnet.append(normal_score)
        S_d_graspnet.append(distance_score)
        nearest_score = get_nearest_score(hand, gripper)
        S_n_graspnet.append(nearest_score)
        if grasp_score == 1:
            S_a_pruning.append(normal_score)
            S_d_pruning.append(distance_score)
            S_n_pruning.append(nearest_score)

    S_a_pruning = np.array(S_a_pruning)
    S_a_grapsnet = np.array(S_a_graspnet)
    S_d_pruning = np.array(S_d_pruning)
    S_d_graspnet = np.array(S_d_graspnet)
    S_n_pruning = np.array(S_n_pruning)
    S_n_graspnet = np.array(S_n_graspnet)

    print(f'Min S_a_graspnet: {np.min(S_a_graspnet)} Max S_a_graspnet: {np.max(S_a_grapsnet)} Avg S_a_graspnet: {np.mean(S_a_graspnet)} Std S_a_graspnet: {np.std(S_a_graspnet)}')
    print(f'Min S_a_pruning: {np.min(S_a_pruning)} Max S_a_pruning: {np.max(S_a_pruning)} Avg S_a_pruning: {np.mean(S_a_pruning)} Std S_a_pruning: {np.std(S_a_pruning)}')
    print(f'Min S_d_graspnet: {np.min(S_d_graspnet)} Max S_d_graspnet: {np.max(S_d_graspnet)} Avg S_d_graspnet: {np.mean(S_d_graspnet)} Std S_d_graspnet: {np.std(S_d_graspnet)}')
    print(f'Min S_d_pruning: {np.min(S_d_pruning)} Max S_d_pruning: {np.max(S_d_pruning)} Avg S_d_pruning: {np.mean(S_d_pruning)} Std S_d_pruning: {np.std(S_d_pruning)}')
    print(f'Min S_n_graspnet: {np.min(S_n_graspnet)} Max S_n_graspnet: {np.max(S_n_graspnet)} Avg S_n_graspnet: {np.mean(S_n_graspnet)} Std S_n_graspnet: {np.std(S_n_graspnet)}')
    print(f'Min S_n_pruning: {np.min(S_n_pruning)} Max S_n_pruning: {np.max(S_n_pruning)} Avg S_n_pruning: {np.mean(S_n_pruning)} Std S_n_pruning: {np.std(S_n_pruning)}')

if __name__ == '__main__':
    main2()
