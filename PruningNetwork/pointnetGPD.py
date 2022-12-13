import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist
import os
import glob

gripper_data = {"min_width": 0.0,
  "force_limit": 235.0,
  "max_width": 0.085,
  "finger_radius": 0.01,
  "max_depth": 0.03,
  "finger_width": 0.015,
  "real_finger_width": 0.0255,
  "hand_height": 0.022,
  "hand_height_two_finger_side": 0.105,
  "hand_outer_diameter": 0.08,
  "hand_depth": 0.125,
  "real_hand_depth": 0.120,
  "init_bite": 0.01
}

S_a = []
S_d = []
S_n = []

def get_hand_points(grasp_bottom_center, approach_normal, binormal):
    hh = gripper_data['hand_height']
    fw = gripper_data['finger_width']
    hod =gripper_data['hand_outer_diameter']
    hd = gripper_data['hand_depth']
    open_w = hod - fw * 2
    minor_pc = np.cross(approach_normal, binormal)
    minor_pc = minor_pc / np.linalg.norm(minor_pc)
    p5_p6 = minor_pc * hh * 0.5 + grasp_bottom_center
    p7_p8 = -minor_pc * hh * 0.5 + grasp_bottom_center
    p5 = -binormal * open_w * 0.5 + p5_p6
    p6 = binormal * open_w * 0.5 + p5_p6
    p7 = binormal * open_w * 0.5 + p7_p8
    p8 = -binormal * open_w * 0.5 + p7_p8
    p1 = approach_normal * hd + p5
    p2 = approach_normal * hd + p6
    p3 = approach_normal * hd + p7
    p4 = approach_normal * hd + p8

    p9 = -binormal * fw + p1
    p10 = -binormal * fw + p4
    p11 = -binormal * fw + p5
    p12 = -binormal * fw + p8
    p13 = binormal * fw + p2
    p14 = binormal * fw + p3
    p15 = binormal * fw + p6
    p16 = binormal * fw + p7

    p17 = -approach_normal * hh + p11
    p18 = -approach_normal * hh + p15
    p19 = -approach_normal * hh + p16
    p20 = -approach_normal * hh + p12
    p = np.vstack([np.array([0, 0, 0]), p1, p2, p3, p4, p5, p6, p7, p8, p9, p10,
                   p11, p12, p13, p14, p15, p16, p17, p18, p19, p20])
    return p

def get_normal_score(normal, hand):
    hand_pcd = o3d.geometry.PointCloud()
    hand_pcd.points = o3d.utility.Vector3dVector(np.concatenate((hand[15:25], hand[110:120])))
    hand_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    hand_normals = np.array(hand_pcd.normals)
    hand_avg_normal = np.mean(hand_normals, axis=0)
    gripper_normal = normal
    return np.abs(np.dot(gripper_normal,hand_avg_normal))

def get_collision_score(gripper_pc, hand):
    dist = cdist(hand, gripper_pc)
    return np.mean(dist)

def get_nearest_score(hand, gripper):
    dist = cdist(hand, gripper)
    return np.min(dist)

def main(grasp, hand):
    center_point = grasp[0:3]
    major_pc = grasp[3:6]  # binormal
    width = grasp[6]
    angle = grasp[7]
    level_score, refine_score = grasp[-2:]
    # cal approach
    cos_t = np.cos(angle)
    sin_t = np.sin(angle)
    R1 = np.c_[[cos_t, 0, sin_t],[0, 1, 0],[-sin_t, 0, cos_t]]
    axis_y = major_pc
    axis_x = np.array([axis_y[1], -axis_y[0], 0])
    if np.linalg.norm(axis_x) == 0:
        axis_x = np.array([1, 0, 0])
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
    approach_normal = R2.dot(R1)[:, 0]
    approach_normal = approach_normal / np.linalg.norm(approach_normal)
    minor_pc = np.cross(major_pc, approach_normal)

    grasp_bottom_center = -gripper_data['hand_depth'] * approach_normal + center_point
    hand_points = get_hand_points(grasp_bottom_center, approach_normal, major_pc)
    # print(hand_points)
    S_a.append(get_normal_score(approach_normal, hand))
    S_d.append(get_collision_score(hand_points, hand))
    S_n.append(get_nearest_score(hand, hand_points))

def compute_metric(object_files, root):
    for object_file in object_files:
        m = np.load(os.path.join(root, object_file), allow_pickle=True)
        m_good = m[m[:, -2] <= 0.4]
        m_good = m_good[np.random.choice(len(m_good), size=5, replace=True)]
        object = object_file.split("/")[-1][:-4]
        object = object[4:]
        print(object)
        data_list = glob.glob(f'/home/kaykay/isaacgym/python/contact_graspnet/pruning_network_data/train/{object}*.npy')
        # print(data_list)
        for data_file in data_list:
            data = np.load(data_file, allow_pickle=True)[()]
            obj_pc = np.array(data['obj_pc'])
            hand = np.array(data['hand'])
            for grasp in m_good:
                main(grasp, hand)

if __name__ == '__main__':
    root_train = '/home/kaykay/isaacgym/python/contact_graspnet/PointNetGPD_grasps_dataset/train/'
    root_test = '/home/kaykay/isaacgym/python/contact_graspnet/PointNetGPD_grasps_dataset/test/'
    object_files_train = os.listdir(root_train)
    object_files_test = os.listdir(root_test)
    compute_metric(object_files_train, root_train)
    compute_metric(object_files_test, root_test)

    S_a = np.array(S_a)
    S_n = np.array(S_n)
    S_d = np.array(S_d)
    print(f'Min S_a: {np.min(S_a)} Max S_A: {np.max(S_a)} Avg S_a: {np.mean(S_a)} Std S_a: {np.std(S_a)}')
    print(f'Min S_d: {np.min(S_d)} Max S_d: {np.max(S_d)} Avg S_d: {np.mean(S_d)} Std: S_d: {np.std(S_d)}')
    print(f'Min S_n: {np.min(S_n)} Max S_n: {np.max(S_n)} Avg S_n: {np.mean(S_n)} Std: S_n: {np.std(S_n)}')
