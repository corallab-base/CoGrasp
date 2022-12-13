from os import path
import sys
sys.path.append(path.abspath('./PoinTr'))

import trimesh
import torch
import torch.nn as nn
import numpy as np
from tools import builder
from utils.config import *
from utils import parser
from tools import builder
from utils import misc
import open3d as o3d
from pointnet2_ops import pointnet2_utils

def get_diameter(vp):
    x = vp[:, 0].reshape((1, -1))
    y = vp[:, 1].reshape((1, -1))
    z = vp[:, 2].reshape((1, -1))
    x_max, x_min, y_max, y_min, z_max, z_min = np.max(x), np.min(x), np.max(y), np.min(y), np.max(z), np.min(z)
    diameter_x = abs(x_max - x_min)
    diameter_y = abs(y_max - y_min)
    diameter_z = abs(z_max - z_min)
    diameter = np.sqrt(diameter_x**2 + diameter_y**2 + diameter_z**2)
    return diameter

def pc_norm(pc):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    print(centroid)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    print(m)
    pc = pc / m
    return pc

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data

def complete_pc(input_pc):

    # load config for PoinTr
    config = cfg_from_yaml_file('/home/kaykay/isaacgym/python/contact_graspnet/PoinTr/cfgs/YCB_models/PoinTr.yaml')
    # load our shape completion model
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, '/home/kaykay/isaacgym/python/contact_graspnet/PoinTr/pretrained/ckpt-best-real.pth')
    base_model.cuda()
    # set model to eval mode
    base_model.eval()

    with torch.no_grad():
        # scaling for PoinTr
        input_pc *= 10

        input_pc = np.expand_dims(input_pc, axis=0)
        input_pc = torch.from_numpy(input_pc).float()
        input_pc = input_pc.contiguous()
        input_pc = input_pc.cuda()
        input_pc = fps(input_pc, 2048)
        input_pc = input_pc.contiguous()
        pred = base_model(input_pc)
        coarse_pc = pred[0]
        dense_pc = pred[1]

        input = input_pc.cpu()[0]
        prediction = dense_pc.cpu()[0]

        print(f'Shape Completion====> Sizes Input: {input.shape}, output: {prediction.shape}')
        # tri_input = trimesh.points.PointCloud(input)
        # tri_input.show()
        # tri_pred = trimesh.points.PointCloud(prediction)
        # tri_pred.show()

        return  prediction/10

def main(input_file):
    # args
    args = parser.get_args()
    # config
    config = get_config(args)

    # load our shape completion model
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts)
    base_model.cuda()
    # set model to eval mode
    base_model.eval()
    with torch.no_grad():
        data = np.load(input_file, allow_pickle=True).item()
        input_pc = data['pc'].astype(np.float32)
        camera_transform_array = data['camera_transform']
        object_translation = data['object_translation']
        object_rotation = data['object_rotation']
        pc_world = input_pc
        pc_world[:, [0,1,2]] = input_pc[:,[2,0,1]]
        pc_world[:, 0] *= -1
        pc_world[:, 2] *= -1
        pc_world = pc_world + camera_transform_array
        pc_center = pc_world - object_translation
        pc_center *= 10
        # pc_center -= [-0.02429158, -0.00386726,  0.03217306]
        # pc_center /= 0.07038764603977679

        # choice = np.random.permutation(pc_center.shape[0])
        # pc_center = pc_center[choice[:3000]]

        input_pc = np.expand_dims(pc_center, axis=0)
        input_pc = torch.from_numpy(input_pc).float()
        input_pc = input_pc.contiguous()
        input_pc = input_pc.cuda()
        input_pc = fps(input_pc, 2048)
        input_pc = input_pc.contiguous()
        pred = base_model(input_pc)
        coarse_pc = pred[0]
        dense_pc = pred[1]

        input = input_pc.cpu()[0]
        prediction = dense_pc.cpu()[0]

        print(f'Sizes Input: {input.shape}, output: {prediction.shape}')
        #
        # gt_directory = "/home/kaykay/isaacgym/assets/urdf/ycb/011_banana/poisson/textured.obj"
        # gt_mesh = o3d.io.read_triangle_mesh(gt_directory)
        # gt_pcd = o3d.geometry.TriangleMesh.sample_points_uniformly(gt_mesh, number_of_points=8192)
        # gt = np.asarray(gt_pcd.points)
        # # gt = pc_norm(gt)
        # gt = np.expand_dims(gt, axis=0)
        # gt = torch.from_numpy(gt).float()
        # gt = gt.contiguous()
        # gt = gt.cuda()
        # test_input, _ = misc.seprate_point_cloud(gt, 8192, [int(8192 * 1/4) , int(8192 * 3/4)], fixed_points = None)
        # test_pred = base_model(test_input)
        # test_pred = test_pred[1]
        # test_input = test_input.cpu()[0]
        # test_pred = test_pred.cpu()[0]

        # print(dense_pc)
        print(f'Shape of input: {input_pc.shape} and Shape of prediction: {dense_pc.shape}')
        tri_input = trimesh.points.PointCloud(input)
        tri_input.show()
        tri_pred = trimesh.points.PointCloud(prediction)
        tri_pred.show()
        #
        # print(f'Shape of input: {test_input.shape} and Shape of prediction: {test_pred.shape} Sand Shape of original: {gt.cpu()[0].shape}')
        # tri_input = trimesh.points.PointCloud(test_input)
        # tri_input.show()
        # tri_pred = trimesh.points.PointCloud(test_pred)
        # tri_pred.show()
        # tri_pred = trimesh.points.PointCloud(gt.cpu()[0])
        # tri_pred.show()

if __name__ == '__main__':
    input_file = '/home/kaykay/isaacgym/python/contact_graspnet/test_data/pc_data_banana0.npy'
    main(input_file)
