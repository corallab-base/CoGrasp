import tensorflow as tf
import numpy as np
# import sys
# print(sys.path)
# from PoinTr import shape_completion
import os
import sys
BASE_DIR = os.path.dirname(__file__)
print(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..'))
print(sys.path)
from pointnet2.utils import tf_util
from pointnet2.utils.pointnet_util import pointnet_sa_module

def merge_object_gripper_hand_pc(object_pc, gripper_pc, hand_pc):
    gripper_pc = tf.cast(gripper_pc, tf.float32)
    hand_pc = tf.cast(hand_pc, tf.float32)
    object_shape = object_pc.shape
    gripper_shape = gripper_pc.shape
    hand_shape = hand_pc.shape
    batch_size = object_shape[0]
    print(batch_size)

    merged_xyz = tf.concat((object_pc, gripper_pc, hand_pc), 1)
    labels = [tf.zeros((object_shape[1],1), dtype=tf.float32), tf.ones((gripper_shape[1],1), dtype=tf.float32), -tf.ones((hand_shape[1],1), dtype=tf.float32)]
    labels = tf.concat(labels, 0)
    labels = tf.expand_dims(labels, 0)
    labels = tf.tile(labels, [batch_size, 1, 1])

    merged_points = tf.concat([merged_xyz, labels], -1)

    return merged_xyz, merged_points

def pruning_network_model(object_pc, gripper_pc, hand_pc, is_training, bn_decay):
    merged_xyz, merged_points = merge_object_gripper_hand_pc(object_pc, gripper_pc, hand_pc)

    # feature abstraction layers
    f1_xyz, f1_points, _ = pointnet_sa_module(merged_xyz, merged_points, npoint=128, radius=0.02, nsample=64, mlp=[64, 64, 128],
                                                mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='feature1')
    f2_xyz, f2_points, _ = pointnet_sa_module(f1_xyz, f1_points, npoint=32, radius=0.04, nsample=128, mlp=[128, 128, 256],
                                                mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='feature2')
    f3_xyz, f3_points, _ = pointnet_sa_module(f2_xyz, f2_points, npoint=None, radius=None, nsample=None, mlp=[256, 256, 512],
                                                mlp2=None, gripu_all=True, is_training=is_training, bn_decay=bn_decay, scope='feature3')

    # fully connected layers
    batch_size = tf.shape(merged_xyz)[0]
    net = tf.reshape(f3_points, [batch_size, -1])
    net = tf_util.fully_connected(
                                net,
                                1024,
                                bn=True,
                                is_training=is_training,
                                scope='fc1',
                                bn_decay=bn_decay)
    net = tf_util.fully_connected(
                                net,
                                1024,
                                bn=True,
                                is_training=is_training,
                                scope='fc2',
                                bn_decay=bn_decay)
    predictions_logits = tf_util.fully_connected(
        net, 2, activation_fn=None, scope='fc_prediction')
    # score = tf_util.fully_connected(
    #     net, 1, activation_fn=None, scope='fc_score')
    predictions_logits = tf.nn.softmax(predictions_logits)

    return predictions_logits

data_filename = '/home/kaykay/isaacgym/python/contact_graspnet/pruning_network_data/train_temp/ce3bbd5651294097968bd8e0c9603776.npy'
data = np.load(data_filename, allow_pickle=True)[()]
object_pc = data['obj_pc']
hand_pc = data['hand_pc']
gripper_pc = data['gripper_pc']
object_pc = np.expand_dims(object_pc, axis=0)
hand_pc = np.expand_dims(hand_pc, axis=0)
gripper_pc = np.expand_dims(gripper_pc, axis=0)
xyz, points = merge_object_gripper_hand_pc(object_pc, gripper_pc, hand_pc)
print(tf.shape(xyz))
print(xyz)
print(tf.shape(points))
print(points)
