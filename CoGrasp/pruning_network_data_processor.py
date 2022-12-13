import numpy as np
import trimesh

filename = '/home/kaykay/isaacgym/python/contact_graspnet/pruning_network_data/train/foam_brick_1d76374678634b2b8f694db99fd886ef.npy'
data = np.load(filename, allow_pickle=True)
data = data[()]
obj_pc = data['obj_pc']
contct_pts = data['gripper_contact_pts']
hand = data['hand']

obj_color = np.tile([10, 10, 10], (obj_pc.shape[0], 1))
hand_color = np.tile([232, 190, 172], (hand.shape[0], 1))
ct_color = np.tile([111, 222, 77], (contct_pts.shape[0], 1))
tri = trimesh.points.PointCloud(np.concatenate((obj_pc, hand, contct_pts)), np.concatenate((obj_pc, hand_color, ct_color)))
tri.show()
