import os
import numpy as np
from sklearn.model_selection import train_test_split

source_dir = '/home/kaykay/isaacgym/python/contact_graspnet/pruning_network_data/train_temp'
source_dir_train = '/home/kaykay/isaacgym/python/contact_graspnet/PruningNetwork/data/train_old'
source_dir_test = '/home/kaykay/isaacgym/python/contact_graspnet/PruningNetwork/data/test_old'
destination_dir_train = '/home/kaykay/isaacgym/python/contact_graspnet/PruningNetwork/data/train'
destination_dir_test = '/home/kaykay/isaacgym/python/contact_graspnet/PruningNetwork/data/test'
# x = os.listdir(source_dir)
# print(len(x))
# x_train, x_test = train_test_split(x, test_size=0.2, random_state=7)
# print(len(x_train))
# print(len(x_test))

x_train = os.listdir(source_dir_train)
x_test = os.listdir(source_dir_test)

for file in x_train:
    data = np.load(os.path.join(source_dir_train, file), allow_pickle=True)[()]
    object_pc = data['obj_pc']
    hand_pc = data['hand_pc']
    gripper_pc = data['gripper_pc']
    label = np.array([data['grasp_score']])
    if (object_pc.shape[0] == 8192 and gripper_pc.shape[0] == 7 and hand_pc.shape[0] == 778):
        os.rename(os.path.join(source_dir_train, file), os.path.join(destination_dir_train, file))

for file in x_test:
    data = np.load(os.path.join(source_dir_test, file), allow_pickle=True)[()]
    object_pc = data['obj_pc']
    hand_pc = data['hand_pc']
    gripper_pc = data['gripper_pc']
    label = np.array([data['grasp_score']])
    if (object_pc.shape[0] == 8192 and gripper_pc.shape[0] == 7 and hand_pc.shape[0] == 778):
        os.rename(os.path.join(source_dir_test, file), os.path.join(destination_dir_test, file))
