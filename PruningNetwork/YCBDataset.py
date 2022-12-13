import os
import torch
import torch.utils.data as data
import numpy as np

class YCB(data.Dataset):
    def __init__(self, root):
        self.data_root = root
        self.files = os.listdir(self.data_root)

        print(f'[DATASET] {len(self.files)} data were loaded')

    def __getitem__(self, idx):
        sample = self.files[idx]
        data = np.load(os.path.join(self.data_root, sample), allow_pickle=True)[()]
        object_pc = data['obj_pc']
        hand_pc = data['hand_pc']
        gripper_pc = data['gripper_pc']
        label = np.array([data['grasp_score']])

        object_pc = torch.tensor(object_pc, dtype=torch.float)
        hand_pc = torch.tensor(hand_pc, dtype=torch.float)
        gripper_pc = torch.tensor(gripper_pc, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)
        # print(sample, hand_pc.shape)

        return object_pc, hand_pc, gripper_pc, label

    def __len__(self):
        return len(self.files)
