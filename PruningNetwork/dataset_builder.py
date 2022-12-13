import torch
from YCBDataset import YCB

def dataset_builder(mode):
    if mode=='train':
        dataset = YCB('/home/kaykay/isaacgym/python/contact_graspnet/PruningNetwork/data/train')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    else:
        dataset = YCB('/home/kaykay/isaacgym/python/contact_graspnet/PruningNetwork/data/test')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    return dataloader
