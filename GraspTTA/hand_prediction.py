from os import path
import sys
sys.path.append(path.abspath('./GraspTTA'))

from network.affordanceNet_obman_mano_vertex import affordanceNet
from network.cmapnet_objhand import pointnet_reg
from dataset.pointcloud_dataset_generation import PointCloud_Dataset

import trimesh
import torch
import mano
import numpy as np

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

def generate_grasps(model, cmap_model, rh_mano, obj_pc, device):
    '''
    Generate diverse grasps for object point clouds
    '''
    model.eval()
    cmap_model.eval()
    rh_mano.eval()

    obj_pc_sampled = obj_pc[np.random.choice(obj_pc.shape[0], 3000, replace=False),:]
    obj_pc_scale = get_diameter(obj_pc_sampled.numpy())
    obj_scale_tensor = torch.tensor(obj_pc_scale).type_as(obj_pc).repeat(3000, 1)
    obj_pc_TTT = torch.cat((obj_pc_sampled, obj_scale_tensor), dim=-1).permute(1, 0)  # [4, 3000]
    # transform to hand prediction network space
    translation = np.array([-0.0793, 0.0208, -0.6924]) + np.random.random(3) * 0.2
    translation = translation.reshape((1, 3))
    transformation = trimesh.transformations.quaternion_matrix((0.5, -0.5, -0.5, 0.5))
    transformation[:3,3] = translation
    obj_pc_transformed = obj_pc_sampled.numpy().copy().T + transformation[:3,3].reshape(-1,1)  # [3, 3000]
    obj_pc_transformed = torch.tensor(obj_pc_transformed, dtype=torch.float32)
    obj_pc_TTT[:3,:] = obj_pc_transformed

    obj_pc_TTT = torch.unsqueeze(obj_pc_TTT, 0) # [1, 4, 3000]
    obj_pc_TTT = obj_pc_TTT.detach().clone().to(device)
    recon_params = []
    recon_param = model.inference(obj_pc_TTT).detach()  # recon [1,61] mano params
    final_mano = rh_mano(betas=recon_param[:, :10], global_orient=recon_param[:, 10:13],
                    hand_pose=recon_param[:, 13:58], transl=recon_param[:, 58:])
    final_mano_verts = final_mano.vertices.squeeze(0).detach().cpu().numpy()  # [778, 3]

    obj_verts = obj_pc_transformed.T

    obj_color = np.tile([139,69,19], (obj_verts.shape[0],1))
    hand_color = np.tile([241, 194, 125], (778,1))
    final_joint_pc = trimesh.points.PointCloud(np.concatenate((obj_verts, final_mano_verts)), np.concatenate((obj_color, hand_color)))
    final_joint_pc.show()

    return obj_verts-translation, final_mano_verts-translation

def predict_hand_grasp(obj_pc, device):
    # affordance network information
    affordance_model_path = '/home/kaykay/isaacgym/python/contact_graspnet/GraspTTA/checkpoints/model_affordance_best_full.pth'
    obj_inchannel = 4
    encoder_layer_sizes = [1024, 512, 256]
    decoder_layer_sizes = [1024, 256, 61]
    latent_size = 64
    condition_size = 1024
    # cmap network information
    cmap_model_path = '/home/kaykay/isaacgym/python/contact_graspnet/GraspTTA/checkpoints/model_cmap_best.pth'

    # build network
    affordance_model = affordanceNet(obj_inchannel=obj_inchannel,
                                     cvae_encoder_sizes=encoder_layer_sizes,
                                     cvae_latent_size=latent_size,
                                     cvae_decoder_sizes=decoder_layer_sizes,
                                     cvae_condition_size=condition_size)  # GraspCVAE
    cmap_model = pointnet_reg(with_rgb=False)  # ContactNet
    # load pre-trained model
    checkpoint_affordance = torch.load(affordance_model_path, map_location=torch.device('cpu'))['network']
    affordance_model.load_state_dict(checkpoint_affordance)
    affordance_model = affordance_model.to(device)
    checkpoint_cmap = torch.load(cmap_model_path, map_location=torch.device('cpu'))['network']
    cmap_model.load_state_dict(checkpoint_cmap)
    cmap_model = cmap_model.to(device)

    # mano hand model
    with torch.no_grad():
        rh_mano = mano.load(model_path='/home/kaykay/isaacgym/python/contact_graspnet/GraspTTA/models/mano/MANO_RIGHT.pkl',
                            model_type='mano',
                            use_pca=True,
                            num_pca_comps=45,
                            batch_size=1,
                            flat_hand_mean=True).to(device)
    rh_faces = torch.from_numpy(rh_mano.faces.astype(np.int32)).view(1, -1, 3).to(device)  # [1, 1538, 3], face indexes

    return generate_grasps(affordance_model, cmap_model, rh_mano, obj_pc, device)
