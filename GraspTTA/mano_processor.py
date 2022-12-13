import open3d as o3d
import torch
import mano
from mano.utils import Mesh
import numpy as np

model_path = '/home/kaykay/isaacgym/python/contact_graspnet/GraspTTA/models/mano/MANO_RIGHT.pkl'
n_comps = 45
batch_size = 10

rh_model = mano.load(model_path=model_path,
                     is_right= True,
                     num_pca_comps=n_comps,
                     batch_size=batch_size,
                     flat_hand_mean=False)

betas = torch.rand(batch_size, 10)*.1
pose = torch.rand(batch_size, n_comps)*.1
global_orient = torch.rand(batch_size, 3)
transl        = torch.rand(batch_size, 3)

output = rh_model(betas=betas,
                  global_orient=global_orient,
                  hand_pose=pose,
                  transl=transl,
                  return_verts=True,
                  return_tips = True)


h_meshes = rh_model.hand_meshes(output)
j_meshes = rh_model.joint_meshes(output)

#visualize hand mesh only
h_meshes[1].show()
print(f'Hand Meshes {h_meshes[1]}')

#visualize joints mesh only
j_meshes[1].show()
print(f'Joint Meshes {j_meshes[1]}')

#visualize hand and joint meshes
hj_meshes = Mesh.concatenate_meshes([h_meshes[1], j_meshes[1]])
hj_meshes.show()

pts, faces = h_meshes[1].sample(100, return_index=True)
normals = h_meshes[1].face_normals[faces]
# print(normals)
# print(normals.shape)

vertices = h_meshes[1].vertices
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(vertices[0:42])
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
radii = [0.005, 0.01, 0.02, 0.04]
rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii))
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.1, origin=[0, 0, 0])
o3d.visualization.draw_geometries([pcd, rec_mesh])

normals = np.array(pcd.normals)
print(normals.shape)
avg_normal = np.mean(normals, axis=0)
print(avg_normal)
