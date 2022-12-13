from isaacgym import gymapi
from isaacgym import gymtorch
from contact_graspnet_util import generate_grasps
from trac_ik_python.trac_ik import IK
from scipy.spatial.transform import Rotation as R
from stl_reader import stl_reader
from obj_reader import obj_reader
from PoinTr import shape_completion
from GraspTTA import hand_prediction
from PruningNetwork.model import PruningNetwork
from PruningNetwork import grasps_evaluation

import mayavi.mlab as mlab
import mesh_utils
import trimesh.transformations as trans
import fcl
import trimesh
import os
import sys
import math
import numpy as np
import glob
import torch
import open3d as o3d
import uuid

sys.path.append('/home/kaykay/isaacgym/python/contact_graspnet/ompl-1.5.2/py-bindings')
import ompl.base as ob
import ompl.geometric as og

"""
Applies (concatenates) 2 rotations
"""
def rotation_concat(quaternion1, quaternion0):
    x0, y0, z0, w0 = quaternion0[0], quaternion0[1], quaternion0[2], quaternion0[3]
    x1, y1, z1, w1 = quaternion1[0], quaternion1[1], quaternion1[2], quaternion1[3]
    return [x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                       -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                       x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                       -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0]

"""
Multiplies 2 quatenioins
"""
def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0.w, quaternion0.x, quaternion0.y, quaternion0.z
    w1, x1, y1, z1 = quaternion1.w, quaternion1.x, quaternion1.y, quaternion1.z
    return gymapi.Quat(x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                       -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                       x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                       -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0)

# acquire the gym interface
gym = gymapi.acquire_gym()

# create sim
sim_params = gymapi.SimParams()
sim_params.substeps = 3
sim_params.dt = 1.0 / 60.0
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.use_gpu_pipeline = False
sim_params.physx.solver_type = 1
# sim_params.physx.use_gpu = True
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1
# sim_params.physx.contact_offset = 0.001
# sim_params.physx.rest_offset = 0.0
# sim_params.physx.friction_offset_threshold = 0.001
# sim_params.physx.friction_correlation_distance = 0.0005

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# set root directory for assets
asset_root = "../../assets/"

# set torch device
device = 'cuda'

# create UR5e with 2F85 gripper asset and pose
ur5e_2f85_asset_file = "urdf/ur5e_mimic_real.urdf"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
asset_options.use_mesh_materials = True
ur5e_2f85_asset = gym.load_asset(sim, asset_root, ur5e_2f85_asset_file, asset_options)
ur5e_2f85_pose = gymapi.Transform()
ur5e_2f85_pose.p = gymapi.Vec3(-0.4125, 0, 0.92)
ur5e_2f85_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5*math.pi)

# default dof position targets
ur5e_num_dofs = gym.get_asset_dof_count(ur5e_2f85_asset)
default_dof_pos = np.zeros(ur5e_num_dofs, dtype=np.float32)
default_dof_pos[1] = -0.5*math.pi
default_dof_pos[2] = 0.5*math.pi
default_dof_pos[3] = -0.5*math.pi
default_dof_pos[4] = -0.5*math.pi
# default_dof_pos[6], default_dof_pos[7], default_dof_pos[8], default_dof_pos[9], default_dof_pos[10], default_dof_pos[11] = 0.1, 0.1, 0.1, -0.1, -0.1, -0.1
track_last_pose = default_dof_pos.tolist()
# default dof states
default_dof_state = np.zeros(ur5e_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"] = default_dof_pos

# create table asset and pose
table_dims = gymapi.Vec3(0.92, 0.92, 0.75)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.46, 0.0, 0.5 * table_dims.z)
rigid_shape_properties = gymapi.RigidShapeProperties()
rigid_shape_properties.friction = 10.0
gym.set_asset_rigid_shape_properties(table_asset, [rigid_shape_properties])

# create mug asset
mug_asset_file = "urdf/ycb/025_mug.urdf"
asset_options = gymapi.AssetOptions()
mug_asset = gym.load_asset(sim, asset_root, mug_asset_file, asset_options)
gym.set_asset_rigid_shape_properties(mug_asset, [rigid_shape_properties])

# create cracker box asset
cracker_box_asset_file = "urdf/ycb/003_cracker_box.urdf"
asset_options = gymapi.AssetOptions()
cracker_box_asset = gym.load_asset(sim, asset_root, cracker_box_asset_file, asset_options)
gym.set_asset_rigid_shape_properties(cracker_box_asset, [rigid_shape_properties])

# create bowl asset
bowl_asset_file = "urdf/ycb/024_bowl.urdf"
asset_options = gymapi.AssetOptions()
bowl_asset = gym.load_asset(sim, asset_root, bowl_asset_file, asset_options)
gym.set_asset_rigid_shape_properties(bowl_asset, [rigid_shape_properties])

# create banana asset
banana_asset_file = "urdf/ycb/011_banana.urdf"
asset_options = gymapi.AssetOptions()
banana_asset = gym.load_asset(sim, asset_root, banana_asset_file, asset_options)
gym.set_asset_rigid_shape_properties(banana_asset, [rigid_shape_properties])

# create hammer asset
hammer_asset_file = "urdf/ycb/048_hammer.urdf"
asset_options = gymapi.AssetOptions()
hammer_asset = gym.load_asset(sim, asset_root, hammer_asset_file, asset_options)
gym.set_asset_rigid_shape_properties(hammer_asset, [rigid_shape_properties])

# create orange asset
orange_asset_file = "urdf/ycb/017_orange.urdf"
asset_options = gymapi.AssetOptions()
orange_asset = gym.load_asset(sim, asset_root, orange_asset_file, asset_options)
gym.set_asset_rigid_shape_properties(orange_asset, [rigid_shape_properties])

# create apple asset
apple_asset_file = "urdf/ycb/013_apple.urdf"
asset_options = gymapi.AssetOptions()
apple_asset = gym.load_asset(sim, asset_root, apple_asset_file, asset_options)
gym.set_asset_rigid_shape_properties(apple_asset, [rigid_shape_properties])


#setup all collision meshes
ur5e_collision_parts = ["meshes/collision/base.stl",
                        "meshes/collision/shoulder.stl",
                        "meshes/collision/upperarm.stl",
                        "meshes/collision/forearm.stl",
                        "meshes/collision/wrist1.stl",
                        "meshes/collision/wrist2.stl",
                        "meshes/collision/wrist3.stl",
                        "meshes/2f85/robotiq_85_base_link_coarse.STL",
                        "meshes/2f85/inner_knuckle_coarse.STL",
                        "meshes/2f85/inner_finger_coarse.STL",
                        "meshes/2f85/outer_knuckle_coarse.STL",
                        "meshes/2f85/inner_knuckle_coarse.STL",
                        "meshes/2f85/inner_finger_coarse.STL",
                        "meshes/2f85/outer_knuckle_coarse.STL",
                        "meshes/2f85/outer_finger_coarse.STL",
                        "meshes/2f85/outer_finger_coarse.STL"]

ur5e_collision_models = []
ur5e_rotations = [R.from_euler('x',  [90], degrees = True),
                  R.from_euler('xy', [90, 180], degrees = True),
                  R.from_euler('xy', [180, 180], degrees = True),
                  R.from_euler('z',  [-180], degrees = True),
                  R.from_euler('x',  [-180], degrees = True),
                  R.from_euler('x',  [90], degrees = True),
                  R.from_euler('z',  [-90], degrees = True),
                  R.from_euler('xyz', [0, 0, 0], degrees = True),
                  R.from_euler('xyz', [0, 0, 0], degrees = True),
                  R.from_euler('xyz', [0, 0, 0], degrees = True),
                  R.from_euler('xyz', [0, 0, 0], degrees = True),
                  R.from_euler('xyz', [0, 0, 0], degrees = True),
                  R.from_euler('xyz', [0, 0, 0], degrees = True),
                  R.from_euler('xyz', [0, 0, 0], degrees = True),
                  R.from_euler('xyz', [0, 0, 0], degrees = True),
                  R.from_euler('xyz', [0, 0, 0], degrees = True)]
ur5e_translations = [[0, 0, 0],
                     [0, 0, 0],
                     [0, -0.138, 0],
                     [0, -0.007, 0],
                     [0, 0.127, 0],
                     [0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]]
for i in range(len(ur5e_collision_parts)):
    parts_path = ur5e_collision_parts[i]
    collision_mesh = stl_reader(asset_root + parts_path)
    m = fcl.BVHModel()
    collision_mesh.transform(ur5e_rotations[i], ur5e_translations[i])
    verts, tris = collision_mesh.get_vertices(), collision_mesh.get_faces()
    m.beginModel(len(verts), len(tris))
    m.addSubModel(verts, tris)
    m.endModel()
    ur5e_collision_models.append(m)

#set all environment collision models
plane_normal = np.array([0.0, 0.0, 1.0])
col_plane = fcl.Plane(plane_normal, 0)
plane_obj = fcl.CollisionObject(col_plane, fcl.Transform())

col_table = fcl.Box(table_dims.x, table_dims.y, table_dims.z)
trans_table = fcl.Transform(np.array([table_pose.p.x, table_pose.p.y, table_pose.p.z]))
table_obj = fcl.CollisionObject(col_table, trans_table)

object_collision_models = [table_obj]

# configure env grid
num_envs = 1
num_per_row = int(math.sqrt(num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

K = np.array([[727.82704671, 0, 512], [0, 727.82704671, 512], [0, 0, 1]])
envs = []
box_idxs = []
camera_handles = []
test_inputs = []
ur5e_handles = []
objects = ['mug', 'bowl', 'orange', 'banana', 'apple']
object_offset = [0.063, 0.057, 0.06, 0.08, 0.06]
object_gripper_dof = [0.31, 0.35, 0.3, 0.32, 0.3]
object_poses = {'banana': gymapi.Vec3(table_pose.p.x, table_pose.p.y + 0.05, table_dims.z),
'mug': gymapi.Vec3(table_pose.p.x + 0.1, table_pose.p.y + 0.3, table_dims.z),
'orange': gymapi.Vec3(table_pose.p.x, table_pose.p.y - 0.15, table_dims.z),
'bowl': gymapi.Vec3(table_pose.p.x - 0.2, table_pose.p.y - 0.25, table_dims.z),
'apple': gymapi.Vec3(table_pose.p.x - 0.1, table_pose.p.y + 0.25, table_dims.z)}
objects_collision_mesh_files = ['/home/kaykay/isaacgym/assets/urdf/ycb/025_mug/google_16k/textured_vhacd.obj',
'/home/kaykay/isaacgym/assets/urdf/ycb/024_bowl/google_16k/textured_vhacd.obj',
'/home/kaykay/isaacgym/assets/urdf/ycb/017_orange/google_16k/textured_vhacd.obj',
'/home/kaykay/isaacgym/assets/urdf/ycb/011_banana/google_16k/textured_vhacd.obj',
'/home/kaykay/isaacgym/assets/urdf/ycb/013_apple/google_16k/textured_vhacd.obj']

simulation_result_root = '/home/kaykay/isaacgym/python/contact_graspnet/SimulationResults/'
ompl_result_root = '/home/kaykay/isaacgym/python/contact_graspnet/SimulationResults/'
target_positions = []
target_rotations = []
ompl_path = []
for i in range(5):
    if os.path.isfile(os.path.join(simulation_result_root, objects[i]) + '_ompl.npy'):
        ompl_result = np.load(os.path.join(simulation_result_root, objects[i]) + '_ompl.npy', allow_pickle=True)[()]
        ompl_path.append(ompl_result['pathlist'])
    else:
        ompl_path.append(None)
    object_result = np.load(os.path.join(simulation_result_root, objects[i]) + '.npy', allow_pickle='True')[()]
    target_rotations.append(object_result['target_rot'])
    target_positions.append(object_result['target_pos'])

for (object, object_file) in zip(objects, objects_collision_mesh_files):
    print(object)
    collision_mesh = obj_reader(object_file)
    m = fcl.BVHModel()
    verts, tris = collision_mesh.get_vertices(), collision_mesh.get_faces()
    m.beginModel(len(verts), len(tris))
    m.addSubModel(verts, tris)
    m.endModel()
    tf = fcl.Transform(np.array([object_poses[object].x, object_poses[object].y, object_poses[object].z]))
    object_collision_models.append(fcl.CollisionObject(m, tf))

# create object segmentationid dict
object_segment_dict = {'table': 1, 'ur5e': 2, 'banana': 3, 'foam_brick': 4, 'mug': 5, 'chips_can': 6, 'cracker_box': 7, 'potted_meat_can': 8, 'master_chef_can': 9, 'sugar_box': 10, 'mustard_bottle': 11,
'tomato_soup_can': 12, 'tuna_fish_can': 13, 'pudding_box': 14, 'strawberry': 15, 'gelatin_box': 16, 'lemon': 17, 'apple': 18, 'peach': 19, 'orange': 20, 'pear': 21, 'plum': 22, 'pitcher_base': 23, 'bleach_cleanser': 24,
'bowl': 25, 'sponge': 26, 'fork': 27, 'spoon': 28, 'knife': 29, 'power_drill': 30, 'wood_block': 31, 'scissors': 32, 'padlock': 33, 'large_marker': 34, 'small_marker': 35, 'phillips_screwdriver': 36, 'flat_screwdriver': 37,
'hammer': 38, 'medium_clamp': 39, 'large_clamp': 40, 'extra_large_clamp': 41, 'mini_soccer_ball': 42, 'softball': 43, 'baseball': 44, 'tennis_ball': 45, 'racquetball': 46, 'golf_ball': 47, 'chain': 48, 'dice': 49, 'a_marbles': 50,
'd_marbles': 51, 'a_cups': 52, 'b_cups': 53, 'c_cups': 54, 'd_cups': 55, 'e_cups': 56, 'f_cups': 57, 'h_cups': 58, 'g_cups': 59, 'i_cups': 60, 'j_cups': 61, 'a_colored_wood_blocks': 62, 'nine_hole_peg_test': 63, 'a_toy_airplane': 64,
'b_toy_airplane': 65, 'c_toy_airplane': 66, 'd_toy_airplane': 67,'e_toy_airplane': 68, 'f_toy_airplane': 69, 'h_toy_airplane': 70, 'i_toy_airplane': 71, 'j_toy_airplane': 72, 'k_toy_airplane': 73, 'a_lego_duplo': 74,
'b_lego_duplo': 75, 'c_lego_duplo': 76, 'd_lego_duplo': 77, 'e_lego_duplo': 78, 'f_lego_duplo': 79, 'g_lego_duplo': 80, 'h_lego_duplo': 81, 'i_lego_duplo': 82, 'j_lego_duplo': 83, 'k_lego_duplo': 84, 'l_lego_duplo': 85, 'm_lego_duplo': 86,
'timer': 87, 'rubiks_cube': 88}

for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add UR5e
    ur5e_2f85_handle = gym.create_actor(env, ur5e_2f85_asset, ur5e_2f85_pose, "ur5e", i, 2, segmentationId=object_segment_dict['ur5e'])
    ur5e_handles.append(ur5e_2f85_handle)
    ur5e_base_idx = gym.find_actor_rigid_body_index(env, ur5e_2f85_handle, "base", gymapi.DOMAIN_SIM)

    #get joint handler
    spj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "shoulder_pan_joint")
    slj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "shoulder_lift_joint")
    ej = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "elbow_joint")
    wj1 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_1_joint")
    wj2 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_2_joint")
    wj3 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_3_joint")
    likj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "left_inner_knuckle_joint")
    lifj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "left_inner_finger_joint")
    lokj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "left_outer_knuckle_joint")
    rikj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "right_inner_knuckle_joint")
    rifj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "right_inner_finger_joint")
    rokj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "right_outer_knuckle_joint")

    # add table
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0, segmentationId=object_segment_dict['table'])
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    # add mug for each environment
    mug_pose = gymapi.Transform()
    mug_pose.p = object_poses['mug']
    mug_handle = gym.create_actor(env, mug_asset, mug_pose, "mug", i, 0, segmentationId=object_segment_dict['mug'])
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, mug_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    # add bowl for each environment
    bowl_pose = gymapi.Transform()
    bowl_pose.p = object_poses['bowl']
    bowl_handle = gym.create_actor(env, bowl_asset, bowl_pose, "bowl", i, 0, segmentationId=object_segment_dict['bowl'])
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, bowl_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    # add banana for each environment
    banana_pose = gymapi.Transform()
    banana_pose.p = object_poses['banana']
    banana_handle = gym.create_actor(env, banana_asset, banana_pose, "banana",i, 0, segmentationId=object_segment_dict['banana'])
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, banana_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    # add orange for each environment
    orange_pose = gymapi.Transform()
    orange_pose.p = object_poses['orange']
    orange_handle = gym.create_actor(env, orange_asset, orange_pose, "orange",i, 0, segmentationId=object_segment_dict['orange'])
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, orange_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    # add apple for each environment
    apple_pose = gymapi.Transform()
    apple_pose.p = object_poses['apple']
    apple_handle = gym.create_actor(env, apple_asset, apple_pose, "apple",i, 0, segmentationId=object_segment_dict['apple'])
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, apple_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    # set initial dof states of UR5e
    gym.set_actor_dof_states(env, ur5e_2f85_handle, default_dof_state, gymapi.STATE_ALL)

    # set initial position targets of UR5e
    gym.set_actor_dof_position_targets(env, ur5e_2f85_handle, default_dof_pos)


for i in range(num_envs):
    # add camera sensor
    camera_properties = gymapi.CameraProperties()
    camera_properties.width = 1280
    camera_properties.height = 720
    camera_properties.horizontal_fov = 70.25
    camera_handle = gym.create_camera_sensor(envs[i], camera_properties)
    camera_transform = gymapi.Transform()
    # camera_transform.p = gymapi.Vec3(1.0, 0.00001, 0.5)
    # camera_transform.r = gymapi.Quat(0.5, 0.5, -0.5, -0.5)
    camera_position = gymapi.Vec3(1.35, 0.0, 1.04)
    camera_target = gymapi.Vec3(0.0, 0.0, 1.04)
    gym.set_camera_location(camera_handle, envs[i], camera_position, camera_target)
    camera_handles.append(camera_handle)

projection_matrix = np.matrix(gym.get_camera_proj_matrix(sim, envs[-1], camera_handles[-1]))
view_matrix = np.matrix(gym.get_camera_view_matrix(sim, envs[-1], camera_handles[-1]))
print(projection_matrix)

# point camera at middle env
cam_pos = gymapi.Vec3(4, 3, 2)
cam_target = gymapi.Vec3(-4, -3, 0)
middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

"""
Get robot pose from dof
"""
def get_pose_from_dof(dofs):
    #link1 pose
    trans1, rot1 = [0, 0, 0], [-0, -math.sqrt(2)/2, math.sqrt(2)/2, -0]

    #link2 pose
    rot2_initial = [-0, -math.sqrt(2)/2, math.sqrt(2)/2, -0]
    rot2_new = R.from_euler('z', dofs[0]).as_quat().tolist()
    rot2_final = rotation_concat(rot2_new, rot2_initial)
    trans2, rot2 = [0, 0, 0.1625], rot2_final

    #link3 pose
    #rot3_initial = [0, -0, 1, 0]
    rot3_initial = [math.sqrt(2)/2, -0, math.sqrt(2)/2, -0]
    rot3_vector = R.from_quat(rot2_new).apply([0, 1, 0])
    rot3_final = rotation_concat(rot2_new, rot3_initial)
    rot3_new = R.from_rotvec(dofs[1]*rot3_vector).as_quat().tolist()
    rot3_final = rotation_concat(rot3_new, rot3_final)
    trans3, rot3 = [0, 0, 0.1625], rot3_final

    #link4 pose
    #rot4_initial = [0, -0, 1, 0]
    rot4_initial = [math.sqrt(2)/2, -0, math.sqrt(2)/2, -0]
    rot4_vector = rot3_vector
    rot4_final = rot3_final
    rot4_offset = R.from_quat(rot3_final).apply([0, 0, 0.425])
    rot4_new = R.from_rotvec(dofs[2]*rot4_vector).as_quat().tolist()
    rot4_final = rotation_concat(rot4_new, rot4_final)
    trans4, rot4 = trans3 + rot4_offset, rot4_final

    #link5 pose
    rot5_offset = R.from_quat(rot4_final).apply([0, -0.1333, 0.3915])
    rot5_initial = [0, -0, 1, 0]
    rot5_vector = rot4_vector
    rot5_final = rotation_concat(rot2_new, rot5_initial)
    rot5_final = rotation_concat(rot3_new, rot5_final)
    rot5_final = rotation_concat(rot4_new, rot5_final)
    rot5_new = R.from_rotvec(dofs[3]*rot5_vector).as_quat().tolist()
    rot5_final = rotation_concat(rot5_new, rot5_final)
    trans5, rot5 = trans4 + rot5_offset, rot5_final

    #link6 pose
    rot6_offset = R.from_quat(rot5_final).apply([0, 0, 0])
    rot6_initial = [0, math.sqrt(2)/2, math.sqrt(2)/2, -0]
    rot6_final = rotation_concat(rot2_new, rot6_initial)
    rot6_final = rotation_concat(rot3_new, rot6_final)
    rot6_final = rotation_concat(rot4_new, rot6_final)
    rot6_final = rotation_concat(rot5_new, rot6_final)
    rot6_vector = [0, 0, -1]
    rot6_vector = R.from_quat(rot2_new).apply(rot6_vector)
    rot6_vector = R.from_quat(rot3_new).apply(rot6_vector)
    rot6_vector = R.from_quat(rot4_new).apply(rot6_vector)
    rot6_vector = R.from_quat(rot5_new).apply(rot6_vector)
    rot6_new = R.from_rotvec(dofs[4]*rot6_vector).as_quat().tolist()
    rot6_final = rotation_concat(rot6_new, rot6_final)
    trans6, rot6 = trans5 + rot6_offset, rot6_final

    #link7 pose
    rot7_offset = R.from_quat(rot6_final).apply([0, -0.0996, 0])
    rot7_initial = [math.sqrt(2)/2, math.sqrt(2)/2, 0, 0]
    rot7_final = rotation_concat(rot2_new, rot7_initial)
    rot7_final = rotation_concat(rot3_new, rot7_final)
    rot7_final = rotation_concat(rot4_new, rot7_final)
    rot7_final = rotation_concat(rot5_new, rot7_final)
    rot7_final = rotation_concat(rot6_new, rot7_final)
    rot7_vector = [0, 1, 0]
    rot7_vector = R.from_quat(rot2_new).apply(rot7_vector)
    rot7_vector = R.from_quat(rot3_new).apply(rot7_vector)
    rot7_vector = R.from_quat(rot4_new).apply(rot7_vector)
    rot7_vector = R.from_quat(rot5_new).apply(rot7_vector)
    rot7_vector = R.from_quat(rot6_new).apply(rot7_vector)
    rot7_new = R.from_rotvec(dofs[5]*rot7_vector).as_quat().tolist()
    rot7_final = rotation_concat(rot7_new, rot7_final)
    trans7, rot7 = trans6 + rot7_offset, rot7_final

    # #camera pose
    # rot8_offset = R.from_quat(rot7_final).apply([0.065, 0, 0.04])
    # rot8_final = rot7
    # trans8, rot8 = trans7 + rot8_offset, rot8_final

    # robotiq_85_base_link_coarse pose
    rot8_offset = R.from_quat(rot7_final).apply([0.094, 0, 0.0])
    rot8_initial = [0, math.sqrt(2)/2, math.sqrt(2)/2, -0]
    rot8_final = rotation_concat(rot2_new, rot8_initial)
    rot8_final = rotation_concat(rot3_new, rot8_final)
    rot8_final = rotation_concat(rot4_new, rot8_final)
    rot8_final = rotation_concat(rot5_new, rot8_final)
    rot8_final = rotation_concat(rot6_new, rot8_final)
    rot8_final = rotation_concat(rot7_new, rot8_final)
    rot8_new = rot7_new
    rot8_final = rotation_concat(rot8_new, rot8_final)
    # rot8_vector =
    trans8, rot8 = trans7 + rot8_offset, rot8_final

    # left inner knuckle pose
    rot9_offset = R.from_quat(rot8_final).apply([0.0127000000001501, 0, 0.0693074999999639])
    rot9_initial = [0, math.sqrt(2)/2, math.sqrt(2)/2, -0]
    rot9_final = rotation_concat(rot2_new, rot9_initial)
    rot9_final = rotation_concat(rot3_new, rot9_final)
    rot9_final = rotation_concat(rot4_new, rot9_final)
    rot9_final = rotation_concat(rot5_new, rot9_final)
    rot9_final = rotation_concat(rot6_new, rot9_final)
    rot9_final = rotation_concat(rot7_new, rot9_final)
    rot9_final = rotation_concat(rot8_new, rot9_final)
    rot9_vector = [0, 0, -1]
    rot9_vector = R.from_quat(rot2_new).apply(rot9_vector)
    rot9_vector = R.from_quat(rot3_new).apply(rot9_vector)
    rot9_vector = R.from_quat(rot4_new).apply(rot9_vector)
    rot9_vector = R.from_quat(rot5_new).apply(rot9_vector)
    rot9_vector = R.from_quat(rot6_new).apply(rot9_vector)
    rot9_vector = R.from_quat(rot7_new).apply(rot9_vector)
    rot9_vector = R.from_quat(rot8_new).apply(rot9_vector)
    rot9_new = R.from_rotvec(dofs[6]*rot9_vector).as_quat().tolist()
    rot9_final = rotation_concat(rot9_new, rot9_final)
    trans9, rot9 = trans8 + rot9_offset, rot9_final

    # left inner finger pose
    rot10_offset = R.from_quat(rot9_final).apply([0.034585310861294, 0, 0.0454970193817975])
    rot10_initial = [0, math.sqrt(2)/2, math.sqrt(2)/2, -0]
    rot10_final = rotation_concat(rot2_new, rot10_initial)
    rot10_final = rotation_concat(rot3_new, rot10_final)
    rot10_final = rotation_concat(rot4_new, rot10_final)
    rot10_final = rotation_concat(rot5_new, rot10_final)
    rot10_final = rotation_concat(rot6_new, rot10_final)
    rot10_final = rotation_concat(rot7_new, rot10_final)
    rot10_final = rotation_concat(rot8_new, rot10_final)
    rot10_final = rotation_concat(rot9_new, rot10_final)
    rot10_vector = [0, 0, -1]
    rot10_vector = R.from_quat(rot2_new).apply(rot10_vector)
    rot10_vector = R.from_quat(rot3_new).apply(rot10_vector)
    rot10_vector = R.from_quat(rot4_new).apply(rot10_vector)
    rot10_vector = R.from_quat(rot5_new).apply(rot10_vector)
    rot10_vector = R.from_quat(rot6_new).apply(rot10_vector)
    rot10_vector = R.from_quat(rot7_new).apply(rot10_vector)
    rot10_vector = R.from_quat(rot8_new).apply(rot10_vector)
    rot10_vector = R.from_quat(rot9_new).apply(rot10_vector)
    rot10_new = R.from_rotvec(dofs[7]*rot10_vector).as_quat().tolist()
    rot10_final = rotation_concat(rot10_new, rot10_final)
    trans10, rot10 = trans9 + rot10_offset, rot10_final

    # left outer knuckle pose
    rot11_offset = R.from_quat(rot8_final).apply([0.0306011444260539, 0, 0.0627920162695395])
    rot11_initial = [0, math.sqrt(2)/2, math.sqrt(2)/2, -0]
    rot11_final = rotation_concat(rot2_new, rot11_initial)
    rot11_final = rotation_concat(rot3_new, rot11_final)
    rot11_final = rotation_concat(rot4_new, rot11_final)
    rot11_final = rotation_concat(rot5_new, rot11_final)
    rot11_final = rotation_concat(rot6_new, rot11_final)
    rot11_final = rotation_concat(rot7_new, rot11_final)
    rot11_final = rotation_concat(rot8_new, rot11_final)
    rot11_vector = [0, 0, -1]
    rot11_vector = R.from_quat(rot2_new).apply(rot11_vector)
    rot11_vector = R.from_quat(rot3_new).apply(rot11_vector)
    rot11_vector = R.from_quat(rot4_new).apply(rot11_vector)
    rot11_vector = R.from_quat(rot5_new).apply(rot11_vector)
    rot11_vector = R.from_quat(rot6_new).apply(rot11_vector)
    rot11_vector = R.from_quat(rot7_new).apply(rot11_vector)
    rot11_vector = R.from_quat(rot8_new).apply(rot11_vector)
    rot11_new = R.from_rotvec(dofs[8]*rot11_vector).as_quat().tolist()
    rot11_final = rotation_concat(rot11_new, rot11_final)
    trans11, rot11 = trans8 + rot11_offset, rot11_final

    # right inner knuckle pose
    rot12_offset = R.from_quat(rot8_final).apply([-0.0126999999998499, 0, 0.0693075000000361])
    rot12_initial = [math.sqrt(2)/2, -0, -0, -math.sqrt(2)/2]
    rot12_final = rotation_concat(rot2_new, rot12_initial)
    rot12_final = rotation_concat(rot3_new, rot12_final)
    rot12_final = rotation_concat(rot4_new, rot12_final)
    rot12_final = rotation_concat(rot5_new, rot12_final)
    rot12_final = rotation_concat(rot6_new, rot12_final)
    rot12_final = rotation_concat(rot7_new, rot12_final)
    rot12_final = rotation_concat(rot8_new, rot12_final)
    rot12_vector = [0, 0, -1]
    rot12_vector = R.from_quat(rot2_new).apply(rot12_vector)
    rot12_vector = R.from_quat(rot3_new).apply(rot12_vector)
    rot12_vector = R.from_quat(rot4_new).apply(rot12_vector)
    rot12_vector = R.from_quat(rot5_new).apply(rot12_vector)
    rot12_vector = R.from_quat(rot6_new).apply(rot12_vector)
    rot12_vector = R.from_quat(rot7_new).apply(rot12_vector)
    rot12_vector = R.from_quat(rot8_new).apply(rot12_vector)
    rot12_new = R.from_rotvec(dofs[9]*rot12_vector).as_quat().tolist()
    rot12_final = rotation_concat(rot12_new, rot12_final)
    trans12, rot12 = trans8 + rot12_offset, rot12_final

    # right inner finger pose
    rot13_offset = R.from_quat(rot12_final).apply([0.0341060475457406, 0, 0.0458573878541688])
    rot13_initial = [math.sqrt(2)/2, -0, -0, -math.sqrt(2)/2]
    rot13_final = rotation_concat(rot2_new, rot13_initial)
    rot13_final = rotation_concat(rot3_new, rot13_final)
    rot13_final = rotation_concat(rot4_new, rot13_final)
    rot13_final = rotation_concat(rot5_new, rot13_final)
    rot13_final = rotation_concat(rot6_new, rot13_final)
    rot13_final = rotation_concat(rot7_new, rot13_final)
    rot13_final = rotation_concat(rot8_new, rot13_final)
    rot13_final = rotation_concat(rot12_new, rot13_final)
    rot13_vector = [0, 0, -1]
    rot13_vector = R.from_quat(rot2_new).apply(rot13_vector)
    rot13_vector = R.from_quat(rot3_new).apply(rot13_vector)
    rot13_vector = R.from_quat(rot4_new).apply(rot13_vector)
    rot13_vector = R.from_quat(rot5_new).apply(rot13_vector)
    rot13_vector = R.from_quat(rot6_new).apply(rot13_vector)
    rot13_vector = R.from_quat(rot7_new).apply(rot13_vector)
    rot13_vector = R.from_quat(rot8_new).apply(rot13_vector)
    rot13_vector = R.from_quat(rot12_new).apply(rot13_vector)
    rot13_new = R.from_rotvec(dofs[10]*rot13_vector).as_quat().tolist()
    rot13_final = rotation_concat(rot13_new, rot13_final)
    trans13, rot13 = trans12 + rot13_offset, rot13_final

    # right outer knuckle pose
    rot14_offset = R.from_quat(rot8_final).apply([-0.0306011444258893, 0, 0.0627920162695395])
    rot14_initial = [math.sqrt(2)/2, -0, -0, -math.sqrt(2)/2]
    rot14_final = rotation_concat(rot2_new, rot14_initial)
    rot14_final = rotation_concat(rot3_new, rot14_final)
    rot14_final = rotation_concat(rot4_new, rot14_final)
    rot14_final = rotation_concat(rot5_new, rot14_final)
    rot14_final = rotation_concat(rot6_new, rot14_final)
    rot14_final = rotation_concat(rot7_new, rot14_final)
    rot14_final = rotation_concat(rot8_new, rot14_final)
    rot14_vector = [0, 0, -1]
    rot14_vector = R.from_quat(rot2_new).apply(rot14_vector)
    rot14_vector = R.from_quat(rot3_new).apply(rot14_vector)
    rot14_vector = R.from_quat(rot4_new).apply(rot14_vector)
    rot14_vector = R.from_quat(rot5_new).apply(rot14_vector)
    rot14_vector = R.from_quat(rot6_new).apply(rot14_vector)
    rot14_vector = R.from_quat(rot7_new).apply(rot14_vector)
    rot14_vector = R.from_quat(rot8_new).apply(rot14_vector)
    rot14_new = R.from_rotvec(dofs[11]*rot14_vector).as_quat().tolist()
    # print(f'Check quat: {dofs[11]}, {rot14_vector}, {rot14_new}, {rot14_final}')
    rot14_final = rotation_concat(rot14_new, rot14_final)
    trans14, rot14 = trans8 + rot14_offset, rot14_final

    # left outer finger pose
    rot15_offset = R.from_quat(rot11_final).apply([0.0316910442266543, 0, -0.00193396375724605])
    rot15_initial = [0, math.sqrt(2)/2, math.sqrt(2)/2, -0]
    rot15_final = rotation_concat(rot2_new, rot15_initial)
    rot15_final = rotation_concat(rot3_new, rot15_final)
    rot15_final = rotation_concat(rot4_new, rot15_final)
    rot15_final = rotation_concat(rot5_new, rot15_final)
    rot15_final = rotation_concat(rot6_new, rot15_final)
    rot15_final = rotation_concat(rot7_new, rot15_final)
    rot15_final = rotation_concat(rot8_new, rot15_final)
    rot15_final = rotation_concat(rot11_new, rot15_final)
    rot15_new = rot11_new
    rot15_final = rotation_concat(rot15_new, rot15_final)
    trans15, rot15 = trans11 + rot15_offset, rot15_final

    # right outer finger pose
    rot16_offset = R.from_quat(rot14_final).apply([0.0317095909367246, 0, -0.0016013564954687])
    rot16_initial = [math.sqrt(2)/2, -0, -0, -math.sqrt(2)/2]
    rot16_final = rotation_concat(rot2_new, rot16_initial)
    rot16_final = rotation_concat(rot3_new, rot16_final)
    rot16_final = rotation_concat(rot4_new, rot16_final)
    rot16_final = rotation_concat(rot5_new, rot16_final)
    rot16_final = rotation_concat(rot6_new, rot16_final)
    rot16_final = rotation_concat(rot7_new, rot16_final)
    rot16_final = rotation_concat(rot8_new, rot16_final)
    rot16_final = rotation_concat(rot14_new, rot16_final)
    rot16_new = rot14_new
    rot16_final = rotation_concat(rot16_new, rot16_final)
    trans16, rot16 = trans14 + rot16_offset, rot16_final

    return [[trans1, rot1],
            [trans2, rot2],
            [trans3, rot3],
            [trans4, rot4],
            [trans5, rot5],
            [trans6, rot6],
            [trans7, rot7],
            [trans8, rot8],
            [trans9, rot9],
            [trans10, rot10],
            [trans11, rot11],
            [trans12, rot12],
            [trans13, rot13],
            [trans14, rot14],
            [trans15, rot15],
            [trans16, rot16]]

class ur5e_valid(ob.StateValidityChecker):
    def __init__(self, si, real_offset):
        super().__init__(si)
        self.real_offset_ = real_offset

    def isValid(self, dof_state):
        pose_array = get_pose_from_dof(dof_state)

        ur5e_self_col = []
        rots = []
        trans = []

        #print (dof_state[0], dof_state[1], dof_state[2], dof_state[3], dof_state[4], dof_state[5])
        #real_offset = np.array(state_tensor[0][:3])
        for t in range(16):
            rotation = np.array(pose_array[t][1])
            translation = np.array(pose_array[t][0] + self.real_offset_)
            rots.append(rotation)
            trans.append(translation)
            r1 = R.from_quat(rotation)
            tf = fcl.Transform(r1.as_matrix(), translation)
            ur5e_self_col.append(fcl.CollisionObject(ur5e_collision_models[t], tf))

        request = fcl.CollisionRequest()
        result = fcl.CollisionResult()
        self_collision_flag = False

        for t in range(7):
            if t != 0:
                if fcl.collide(ur5e_self_col[t], plane_obj, request, result):
                    self_collision_flag = True
                    break
            col_with_other_part = False
            for q in range(7):
                if q < t-1 or q > t + 1:
                    if fcl.collide(ur5e_self_col[t], ur5e_self_col[q], request, result):
                        col_with_other_part = True
                        break
            if col_with_other_part:
                self_collision_flag = True
                break

        env_collision_flag = False
        manager1 = fcl.DynamicAABBTreeCollisionManager()
        manager1.registerObjects(ur5e_self_col)
        manager1.setup()

        manager2 = fcl.DynamicAABBTreeCollisionManager()
        manager2.registerObjects(object_collision_models)
        manager2.setup()

        req = fcl.CollisionRequest(num_max_contacts = 100, enable_contact = True)
        rdata = fcl.CollisionData(request = req)
        manager1.collide(manager2, rdata, fcl.defaultCollisionCallback)
        if rdata.result.is_collision:
            env_collision_flag = True

        return self_collision_flag == False and env_collision_flag == False

"""
Check Ur5e for collision
"""
def ur5e_in_collision(dof_result, real_offset):

    pose_array = get_pose_from_dof(dof_result)

    ur5e_self_col = []
    rots = []
    trans = []
    #real_offset = np.array(state_tensor[0][:3])
    for t in range(16):
        rotation = np.array(pose_array[t][1])
        translation = np.array(pose_array[t][0] + real_offset)
        rots.append(rotation)
        trans.append(translation)
        r1 = R.from_quat(rotation)
        tf = fcl.Transform(r1.as_matrix(), translation)
        ur5e_self_col.append(fcl.CollisionObject(ur5e_collision_models[t], tf))

    request = fcl.CollisionRequest()
    result = fcl.CollisionResult()
    self_collision_flag = False

    for t in range(7):
        if t != 0:
            if fcl.collide(ur5e_self_col[t], plane_obj, request, result):
                self_collision_flag = True
                break
        col_with_other_part = False
        for q in range(7):
            if q < t-1 or q > t + 1:
                if fcl.collide(ur5e_self_col[t], ur5e_self_col[q], request, result):
                    col_with_other_part = True
                    break
        if col_with_other_part:
            self_collision_flag = True
            break

    env_collision_flag = False
    manager1 = fcl.DynamicAABBTreeCollisionManager()
    manager1.registerObjects(ur5e_self_col)
    manager1.setup()

    manager2 = fcl.DynamicAABBTreeCollisionManager()
    manager2.registerObjects(object_collision_models)
    manager2.setup()

    req = fcl.CollisionRequest(num_max_contacts = 100, enable_contact = True)
    rdata = fcl.CollisionData(request = req)
    manager1.collide(manager2, rdata, fcl.defaultCollisionCallback)
    if rdata.result.is_collision:
        env_collision_flag = True

    return self_collision_flag == True or env_collision_flag == True

space = ob.RealVectorStateSpace(0)
space.addDimension(-6.28, 6.28)
space.addDimension(-6.28, 6.28)
space.addDimension(-3.14, 3.14)
space.addDimension(-6.28, 6.28)
space.addDimension(-6.28, 6.28)
space.addDimension(-6.28, 6.28)

space.addDimension(0, 3.14)
space.addDimension(0, 3.14)
space.addDimension(0, 3.14)
space.addDimension(-3.14, 0)
space.addDimension(-3.14, 0)
space.addDimension(-3.14, 0)

#### Prepare Tensors ####
gym.prepare_sim(sim)

if not os.path.exists("graphics_images"):
    os.mkdir("graphics_images")

urdf_string = ''
with open("../../assets/urdf/ur5e_mimic_real.urdf") as f:
    urdf_str = f.read()

ik_solver = IK("base_link", "robotiq_85_base_link", urdf_string = urdf_str)
print(ik_solver.joint_names)
input_captured_flag = False
target_set = False
flag = False
observe = False
frame_count = 0
grasps_to_simulate_cam = []
object_idx = 0
grasp_idx = 0
total_grasps = 0
last_collision_frame = 0
dof_set_frame = 0
dof_result = None
dof_target = None
track_angle_path = []

# simulation loop
while not gym.query_viewer_has_closed(viewer):

    # step the physics simulation
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    #update viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # render the camera sensors
    gym.render_all_camera_sensors(sim)

    if flag == False:
        print(f'Simulating result for {objects[object_idx]}')
        target_pos = target_positions[object_idx]
        target_rot = target_rotations[object_idx]
        target_pos = gymapi.Vec3(target_pos[0], target_pos[1], target_pos[2])
        target_pos_init = gymapi.Vec3(target_pos.x, target_pos.y, target_pos.z + 0.1)
        target_rot = gymapi.Quat(target_rot[0], target_rot[1], target_rot[2], target_rot[3])
        robot_offset = R.from_quat([target_rot.x, target_rot.y, target_rot.z, target_rot.w]).apply([0., 0., object_offset[object_idx]])
        robot_pos = gymapi.Vec3(-robot_offset[0] + target_pos.x - ur5e_2f85_pose.p.x, -robot_offset[2] + target_pos.z - ur5e_2f85_pose.p.z,+robot_offset[1] + -target_pos.y + ur5e_2f85_pose.p.y)
        robot_pos_init = gymapi.Vec3(-robot_offset[0] + target_pos_init.x - ur5e_2f85_pose.p.x, -robot_offset[2] + target_pos_init.z - ur5e_2f85_pose.p.z,+robot_offset[1] + -target_pos_init.y + ur5e_2f85_pose.p.y)
        robot_rot = quaternion_multiply(gymapi.Quat(-math.sqrt(2)/2, 0, 0, math.sqrt(2)/2), target_rot)
        for k in range(1000):
            seed_state = [np.random.rand()]*ik_solver.number_of_joints
            dof_result = ik_solver.get_ik(seed_state, robot_pos.x, robot_pos.y, robot_pos.z , robot_rot.x, robot_rot.y, robot_rot.z, robot_rot.w)
            if(dof_result):
                print(f'ROBOT FRAME: {seed_state} {robot_pos.x} {robot_pos.y} {robot_pos.z} {robot_rot.x} {robot_rot.y} {robot_rot.z} {robot_rot.w}')
                break

        if dof_result:
                    _state_tensor = gym.acquire_rigid_body_state_tensor(sim)
                    state_tensor = gymtorch.wrap_tensor(_state_tensor)
                    state_tensor_cpu = state_tensor.cpu()

                    real_offset = np.array(state_tensor_cpu[0][:3])
                    dof_result = dof_result + (0.0, 0.0, 0.0, -0.0, -0.0, -0.0)

                    end_state_collision = ur5e_in_collision(dof_result, real_offset)

                    if end_state_collision:
                        print(f'End state collision!!')
                        pass
                    else:
                        print("Simulating Grasp")

                        pose_array = get_pose_from_dof(dof_result[:12])

                        rots = []
                        trans = []
                        for t in range(16):
                            rotation = np.array(pose_array[t][1])
                            translation = np.array(pose_array[t][0] + real_offset)
                            rots.append(rotation)
                            trans.append(translation)

                        gym.clear_lines(viewer)

                        # draw object lines
                        for (object, object_file) in zip(objects,objects_collision_mesh_files):
                            collision_mesh = obj_reader(object_file)
                            object_vertices, object_faces = collision_mesh.get_vertices(), collision_mesh.get_faces()
                            object_vertices+= np.array([object_poses[object].x, object_poses[object].y, object_poses[object].z])
                            object_lines = []

                            for v1, v2, v3 in object_faces:
                                object_lines += list(object_vertices[v1])
                                object_lines += list(object_vertices[v2])
                                object_lines += list(object_vertices[v1])
                                object_lines += list(object_vertices[v3])
                                object_lines += list(object_vertices[v2])
                                object_lines += list(object_vertices[v3])

                            gym.add_lines(viewer, envs[-1], len(object_lines)//6, object_lines, [1, 0, 0])

                        # draw robot lines
                        for i in range(16):
                            parts_path = ur5e_collision_parts[i]
                            collision_mesh = stl_reader(asset_root + parts_path)
                            collision_mesh.transform(ur5e_rotations[i], ur5e_translations[i])
                            verts, tris = collision_mesh.get_vertices(), collision_mesh.get_faces()
                            all_lines = []
                            # print(verts, tris)
                            r2 = R.from_quat(rots[i])
                            for t in range(len(verts)):
                                verts[t] = r2.apply(verts[t])
                                verts[t] += np.array(trans[i])

                            for v1, v2, v3 in tris:
                                all_lines += list(verts[v1])
                                all_lines += list(verts[v2])
                                all_lines += list(verts[v1])
                                all_lines += list(verts[v3])
                                all_lines += list(verts[v2])
                                all_lines += list(verts[v3])

                            gym.add_lines(viewer, envs[-1], len(all_lines)//6, all_lines, [1, 0, 0])

                        for draw_index in range(100):
                            gym.simulate(sim)
                            gym.fetch_results(sim, True)

                            # update the viewer
                            gym.step_graphics(sim)
                            gym.draw_viewer(viewer, sim, True)


                            gym.sync_frame_time(sim)

                        # start ompl planning
                        si = ob.SpaceInformation(space)

                        validityChecker = ur5e_valid(si, real_offset)
                        si.setStateValidityChecker(validityChecker)
                        si.setStateValidityCheckingResolution(0.05)
                        si.setup()

                        col_free_path_found = False
                        trial_counter = 0
                        pathlist = []

                        if ompl_path[object_idx] is None:
                            while trial_counter <= 7:
                                trial_counter += 1

                                start = ob.State(space)
                                start[0] = track_last_pose[0]
                                start[1] = track_last_pose[1]
                                start[2] = track_last_pose[2]
                                start[3] = track_last_pose[3]
                                start[4] = track_last_pose[4]
                                start[5] = track_last_pose[5]

                                goal = ob.State(space)
                                goal[0] = dof_result[0]
                                goal[1] = dof_result[1]
                                goal[2] = dof_result[2]
                                goal[3] = dof_result[3]
                                goal[4] = dof_result[4]
                                goal[5] = dof_result[5]

                                pdef = ob.ProblemDefinition(si)
                                pdef.setStartAndGoalStates(start, goal)
                                shortestPathObjective = ob.PathLengthOptimizationObjective(si)
                                pdef.setOptimizationObjective(shortestPathObjective)
                                optimizingPlanner = og.RRTstar(si)
                                optimizingPlanner.setProblemDefinition(pdef)
                                optimizingPlanner.setRange(1000000)
                                optimizingPlanner.setup()
                                temp_res = optimizingPlanner.solve(100)
                                res = None
                                path = pdef.getSolutionPath()
                                pathlist = []

                                for i in range(path.getStateCount()):
                                    state = path.getState(i)
                                    pathlist.append((state[0], state[1],
                                                     state[2], state[3],
                                                     state[4], state[5]))

                                res = pathlist

                                temp_difference = 0

                                for dof_index in range(len(pathlist[-1])):
                                    temp_difference += (pathlist[-1][dof_index] - dof_result[dof_index])**2

                                col_free_path_found = (temp_difference < 1e-4)

                                if col_free_path_found:
                                    break
                        else:
                            pathlist = ompl_path[object_idx]
                            col_free_path_found = True

                        if col_free_path_found:
                            _body_tensor = gym.acquire_rigid_body_state_tensor(sim)
                            body_tensor = gymtorch.wrap_tensor(_body_tensor)

                            print(f'OMPL solution found, moving arm')

                            if ompl_path[object_idx] is None:
                                simulation_result_file = "/home/kaykay/isaacgym/python/contact_graspnet/SimulationResults/Real{}_ompl.npy".format(objects[object_idx])
                                simulation_data = {'pathlist': pathlist}
                                np.save(simulation_result_file, simulation_data)

                            track_last_pose = dof_result

                            for gripper_index in range(100):
                                # step the physics
                                gym.simulate(sim)
                                gym.fetch_results(sim, True)

                                # update the viewer
                                gym.step_graphics(sim)
                                gym.draw_viewer(viewer, sim, True)

                                gym.sync_frame_time(sim)
                                gym.set_dof_target_position(envs[-1], likj, 0.0)
                                gym.set_dof_target_position(envs[-1], lifj, 0.0)
                                gym.set_dof_target_position(envs[-1], lokj, 0.0)
                                gym.set_dof_target_position(envs[-1], rikj, -0.0)
                                gym.set_dof_target_position(envs[-1], rifj, -0.0)
                                gym.set_dof_target_position(envs[-1], rokj, -0.0)

                            for pathseg in pathlist:
                                for path_seg_index in range(100):
                                    # step the physics
                                    gym.simulate(sim)
                                    gym.fetch_results(sim, True)

                                    # update the viewer
                                    gym.step_graphics(sim)
                                    gym.draw_viewer(viewer, sim, True)

                                    gym.sync_frame_time(sim)

                                    gym.set_dof_target_position(envs[-1], spj, pathseg[0])
                                    gym.set_dof_target_position(envs[-1], slj, pathseg[1])
                                    gym.set_dof_target_position(envs[-1], ej,  pathseg[2])
                                    gym.set_dof_target_position(envs[-1], wj1, pathseg[3])
                                    gym.set_dof_target_position(envs[-1], wj2, pathseg[4])
                                    gym.set_dof_target_position(envs[-1], wj3, pathseg[5])

                            #assign tensor here
                            for assign_pose_index in range(7):

                                temp_trans, temp_rot = pose_array[assign_pose_index]

                                state_tensor[assign_pose_index+2][0] = temp_trans[0]
                                state_tensor[assign_pose_index+2][1] = temp_trans[1]
                                state_tensor[assign_pose_index+2][2] = temp_trans[2]
                                state_tensor[assign_pose_index+2][3] = temp_rot[0]
                                state_tensor[assign_pose_index+2][4] = temp_rot[1]
                                state_tensor[assign_pose_index+2][5] = temp_rot[2]
                                state_tensor[assign_pose_index+2][6] = temp_rot[3]

                            for tensor_assign_index in range(100):
                                gym.simulate(sim)
                                gym.fetch_results(sim, True)

                                # update the viewer
                                gym.step_graphics(sim)
                                gym.draw_viewer(viewer, sim, True)

                                gym.sync_frame_time(sim)

                            flag = True
                            dof_set_frame = frame_count
                        else:
                            print(f'Collision free path not found')
                            end_state_collision = True
                            object_idx += 1

        else:
            object_idx += 1
    if (frame_count - dof_set_frame > 10) and flag and observe == False:
        print(f'Enter - Grip(g) or Next Grasp(n) or Observe(o) or Lift(l): ')
        proceed = input()
        if proceed == 'n':
            flag = False
            object_idx += 1
        elif proceed == 'g':
            dof_set_frame = frame_count
            for gripper_idx in range(100):
                # step the physics
                gym.simulate(sim)
                gym.fetch_results(sim, True)

                # update the viewer
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, True)

                gym.sync_frame_time(sim)

                gym.set_dof_target_position(envs[-1], likj, object_gripper_dof[object_idx])
                gym.set_dof_target_position(envs[-1], lifj, object_gripper_dof[object_idx])
                gym.set_dof_target_position(envs[-1], lokj, object_gripper_dof[object_idx])
                gym.set_dof_target_position(envs[-1], rikj, -object_gripper_dof[object_idx])
                gym.set_dof_target_position(envs[-1], rifj, -object_gripper_dof[object_idx])
                gym.set_dof_target_position(envs[-1], rokj, -object_gripper_dof[object_idx])
        elif proceed == 'o':
            observe = True
        elif proceed == 'l':
            dof_set_frame = frame_count
            track_last_pose = default_dof_pos.tolist()
            for path_seg_index in range(100):
                # step the physics
                gym.simulate(sim)
                gym.fetch_results(sim, True)

                # update the viewer
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, True)

                gym.sync_frame_time(sim)

                gym.set_dof_target_position(envs[-1], spj, default_dof_pos[0].item())
                gym.set_dof_target_position(envs[-1], slj, default_dof_pos[1].item())
                gym.set_dof_target_position(envs[-1], ej,  default_dof_pos[2].item())
                gym.set_dof_target_position(envs[-1], wj1, default_dof_pos[3].item())
                gym.set_dof_target_position(envs[-1], wj2, default_dof_pos[4].item())
                gym.set_dof_target_position(envs[-1], wj3, default_dof_pos[5].item())

            pose_array = get_pose_from_dof(track_last_pose)
            #assign tensor here
            for assign_pose_index in range(7):

                temp_trans, temp_rot = pose_array[assign_pose_index]

                state_tensor[assign_pose_index+2][0] = temp_trans[0]
                state_tensor[assign_pose_index+2][1] = temp_trans[1]
                state_tensor[assign_pose_index+2][2] = temp_trans[2]
                state_tensor[assign_pose_index+2][3] = temp_rot[0]
                state_tensor[assign_pose_index+2][4] = temp_rot[1]
                state_tensor[assign_pose_index+2][5] = temp_rot[2]
                state_tensor[assign_pose_index+2][6] = temp_rot[3]

            for tensor_assign_index in range(100):
                gym.simulate(sim)
                gym.fetch_results(sim, True)

                # update the viewer
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, True)

                gym.sync_frame_time(sim)

    frame_count += 1
    gym.sync_frame_time(sim)

# cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
