import numpy as np
import trimesh
import open3d as o3d
import mesh_utils
import mayavi.mlab as mlab
from scipy.spatial.distance import cdist

def get_gripper_pc(gripper_control_points, gripper_grasp):
    cam_pose = np.eye(4)
    pts = np.matmul(gripper_control_points, gripper_grasp[:3, :3].T)
    # pts -= object_translation
    pts += np.expand_dims(gripper_grasp[:3, 3], 0)
    pts_homog = np.concatenate((pts, np.ones((7, 1))),axis=1)
    pts = np.dot(pts_homog, cam_pose.T)[:,:3]
    return pts


def plot_data(predictions, obj_pc, hand):

    gripper = mesh_utils.create_gripper('panda')
    gripper_control_points = gripper.get_control_point_tensor(1, False, convex_hull=False).squeeze()
    mid_point = 0.5*(gripper_control_points[1, :] + gripper_control_points[2, :])
    grasp_line_plot = np.array([np.zeros((3,)), mid_point, gripper_control_points[1], gripper_control_points[3],
                            gripper_control_points[1], gripper_control_points[2], gripper_control_points[4]])

    all_pts = []
    connections = []
    index = 0
    N = 7
    cam_pose = np.eye(4)

    for i,g in enumerate(predictions):
        gripper_control_points_closed = grasp_line_plot.copy()
        gripper_control_points_closed[2:,0] = np.sign(grasp_line_plot[2:,0]) * gripper_width/2

        pts = np.matmul(gripper_control_points_closed, g[:3, :3].T)
        # pts -= object_translation
        pts += np.expand_dims(g[:3, 3], 0)
        pts_homog = np.concatenate((pts, np.ones((7, 1))),axis=1)
        pts = np.dot(pts_homog, cam_pose.T)[:,:3]

        # if (np.min(pts[:,0]) > 0.2-object_translation[0] and np.max(pts[:,2]) < 0.3-object_translation[2]):
        all_pts.append(pts)
        connections.append(np.vstack([np.arange(index,   index + N - 1.5),
                                      np.arange(index + 1, index + N - .5)]).T)
        index += N

    fig = mlab.figure('Pred Grasps')
    mlab.view(azimuth=180, elevation=180, distance=0.2)
    mlab.points3d(obj_pc[:, 0], obj_pc[:, 1], obj_pc[:, 2], color=(.03921568627, .03921568627, .03921568627))
    mlab.points3d(hand[:, 0], hand[:, 1], hand[:, 2], color=(.90980392156, .74509803921, .67450980392))

    all_pts = np.vstack(all_pts)
    connections = np.vstack(connections)
    src = mlab.pipeline.scalar_scatter(all_pts[:,0], all_pts[:,1], all_pts[:,2])
    src.mlab_source.dataset.lines = connections
    src.update()
    lines =mlab.pipeline.tube(src, tube_radius=.0008, tube_sides=12)
    mlab.pipeline.surface(lines, color=(0,1.,0), opacity=1.0)
    mlab.show()


filename = '/home/kaykay/isaacgym/python/contact_graspnet/pruning_network_data/train/banana_68212a95cf924584a2a24af41516d1b9.npy'
data = np.load(filename, allow_pickle=True)
data = data[()]
obj_pc = np.array(data['obj_pc'])
contact_pts = np.array(data['gripper_contact_pts'])
hand = np.array(data['hand'])
gripper_pred = np.array(data['gripper_pred'])
hand_pcd = o3d.geometry.PointCloud()
hand_pcd.points = o3d.utility.Vector3dVector(np.concatenate((hand[15:25], hand[110:120])))
hand_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
hand_normals = np.array(hand_pcd.normals)
hand_avg_normal = np.mean(hand_normals, axis=0)
# print(f'Hand normal is: {hand_avg_normal}')

gripper_width=0.08
gripper_predictions=[]
score_normal = []
score_distance = []
score_test_normal = []
score_test_distance = []

gripper = mesh_utils.create_gripper('panda')
gripper_control_points = gripper.get_control_point_tensor(1, False, convex_hull=False).squeeze()
mid_point = 0.5*(gripper_control_points[1, :] + gripper_control_points[2, :])
grasp_line_plot = np.array([np.zeros((3,)), mid_point, gripper_control_points[1], gripper_control_points[3],
                        gripper_control_points[1], gripper_control_points[2], gripper_control_points[4]])
gripper_control_points_closed = grasp_line_plot.copy()
gripper_control_points_closed[2:,0] = np.sign(grasp_line_plot[2:,0]) * gripper_width/2

for i, gripper_grasp in enumerate(gripper_pred):
    gripper_normal = gripper_grasp[:3, 2]
    # print(f'Gripper Normal:{gripper_normal}')

    normal_product = -(np.dot(gripper_normal,hand_avg_normal))
    score_normal.append(normal_product)

    test_product = np.abs(np.dot(gripper_normal,hand_avg_normal))
    score_test_normal.append(test_product)

    dist = np.sqrt(((hand - contact_pts[i])**2).sum(1))
    avg_dist = np.mean(dist)
    score_distance.append(avg_dist)

    gripper_pc = get_gripper_pc(gripper_control_points_closed, gripper_grasp)
    test_dist = cdist(hand, gripper_pc)
    # print(test_dist)
    score_test_distance.append(np.mean(test_dist))

score_normal = np.array(score_normal)
score_distance = np.array(score_distance)
score_test_normal = np.array(score_test_normal)
score_test_distance = np.array(score_test_distance)

plot_data(gripper_pred, obj_pc, hand)

contact_ind = np.argpartition(score_distance, -10)[-10:]
print(f'Best grasps using distance: {score_distance[contact_ind]} with indices: {contact_ind}')
contact_predictions = gripper_pred[contact_ind]
plot_data(contact_predictions, obj_pc, hand)

test_contact_ind = np.argpartition(score_test_distance, -10)[-10:]
print(f'Best grasps using test distance: {score_test_distance[test_contact_ind]} with indices: {test_contact_ind}')
test_contact_predictions = gripper_pred[test_contact_ind]
plot_data(test_contact_predictions, obj_pc, hand)


normal_ind = np.argpartition(score_normal, -10)[-10:]
print(f'Best grasps using normal: {score_normal[normal_ind]} with indices: {normal_ind}')
normal_predictions = gripper_pred[normal_ind]
plot_data(normal_predictions, obj_pc, hand)

test_normal_ind = np.argpartition(score_test_normal, -10)[-10:]
print(f'Best grasps using test normal: {score_test_normal[test_normal_ind]} with indices: {test_normal_ind}')
test_normal_predictions = gripper_pred[test_normal_ind]
plot_data(test_normal_predictions, obj_pc, hand)
