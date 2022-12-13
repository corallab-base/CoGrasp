# from PruningNetwork.model import PruningNetwork
from model import PruningNetwork
import numpy as np
import torch
# from contact_graspnet import mesh_utils
from contact_graspnet.contact_graspnet import mesh_utils
import mayavi.mlab as mlab

def load_model(epoch):
    model = PruningNetwork()
    model.to(device)
    PATH = f'/home/kaykay/isaacgym/python/contact_graspnet/PruningNetwork/checkpoints/checkpoint_modified/epoch_{epoch}.pt'
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def merge_object_gripper_hand_pc(object_pc, gripper_pc, hand_pc):
    object_shape = object_pc.shape
    gripper_shape = gripper_pc.shape
    hand_shape = hand_pc.shape
    batch_size = object_shape[0]

    merged_xyz = torch.cat((object_pc, gripper_pc, hand_pc), 1)
    labels = [torch.zeros((object_shape[1],1), dtype=torch.float), torch.ones((gripper_shape[1],1), dtype=torch.float), -torch.ones((hand_shape[1],1), dtype=torch.float)]
    labels = torch.cat(labels, 0)
    labels = labels.expand(1, labels.shape[0], labels.shape[1])
    labels = torch.tile(labels, [batch_size, 1, 1])

    merged_points = torch.concat([merged_xyz, labels], -1)
    merged_points = torch.permute(merged_points, (0, 2, 1))

    return merged_xyz.contiguous(), merged_points.contiguous()

def get_gripper_pc(gripper_control_points, gripper_grasp):
    cam_pose = np.eye(4)
    pts = np.matmul(gripper_control_points, gripper_grasp[:3, :3].T)
    # pts -= object_translation
    pts += np.expand_dims(gripper_grasp[:3, 3], 0)
    pts_homog = np.concatenate((pts, np.ones((7, 1))),axis=1)
    pts = np.dot(pts_homog, cam_pose.T)[:,:3]
    return pts


def plot_data(predictions, obj_pc, hand_pc, gripper_control_points_closed):
    all_pts = []
    connections = []
    index = 0
    N = 7
    cam_pose = np.eye(4)

    for i,g in enumerate(predictions):
        pts = get_gripper_pc(gripper_control_points_closed, g)
        all_pts.append(pts)
        connections.append(np.vstack([np.arange(index,   index + N - 1.5),
                                      np.arange(index + 1, index + N - .5)]).T)
        index += N

    fig = mlab.figure('Pred Grasps')
    mlab.view(azimuth=180, elevation=180, distance=0.2)
    mlab.points3d(obj_pc[:, 0], obj_pc[:, 1], obj_pc[:, 2], color=(0.54296875, 0.26953125, 0.07421875))
    mlab.points3d(hand_pc[:, 0], hand_pc[:, 1], hand_pc[:, 2], color=(0.94140625, 0.7578125, 0.48828125))

    all_pts = np.vstack(all_pts)
    connections = np.vstack(connections)
    src = mlab.pipeline.scalar_scatter(all_pts[:,0], all_pts[:,1], all_pts[:,2])
    src.mlab_source.dataset.lines = connections
    src.update()
    lines =mlab.pipeline.tube(src, tube_radius=.0008, tube_sides=12)
    mlab.pipeline.surface(lines, color=(0,1.,0), opacity=1.0)
    mlab.show()

def evaluate(model, obj_pc, hand_pc, grasp, gripper_control_points_closed):
    gripper_pc = get_gripper_pc(gripper_control_points_closed, grasp)
    gripper_pc_pruning = np.expand_dims(gripper_pc, axis=0)
    hand_pc_pruning = np.expand_dims(hand_pc, axis=0)
    obj_pc_pruning = np.expand_dims(obj_pc, axis=0)
    obj_pc_pruning = torch.tensor(obj_pc_pruning, dtype=torch.float)
    hand_pc_pruning = torch.tensor(hand_pc_pruning, dtype=torch.float)
    gripper_pc_pruning = torch.tensor(gripper_pc_pruning, dtype=torch.float)

    merged_xyz, merged_features = merge_object_gripper_hand_pc(obj_pc_pruning, gripper_pc_pruning, hand_pc_pruning)
    merged_xyz = merged_xyz.cuda()
    merged_features = merged_features.cuda()
    res = model(merged_xyz, merged_features)

    # gc.collect()
    return res

def main():
    test_data_path = '/home/kaykay/isaacgym/python/contact_graspnet/pruning_network_data/train/power_drill_9472b11cfde3461087f490a13733ad50.npy'
    data = np.load(test_data_path, allow_pickle=True)[()]
    obj_pc = np.array(data['obj_pc'])
    hand_pc = np.array(data['hand'])
    gripper_pred = np.array(data['gripper_pred'])
    scores = np.array(data['scores'])

    gripper_width=0.08
    gripper = mesh_utils.create_gripper('panda')
    gripper_control_points = gripper.get_control_point_tensor(1, False, convex_hull=False).squeeze()
    mid_point = 0.5*(gripper_control_points[1, :] + gripper_control_points[2, :])
    grasp_line_plot = np.array([np.zeros((3,)), mid_point, gripper_control_points[1], gripper_control_points[3],
                            gripper_control_points[1], gripper_control_points[2], gripper_control_points[4]])
    gripper_control_points_closed = grasp_line_plot.copy()
    gripper_control_points_closed[2:,0] = np.sign(grasp_line_plot[2:,0]) * gripper_width/2

    plot_data(gripper_pred, obj_pc, hand_pc, gripper_control_points_closed)

    # plot top 5 predictions from contact_graspnet
    top_score_ind = np.argpartition(scores, -5)[-5:]
    top_gripper_pred = gripper_pred[top_score_ind]
    plot_data(top_gripper_pred, obj_pc, hand_pc, gripper_control_points_closed)

    model = load_model(175)
    model.eval()
    selected_pred = []
    pruning_network_scores = []

    for grasp in gripper_pred:
        res = evaluate(model, obj_pc, hand_pc, grasp, gripper_control_points_closed)
        if res>0.98:
            # print(res)
            selected_pred.append(grasp)
            pruning_network_scores.append(res)
    selected_pred = np.array(selected_pred)
    plot_data(selected_pred, obj_pc, hand_pc, gripper_control_points_closed)

    # plot top 5 predictions from pruning network
    top_res_ind = np.argpartition(pruning_network_scores, -5)[-5:]
    top_res_grasps = selected_pred[top_res_ind]
    plot_data(top_res_grasps, obj_pc, hand_pc, gripper_control_points_closed)

device = 'cuda'
if __name__ == '__main__':
    main()
