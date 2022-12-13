import numpy as np
import os
import open3d as o3d
import uuid
from scipy.spatial.distance import cdist
from contact_graspnet.contact_graspnet import mesh_utils
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

def get_normal_score(gripper_grasp, hand):
    hand_pcd = o3d.geometry.PointCloud()
    hand_pcd.points = o3d.utility.Vector3dVector(np.concatenate((hand[15:25], hand[110:120])))
    hand_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    hand_normals = np.array(hand_pcd.normals)
    hand_avg_normal = np.mean(hand_normals, axis=0)
    gripper_normal = gripper_grasp[:3, 2]
    return np.abs(np.dot(gripper_normal,hand_avg_normal))

def get_collision_score(gripper_pc, hand):
    dist = cdist(hand, gripper_pc)
    return np.mean(dist)

def get_gripper_pc(gripper_control_points, gripper_grasp):
    cam_pose = np.eye(4)
    pts = np.matmul(gripper_control_points, gripper_grasp[:3, :3].T)
    # pts -= object_translation
    pts += np.expand_dims(gripper_grasp[:3, 3], 0)
    pts_homog = np.concatenate((pts, np.ones((7, 1))),axis=1)
    pts = np.dot(pts_homog, cam_pose.T)[:,:3]
    return pts

def get_grasp_score(normal_score, collision_score, normal_median, collision_median):
    # check from medium as threshold
    return 1 if (normal_score >= normal_median and collision_score >= collision_median) else 0

def get_nearest_score(hand, gripper):
    dist = cdist(hand, gripper)
    return np.min(dist)

def ablation_study(factor1, factor2):
    success_Sa = []
    success_Sd = []
    success_Sn = []
    fail_Sa = []
    fail_Sd = []
    fail_Sn = []
    root_directory = '/home/kaykay/isaacgym/python/contact_graspnet/pruning_network_data/train'

    gripper_width=0.08
    gripper = mesh_utils.create_gripper('panda')
    gripper_control_points = gripper.get_control_point_tensor(1, False, convex_hull=False).squeeze()
    mid_point = 0.5*(gripper_control_points[1, :] + gripper_control_points[2, :])
    grasp_line_plot = np.array([np.zeros((3,)), mid_point, gripper_control_points[1], gripper_control_points[3],
                            gripper_control_points[1], gripper_control_points[2], gripper_control_points[4]])
    gripper_control_points_closed = grasp_line_plot.copy()
    gripper_control_points_closed[2:,0] = np.sign(grasp_line_plot[2:,0]) * gripper_width/2

    counter = 0
    for file in os.listdir(root_directory):
        counter += 1
        data = np.load(os.path.join(root_directory,file), allow_pickle=True)[()]
        obj_pc = np.array(data['obj_pc'])
        hand = np.array(data['hand'])
        gripper_pred = np.array(data['gripper_pred'])
        gripper_pcs = []
        normal_scores = []
        collision_scores = []
        nearest_scores = []

        for grasp in gripper_pred:
            gripper_pc = get_gripper_pc(gripper_control_points_closed, grasp)
            normal_score = get_normal_score(grasp, hand)
            collision_score = get_collision_score(gripper_pc, hand)
            nearest_score = get_nearest_score(hand, gripper_pc)

            gripper_pcs.append(gripper_pc)
            normal_scores.append(normal_score)
            collision_scores.append(collision_score)
            nearest_scores.append(nearest_score)

        gripper_pcs = np.array(gripper_pcs)
        normal_scores = np.array(normal_scores)
        collision_scores = np.array(collision_scores)
        nearest_scores = np.array(nearest_scores)

        threshold1 = int(normal_scores.size * factor1)
        threshold2 = int(normal_scores.size * factor2)
        # print(normal_scores.size,  threshold)

        for i, grasp in enumerate(gripper_pred):
            grasp_score = get_grasp_score(normal_scores[i], collision_scores[i], normal_scores[threshold1], collision_scores[threshold2])
            if (grasp_score == 1):
                success_Sa.append(normal_scores[i])
                success_Sd.append(collision_scores[i])
                success_Sn.append(nearest_scores[i])
            else:
                fail_Sa.append(normal_scores[i])
                fail_Sd.append(collision_scores[i])
                fail_Sn.append(nearest_scores[i])

    # print(f'For factors: {factor1}, factor{2}')
    # print(f'Success Sa: {np.mean(success_Sa)}')
    # print(f'Success Sd: {np.mean(success_Sd)}')
    # print(f'Success Sn: {np.mean(success_Sn)}')
    # print(f'Total Success: {len(success_Sa)}')
    # print(f'Fail Sa: {np.mean(fail_Sa)}')
    # print(f'Fail Sd: {np.mean(fail_Sd)}')
    # print(f'Fail Sn: {np.mean(fail_Sn)}')
    # print(f'Total Fail: {len(fail_Sa)}')
    return np.mean(success_Sa), np.mean(success_Sd), np.mean(success_Sn), len(success_Sa)


def build_data_for_evaluation():
    success_grasps = []
    fail_grasps = []
    root_directory = '/home/kaykay/isaacgym/python/contact_graspnet/pruning_network_data/train'

    gripper_width=0.08
    gripper = mesh_utils.create_gripper('panda')
    gripper_control_points = gripper.get_control_point_tensor(1, False, convex_hull=False).squeeze()
    mid_point = 0.5*(gripper_control_points[1, :] + gripper_control_points[2, :])
    grasp_line_plot = np.array([np.zeros((3,)), mid_point, gripper_control_points[1], gripper_control_points[3],
                            gripper_control_points[1], gripper_control_points[2], gripper_control_points[4]])
    gripper_control_points_closed = grasp_line_plot.copy()
    gripper_control_points_closed[2:,0] = np.sign(grasp_line_plot[2:,0]) * gripper_width/2

    counter = 0
    for file in os.listdir(root_directory):
        counter += 1
        data = np.load(os.path.join(root_directory,file), allow_pickle=True)[()]
        obj_pc = np.array(data['obj_pc'])
        hand = np.array(data['hand'])
        gripper_pred = np.array(data['gripper_pred'])
        gripper_pcs = []
        normal_scores = []
        collision_scores = []

        for grasp in gripper_pred:
            gripper_pc = get_gripper_pc(gripper_control_points_closed, grasp)
            normal_score = get_normal_score(grasp, hand)
            collision_score = get_collision_score(gripper_pc, hand)

            gripper_pcs.append(gripper_pc)
            normal_scores.append(normal_score)
            collision_scores.append(collision_score)

        gripper_pcs = np.array(gripper_pcs)
        normal_scores = np.array(normal_scores)
        collision_scores = np.array(collision_scores)

        for i, grasp in enumerate(gripper_pred):
            grasp_score = get_grasp_score(normal_scores[i], collision_scores[i], np.median(normal_scores), np.median(collision_scores))

            filename = "/home/kaykay/isaacgym/python/contact_graspnet/pruning_network_data/train_temp/{}.npy".format(uuid.uuid4().hex)
            required_data = {'obj_pc': obj_pc, 'hand_pc': hand, 'gripper_pc': gripper_pcs[i], 'normal_score': normal_scores[i], 'collision_score': collision_scores[i], 'grasp_score': grasp_score}
            np.save(filename, required_data)

            if (grasp_score == 1):
                success_grasps.append(filename)
            else:
                fail_grasps.append(filename)
        print(f'Data saved for file: {file} with counter: {counter}')

    # summary = "/home/kaykay/isaacgym/python/contact_graspnet/pruning_network_data/train_temp/summary_{}".format(uuid.uuid4().hex)
    # summary_data = {'normals': np.array(normals), 'collisions': np.array(collisions)}
    # np.save(summary, summary_data)
    # print(f'Summary Data saved with name: {summary}')

    textfile = open("/home/kaykay/isaacgym/python/contact_graspnet/pruning_network_data/success.txt", "w")
    for element in success_grasps:
        textfile.write(element + "\n")
    textfile.close()

    textfile = open("/home/kaykay/isaacgym/python/contact_graspnet/pruning_network_data/fail.txt", "w")
    for element in fail_grasps:
        textfile.write(element + "\n")
    textfile.close()

def plot():
    success_Sa = [0.6276821709772114, 0.6155815667882416, 0.6258144452193826, 0.6150351103656565, 0.609999088852437, 0.6097360167416712, 0.6083207864772767, 0.6208901085391063, 0.6227224347506015, 0.6140424398067998, 0.6195580429494606, 0.6244562083146659, 0.6137341175280153]
    success_Sd = [0.1335254171205536, 0.13248748712277073, 0.1327260369271979, 0.13044930185833803, 0.13151887983396301, 0.13047951457207724, 0.1293918641664867, 0.13206526916847228, 0.13201037232234225, 0.1309847398225944, 0.13160333587855388, 0.13216861978985067, 0.13017710523078704]
    success_Sn = [0.04115458964396263, 0.03993521781106894, 0.0402584616531028, 0.03888342113045098, 0.03930502249798252, 0.0383011009211012, 0.0373719557106579, 0.040067256719865524, 0.04000418836684113, 0.0385253176394628, 0.03967477678762984, 0.03964639672504049, 0.038017504442164965]
    success_tot = np.array([23621, 23592, 23836, 24257, 25113, 24913, 24562, 23680, 23542, 23518])
    fail_Sa = [0.6074939900212998, 0.6196497141390996, 0.6082362135062173, 0.6115406445943645, 0.6114365792926865, 0.6188571430743959, 0.6083207864772767, 0.6114676711710157, 0.6099622893119415, 0.6093503950317527, 0.6107178069721232, 0.6126571013220292, 0.6084364221934481]
    fail_Sd = [0.1307162551879343, 0.1318886673099147, 0.13231104713703207, 0.13085465968705326, 0.13044831255534395, 0.13146232604805713, 0.1293918641664867, 0.1330809231939174, 0.13296895032842126, 0.13150072420862044, 0.13091326322457375, 0.13256896847655356, 0.13051760461417852]
    fail_Sn = [0.03875360184102167, 0.03968424916175029, 0.04010527221465887, 0.03870837441574722, 0.03827637334734487, 0.039343133033027916, 0.0373719557106579, 0.04082577125347807, 0.04077739308659253, 0.03982067690372262, 0.0393026031838291, 0.040021181211487664, 0.03861604040536632]
    fail_tot = np.array([22634, 23698, 23892, 24224, 25113, 25101, 24893, 23472, 23230, 23097])
    x = np.array([.25, .3, .35, .45, .5, .55, .6, .65, .7, .75])
    # y = [.228, .2537, .2497, .2481, .2182, .2413, .2421, .2467, .229, .2324, .2526, .2518, .2325]
    # y = [0.6276821709772114, 0.6155815667882416, 0.6258144452193826, 0.6150351103656565, 0.609999088852437, 0.6097360167416712, 0.6083207864772767, 0.6208901085391063, 0.6227224347506015, 0.6140424398067998, 0.6195580429494606, 0.6244562083146659, 0.6137341175280153]
    # y2 = [0.6074939900212998, 0.6196497141390996, 0.6082362135062173, 0.6115406445943645, 0.6114365792926865, 0.6188571430743959, 0.6083207864772767, 0.6114676711710157, 0.6099622893119415, 0.6093503950317527, 0.6107178069721232, 0.6126571013220292, 0.6084364221934481]
    y = []
    x_y_spline = make_interp_spline(x, success_tot)
    x_y2_spline = make_interp_spline(x, fail_tot)
    x_ = np.linspace(x.min(), x.max(), 500)
    y_ = x_y_spline(x_)
    y2_ = x_y2_spline(x_)
    # for i in range(13):
    #     y.append((success_Sd[i])/success_tot[i])
    plt.plot(x_, y_, color='r', label=r'$\lambda_d$ fixed')
    plt.plot(x_, y2_, color='b', label=r'$\lambda_a$ fixed')

    plt.ylabel('Total Valid Grasp')
    plt.xlabel(r'Percentile of Score as Threshold ($\lambda_a$ or $\lambda_d$)')
    plt.legend(loc="upper left")
    plt.axvline(x=0.5, linestyle='--', color="black")
    # plt.title('Total Grasps vs Approach Score Threshold')
    plt.grid()
    plt.show()

# build_data_for_evaluation()
# factors = [.2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8]
# normals_sa = []
# normals_sd = []
# normals_sn = []
# normals_tot = []
# collisions_sa = []
# collisions_sd = []
# collisions_sn = []
# collisions_tot = []
# for factor in factors:
#     print(factor, 0.5)
#     a, d, n, tot = ablation_study(factor, .5)
#     normals_sa.append(a)
#     normals_sd.append(d)
#     normals_sn.append(n)
#     normals_tot.append(tot)
# for factor in factors:
#     print(0.5, factor)
#     a, d, n, tot = ablation_study(.5, factor)
#     collisions_sa.append(a)
#     collisions_sd.append(d)
#     collisions_sn.append(n)
#     collisions_tot.append(tot)
#
# print(f'Normals tot: {normals_tot}')
# print(normals_sa)
# print(normals_sd)
# print(normals_sn)
# print(f'Collisions tot: {collisions_tot}')
# print(collisions_sa)
# print(collisions_sd)
# print(collisions_sn)

plot()
