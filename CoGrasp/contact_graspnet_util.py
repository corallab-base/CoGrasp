from inference import inference

import subprocess
import os

def generate_grasps(input_path):
    K = [911.445649104, 0, 641.169, 0, 891.51236121, 352.77, 0, 0, 1]
    # os.system(f'python contact_graspnet/inference.py --np_path={input_path} --K="{K}" --local_regions --filter_grasps')
    subprocess.run(['python', 'contact_graspnet/inference.py', '--np_path', input_path, '--K', '[911.445649104, 0, 641.169, 0, 891.51236121, 352.77, 0, 0, 1]', '--local_regions', '--filter_grasps'])
    # return pred_grasps_cam, scores, contact_pts, gripper_openings
