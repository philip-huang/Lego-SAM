# run on clean images to get OnlineLegoInferer calibrated_T (update in lego_online_infer.py)

import lego_online_infer
import lego_visualize
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import time

import numpy as np
import os


def num_frames(task, SIM_DATA_ROOT):
    assembly_key = task
    live_image_dir1 = Path(SIM_DATA_ROOT, "cam1", assembly_key)
    live_image_dir2 = Path(SIM_DATA_ROOT, "cam2", assembly_key)

    # open the live image folders and count the number of images
    live_images_cam1 = list(live_image_dir1.glob("*.jpg"))
    live_images_cam2 = list(live_image_dir2.glob("*.jpg"))
    length = min(len(live_images_cam1), len(live_images_cam2))  # Ensure both cameras have the same number of images
    assert(length > 0)
    
    return length


def load_frame(task, frame_idx, SIM_DATA_ROOT):
    assembly_key = task
    live_image_dir1 = Path(SIM_DATA_ROOT, "cam1", assembly_key)
    live_image_dir2 = Path(SIM_DATA_ROOT, "cam2", assembly_key)

    img1_name = f'{frame_idx:06d}.jpg'
    img2_name = f'{frame_idx:06d}.jpg'
    live_image_cam1_np = cv2.cvtColor(cv2.imread(str(live_image_dir1 / img1_name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)  # Convert to RGB
    live_image_cam2_np = cv2.cvtColor(cv2.imread(str(live_image_dir2 / img2_name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)  # Convert to RGB
    live_depth_cam1_np = np.load(str(live_image_dir1 / img1_name).replace('.jpg', '_depth.npz'))['arr_0']
    live_depth_cam2_np = np.load(str(live_image_dir2 / img2_name).replace('.jpg', '_depth.npz'))['arr_0']

    if live_image_cam1_np is None or live_image_cam2_np is None:
        print(f"Error: Failed to read one of the live images for assembly key '{assembly_key}'.")
        exit(0)

    return live_image_cam1_np, live_image_cam2_np, live_depth_cam1_np, live_depth_cam2_np


if __name__ == '__main__':
    calibrate = True
    task = 'fish_high'
    i = -1 # -1: length-1

    if calibrate:
        compute_new_T = True # true: use ransac, false: use old calibration
        reject_bad_T = False # true: reject align if outside old calibration, false: no rejection
    else:
        compute_new_T = False # true: use ransac, false: use old calibration
        reject_bad_T = True # true: reject align if outside old calibration, false: no rejection

    ################

    # Initialize inferer
    SIM_DATA_ROOT = "outputs" # Root dir where sim_cam1/, sim_cam2/ exist
    SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt" # Path to your SAM2 model
    SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"    # Path to your SAM2 config

    inferer = lego_online_infer.OnlineLegoInferer(
        sim_data_root_dir=SIM_DATA_ROOT,
        sam2_checkpoint_path=SAM2_CHECKPOINT,
        sam2_model_config_path=SAM2_CONFIG,
        device="cuda" # Use "cuda" if available and desired
    )

    ################

    length = num_frames(task, SIM_DATA_ROOT)
    if i == -1: i = length - 1
    live_image_cam1_np, live_image_cam2_np, live_depth_cam1_np, live_depth_cam2_np = load_frame(task, i, SIM_DATA_ROOT)

    results = inferer.infer_dual_camera(
        live_image_cam1_np,
        live_image_cam2_np,
        task,
        live_depth_cam1_np=live_depth_cam1_np,
        live_depth_cam2_np=live_depth_cam2_np,
        compute_new_T=compute_new_T,
        save=False,
        display=True,
    )


    for sim_id, data in results.details.items():
        if sim_id != results.best_sim_id: continue #
        print(f"  Sim ID {sim_id}: Cam1 IoU={data['cam1_iou']:.4f}, Cam2 IoU={data['cam2_iou']:.4f}, Combined IoU: {data['combined_iou']:.4f}")

    print('====================')

    print("self.calibrated_T = {'sim_cam1': (np.array(" + str(results.T1[0].tolist()) + "), " 
                                            + str(results.T1[1]) + "),"
                                            + "\n\t\t'sim_cam2': (np.array(" + str(results.T2[0].tolist()) + "), " 
                                            + str(results.T2[1]) + ")}")
    print()