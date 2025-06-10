'''
Visualize cutouts and IoU
'''

import lego_online_infer
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import minimum_filter, maximum_filter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import sys
import os
import time
from tqdm import tqdm
import subprocess


def _fig_to_image(fig):
    FigureCanvas(fig)  # attach a canvas if not already attached
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape((h, w, 4))
    return img


def _combine(img, transformed_cutout, color, size=5):
    preprocessed_img = np.array(img)

    min_vals = minimum_filter(transformed_cutout.astype(np.uint8), size=size)
    max_vals = maximum_filter(transformed_cutout.astype(np.uint8), size=size)
    mask = (min_vals == 0) & (max_vals == 255)

    preprocessed_img[mask == True] = color

    return preprocessed_img


def _color_gradient(score):
    colors = [(0.0, "#FF0000"), (0.5, "#FF0000"), (0.87, "#FFE600"), (0.95, "#00FF00"), (1.0, "#00FF00")]
    cmap = LinearSegmentedColormap.from_list("hex_gradient", colors)
    return [c*255 for c in cmap(score)[:3]]


def visualize(inferer, img1, img2, results, save_path="", display_plt=False, output_np_array=True):
    _, _, transformed_cutouts1, transformed_cutouts2, best_id, best_score, details = results

    if best_id is not None:
        color1 = _color_gradient(details[best_id]['cam1_iou'])
        color2 = _color_gradient(details[best_id]['cam2_iou'])
    else:
        color1 = color2 = [255, 0, 0]
        
    img1_segemented = inferer.segmenter_cam1.load_and_preprocess_image(img1)
    if transformed_cutouts1 is not None:
        img1_segemented = _combine(img1_segemented, transformed_cutouts1[0], [255,255,255], size=3)
        img1_segemented = _combine(img1_segemented, transformed_cutouts1[1], color1, size=7)

    img2_segemented = inferer.segmenter_cam2.load_and_preprocess_image(img2)
    if transformed_cutouts2 is not None:
        img2_segemented = _combine(img2_segemented, transformed_cutouts2[0], [255,255,255], size=3)
        img2_segemented = _combine(img2_segemented, transformed_cutouts2[1], color2, size=7)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    axs[0].imshow(img1_segemented)
    axs[0].set_title('Camera 1')
    axs[1].imshow(img2_segemented)
    axs[1].set_title('Camera 2')

    for ax in axs:
        ax.axis('off')

    # text below the subplots
    fig.text(0.04, 0.01, f'Step: {best_id}\nScore: {best_score:.4f}', ha='left', va='bottom', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 1.2])  # leave room for bottom text
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    if display_plt:
        plt.show()

    if output_np_array:
        return _fig_to_image(fig)
    
    plt.close()
    

# Example:
# python3 lego_visualize.py $task

# outputs video sideviewfail_$task.mp4

if __name__ == "__main__":
    TASK = sys.argv[1] # cliff, R, faucet, etc.
    temp_video_dir = './temp_online_inference/video'

    ################
    
    print('Initializing inferer...')
    SIM_DATA_ROOT = "outputs" # Root dir where sim_cam1/, sim_cam2/ exist
    SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt" # Path to your SAM2 model
    SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"    # Path to your SAM2 config

    inferer = lego_online_infer.OnlineLegoInferer(
        sim_data_root_dir=SIM_DATA_ROOT,
        sam2_checkpoint_path=SAM2_CHECKPOINT,
        sam2_model_config_path=SAM2_CONFIG,
        device="cuda" # Use "cuda" if available and desired
    )

    assembly_key = TASK
    live_image_dir1 = Path(SIM_DATA_ROOT, "cam1", assembly_key)
    live_image_dir2 = Path(SIM_DATA_ROOT, "cam2", assembly_key)

    # open the live image folders and count the number of images
    live_images_cam1 = list(live_image_dir1.glob("*.jpg"))
    live_images_cam2 = list(live_image_dir2.glob("*.jpg"))
    length = min(len(live_images_cam1), len(live_images_cam2))  # Ensure both cameras have the same number of images

    ################

    inferer.cleanup_all_temp_dirs()
    os.makedirs(temp_video_dir, exist_ok=True)

    for i in tqdm(range(length), total=length):
        img1_name = f'{i:06d}.jpg'
        img2_name = f'{i:06d}.jpg'
        live_image_cam1_np = cv2.imread(str(live_image_dir1 / img1_name), cv2.IMREAD_COLOR)
        live_image_cam2_np = cv2.imread(str(live_image_dir2 / img2_name), cv2.IMREAD_COLOR)

        if live_image_cam1_np is None or live_image_cam2_np is None:
            print(f"Error: Failed to read one of the live images for assembly key '{assembly_key}'.")
            continue
        live_image_cam1_np = cv2.cvtColor(live_image_cam1_np, cv2.COLOR_BGR2RGB)  # Convert to RGB
        live_image_cam2_np = cv2.cvtColor(live_image_cam2_np, cv2.COLOR_BGR2RGB)

        results = inferer.infer_dual_camera(
            live_image_cam1_np,
            live_image_cam2_np,
            assembly_key
        )

        visualize(inferer, live_image_cam1_np, live_image_cam2_np, results,
                save_path=os.getcwd()+f'/temp_online_inference/video/{i:06d}.png', display_plt=False)

    subprocess.call(f'ffmpeg -framerate 30 -i ./temp_online_inference/video/%06d.png -c:v libx264 -r 30 -pix_fmt yuv420p sideviewfail_{TASK}.mp4', shell=True, text=True)