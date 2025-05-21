import os
import glob
import json
import cv2
import numpy as np
from pathlib import Path
import time
import shutil
from PIL import Image # For saving numpy array to pass to segmenter

# Assuming lego_segmenter.py is in the Python path or same directory
from lego_segmenter import LegoSegmenter
# For a real setup, you'd import these from lego_match_sim.py:
from lego_match_sim import calculate_mask_iou, extract_sim_id_from_filename

class OnlineLegoInferer:
    """
    Handles online inference for Lego assembly state detection using dual cameras.
    It segments live images, compares them with simulation masks via IoU,
    and determines the best matching simulation state.
    """

    def __init__(self,
                 sim_data_root_dir: str,
                 sam2_checkpoint_path: str,
                 sam2_model_config_path: str,
                 temp_base_dir: str = "temp_online_inference",
                 device: str = "cuda",
                 dino_box_threshold: float = 0.3,
                 dino_text_threshold: float = 0.2,
                 segmenter_text_prompt: str = "assembled lego structure."
                 ):
        self.sim_data_root_dir = Path(sim_data_root_dir)
        self.temp_base_dir = Path(temp_base_dir)
        self.device = device
        self.IMAGE_EXTENSIONS = ('.png',)

        shutil.rmtree(self.temp_base_dir, ignore_errors=True) # Clean up from previous runs
        self.temp_base_dir.mkdir(parents=True, exist_ok=True)
        
        #self.temp_input_image_dir = self.temp_base_dir / "live_inputs"
        #self.temp_input_image_dir.mkdir(parents=True, exist_ok=True)

        segmenter_cam1_output_dir = self.temp_base_dir / "segmenter_cam1_out"
        segmenter_cam1_output_dir.mkdir(parents=True, exist_ok=True)
        self.segmenter_cam1 = LegoSegmenter(
            sam2_checkpoint_path=sam2_checkpoint_path,
            sam2_model_config_path=sam2_model_config_path,
            camera_name="cam1",
            output_dir=str(segmenter_cam1_output_dir),
            device=device,
            dump_json_results=False,
            dino_box_threshold=dino_box_threshold,
            dino_text_threshold=dino_text_threshold,
            default_text_prompt=segmenter_text_prompt
        )

        segmenter_cam2_output_dir = self.temp_base_dir / "segmenter_cam2_out"
        segmenter_cam2_output_dir.mkdir(parents=True, exist_ok=True)
        self.segmenter_cam2 = LegoSegmenter(
            sam2_checkpoint_path=sam2_checkpoint_path,
            sam2_model_config_path=sam2_model_config_path,
            camera_name="cam2",
            output_dir=str(segmenter_cam2_output_dir),
            device=device,
            dump_json_results=False,
            dino_box_threshold=dino_box_threshold,
            dino_text_threshold=dino_text_threshold,
            default_text_prompt=segmenter_text_prompt
        )
        self.sim_mask_info_cache = {} # Cache: {assembly_key: {sim_cam_name: {sim_id: path}}}

    def _get_simulation_masks_for_key_and_cam(self, assembly_key: str, sim_camera_name: str) -> dict[int, str]:
        """
        Loads and caches paths to simulation masks for a given assembly_key and sim_camera_name.
        Returns a dictionary mapping sim_id (int) to mask_path (str).
        sim_camera_name: "sim_cam1" or "sim_cam2".
        """
        if assembly_key in self.sim_mask_info_cache and \
           sim_camera_name in self.sim_mask_info_cache[assembly_key]:
            return self.sim_mask_info_cache[assembly_key][sim_camera_name]

        sim_ref_dir = self.sim_data_root_dir / sim_camera_name / f"{assembly_key}"
        mask_paths_with_ids = {}

        if not sim_ref_dir.is_dir():
            print(f"Warning: Simulation reference directory not found: {sim_ref_dir}")
        else:
            sim_files_in_dir = []
            print(sim_ref_dir)
            for ext in self.IMAGE_EXTENSIONS:
                sim_files_in_dir.extend(list(sim_ref_dir.glob(f"cutout_*{ext}"))) # More specific glob

            for sim_path in sim_files_in_dir:
                sim_id = extract_sim_id_from_filename(sim_path.name)
                if sim_id is not None:
                    sim_grayscale = cv2.imread(str(sim_path), cv2.IMREAD_GRAYSCALE)
                    mask_paths_with_ids[sim_id] = sim_grayscale
        
        if assembly_key not in self.sim_mask_info_cache:
            self.sim_mask_info_cache[assembly_key] = {}
        self.sim_mask_info_cache[assembly_key][sim_camera_name] = mask_paths_with_ids
        return mask_paths_with_ids

    def _segment_live_image(self, image_data_np: np.ndarray, camera_name: str) -> np.ndarray | None:
        """
        Segments a live image (NumPy array, RGB) using LegoSegmenter in-memory.
        Returns the primary cutout mask as a NumPy array (uint8, 0 or 255), or None.
        """
        segmenter = self.segmenter_cam1 if camera_name == "cam1" else self.segmenter_cam2
        
        mask_np = segmenter.generate_single_mask_from_data(image_data_np) # text_prompt uses segmenter's default

        if mask_np is None:
            # print(f"Debug: Segmentation returned None for live image from {camera_name}.")
            pass # Error already printed by LegoSegmenter or this method earlier
        
        return mask_np

    def infer_dual_camera(self,
                          live_image_cam1_np: np.ndarray, # Expected RGB
                          live_image_cam2_np: np.ndarray, # Expected RGB
                          assembly_key: str
                          ) -> tuple[int | None, float, dict]:
        """
        Performs inference using two live camera images against simulation masks.

        Args:
            live_image_cam1_np: NumPy array for camera 1 image (RGB).
            live_image_cam2_np: NumPy array for camera 2 image (RGB).
            assembly_key: The key for the current assembly stage (e.g., "S").

        Returns:
            Tuple: (best_overall_sim_id, max_combined_iou_score, per_sim_id_details)
            - best_overall_sim_id (int | None): The sim_id with the highest combined IoU.
            - max_combined_iou_score (float): The score itself.
            - per_sim_id_details (dict): Detailed IoUs for each sim_id.
              { sim_id: {'cam1_iou': float, 'cam2_iou': float, 'combined_iou': float,
                         'cam1_sim_mask': str|None, 'cam2_sim_mask': str|None,
                         'live_cutout_cam1': str|None, 'live_cutout_cam2': str|None }, ... }
        """
        live_image_unique_id = f"{assembly_key}_{(time.time_ns()//100000000):d}"

        live_cutout_cam1_data = self._segment_live_image(live_image_cam1_np, "cam1")
        live_cutout_cam2_data = self._segment_live_image(live_image_cam2_np, "cam2")        

        if live_cutout_cam1_data is None and live_cutout_cam2_data is None:
            print(f"Error: Segmentation failed or no objects found for both cameras for assembly key '{assembly_key}'.")
            return None, None, None, -1.0, {}

        # Save the segmented cutouts to temporary files (image is rgb)
        live_cutout_cam1_path = self.temp_base_dir / f"live_cutout_{live_image_unique_id}_cam1.png"
        live_cutout_cam2_path = self.temp_base_dir / f"live_cutout_{live_image_unique_id}_cam2.png"
        bgr_cutout_cam1_data = cv2.cvtColor(live_cutout_cam1_data, cv2.COLOR_RGB2BGR)
        bgr_cutout_cam2_data = cv2.cvtColor(live_cutout_cam2_data, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(live_cutout_cam2_path), bgr_cutout_cam2_data)
        cv2.imwrite(str(live_cutout_cam1_path), bgr_cutout_cam1_data)

        sim_masks_cam1_map = self._get_simulation_masks_for_key_and_cam(assembly_key, "sim_cam1")
        sim_masks_cam2_map = self._get_simulation_masks_for_key_and_cam(assembly_key, "sim_cam2")

        if not sim_masks_cam1_map and not sim_masks_cam2_map:
            print(f"Warning: No simulation masks found for assembly key '{assembly_key}' for any camera.")
            return None, None, None, -1.0, {}

        unique_sim_ids = set(sim_masks_cam1_map.keys()).union(set(sim_masks_cam2_map.keys()))
        if not unique_sim_ids:
            print(f"Warning: No sim_ids could be extracted for assembly key '{assembly_key}'.")
            return None, None, None, -1.0, {}

        best_overall_sim_id = None
        max_combined_iou = -1.0
        all_sim_id_results = {}

        for sim_id in unique_sim_ids:
            iou_cam1 = 0.0
            sim_mask_p1 = sim_masks_cam1_map.get(sim_id)
            if sim_mask_p1 is not None: # Ensure sim exist
                # live_cutout_cam1_data can be None, _calculate_mask_iou handles it
                # convert to grayscale for IoU calculation
                live_cutout_cam1_data_gray = cv2.cvtColor(live_cutout_cam1_data, cv2.COLOR_RGB2GRAY)
                iou_cam1 = calculate_mask_iou(live_cutout_cam1_data_gray, sim_mask_p1)

            iou_cam2 = 0.0
            sim_mask_p2 = sim_masks_cam2_map.get(sim_id)
            if sim_mask_p2 is not None: # Ensure sim exist
                # live_cutout_cam2_data can be None, _calculate_mask_iou handles it
                live_cutout_cam2_data_gray = cv2.cvtColor(live_cutout_cam2_data, cv2.COLOR_RGB2GRAY)
                iou_cam2 = calculate_mask_iou(live_cutout_cam2_data_gray, sim_mask_p2)
            
            num_valid_ious = 0
            current_sum_iou = 0.0
            # A "valid" IoU for averaging means a comparison was attempted and yielded a score.
            # If a live mask is None, its IoU is 0. We count it if a sim mask existed for that view.
            if sim_mask_p1 is not None: 
                num_valid_ious += 1
                current_sum_iou += iou_cam1
            if sim_mask_p2 is not None:
                num_valid_ious += 1
                current_sum_iou += iou_cam2
            
            combined_iou = (current_sum_iou / num_valid_ious) if num_valid_ious > 0 else 0.0

            all_sim_id_results[sim_id] = {
                'cam1_iou': iou_cam1, # Will be 0 if live_cutout_cam1_data was None or sim_mask_p1 was None
                'cam2_iou': iou_cam2, # Will be 0 if live_cutout_cam2_data was None or sim_mask_p2 was None
                'combined_iou': combined_iou,
                'cam1_sim_mask': sim_mask_p1,
                'cam2_sim_mask': sim_mask_p2
            }

            if combined_iou > max_combined_iou:
                max_combined_iou = combined_iou
                best_overall_sim_id = sim_id

        return live_cutout_cam1_data, live_cutout_cam2_data, best_overall_sim_id, max_combined_iou, all_sim_id_results

    def cleanup_all_temp_dirs(self):
        """Cleans up the base temporary directory used by this instance."""
        print(f"Cleaning up base temporary directory: {self.temp_base_dir}")
        shutil.rmtree(self.temp_base_dir, ignore_errors=True)

# Example Usage (Illustrative - not for ROS yet):
if __name__ == "__main__":
    # --- Configuration ---
    # Adjust these paths based on your project structure
    SIM_DATA_ROOT = "outputs" # Root dir where sim_cam1/, sim_cam2/ exist
    SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt" # Path to your SAM2 model
    SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"    # Path to your SAM2 config
    
    # Create dummy simulation masks and live images for testing
    # This setup is simplified. Your actual sim masks would be more complex.
    dummy_assembly_key = "test_assembly"
    Path(SIM_DATA_ROOT, "sim_cam1", f"cutout_{dummy_assembly_key}").mkdir(parents=True, exist_ok=True)
    Path(SIM_DATA_ROOT, "sim_cam2", f"cutout_{dummy_assembly_key}").mkdir(parents=True, exist_ok=True)

    # Create dummy 100x100 black images with a white square for sim masks
    for i in range(1, 3): # sim_id 1 and 2
        for cam_idx in [1, 2]:
            sim_mask = np.zeros((100, 100), dtype=np.uint8)
            sim_mask[20:80, 20:20+(i*20)] = 255 # Different width for sim_id 1 vs 2
            cv2.imwrite(
                str(Path(SIM_DATA_ROOT, f"sim_cam{cam_idx}", f"cutout_{dummy_assembly_key}", f"cutout_000{i}_simcam{cam_idx}_mask_0.png")),
                sim_mask
            )
    
    # Create dummy live images (e.g., one that matches sim_id 0001 better)
    live_img_cam1_np = np.zeros((480, 640, 3), dtype=np.uint8) # Dummy RGB
    live_img_cam1_np[100:200, 100:150, :] = 128 # Some content for segmenter
    
    live_img_cam2_np = np.zeros((480, 640, 3), dtype=np.uint8) # Dummy RGB
    live_img_cam2_np[150:250, 150:200, :] = 100

    # Initialize inferer
    inferer = OnlineLegoInferer(
        sim_data_root_dir=SIM_DATA_ROOT,
        sam2_checkpoint_path=SAM2_CHECKPOINT,
        sam2_model_config_path=SAM2_CONFIG,
        device="cpu" # Use "cuda" if available and desired
    )

    print(f"Starting inference for assembly key: {dummy_assembly_key}")
    best_id, best_score, details = inferer.infer_dual_camera(
        live_image_cam1_np,
        live_image_cam2_np,
        dummy_assembly_key
    )

    print(f"\n--- Inference Results ---")
    print(f"Best Matching Sim ID: {best_id}")
    print(f"Best Combined IoU Score: {best_score:.4f}")
    print("\nDetails per Sim ID:")
    for sim_id, data in details.items():
        print(f"  Sim ID {sim_id}:")
        print(f"    Cam1 IoU: {data['cam1_iou']:.4f} (SimMask: {Path(data['cam1_sim_mask']).name if data['cam1_sim_mask'] else 'N/A'}, LiveCutout: {Path(data['live_cutout_cam1']).name if data['live_cutout_cam1'] else 'N/A'})")
        print(f"    Cam2 IoU: {data['cam2_iou']:.4f} (SimMask: {Path(data['cam2_sim_mask']).name if data['cam2_sim_mask'] else 'N/A'}, LiveCutout: {Path(data['live_cutout_cam2']).name if data['live_cutout_cam2'] else 'N/A'})")
        print(f"    Combined IoU: {data['combined_iou']:.4f}")

    # Clean up all temporary files and directories created by the inferer
    inferer.cleanup_all_temp_dirs()
    # Clean up dummy sim files
    shutil.rmtree(Path(SIM_DATA_ROOT, "sim_cam1", f"cutout_{dummy_assembly_key}"), ignore_errors=True)
    shutil.rmtree(Path(SIM_DATA_ROOT, "sim_cam2", f"cutout_{dummy_assembly_key}"), ignore_errors=True)
    print("Dummy files cleaned up.")