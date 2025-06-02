"""
Python class for segmenting lego structures using Grounding DINO and SAM2.
Works with static images or callable as API
"""
import argparse
import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
import time
from pathlib import Path
from supervision.draw.color import ColorPalette
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Placeholder for utils.supervision_utils.CUSTOM_COLOR_MAP
# This should be a list of hex color strings, ensure it's defined if used from utils
try:
    from utils.supervision_utils import CUSTOM_COLOR_MAP
except ImportError:
    CUSTOM_COLOR_MAP = [
        "#FF5733", "#33FF57", "#3357FF", "#FF33A1", "#A133FF",
        "#33FFA1", "#FF8C33", "#33FF8C", "#8C33FF", "#FF3333"
    ]


class LegoSegmenter:
    def __init__(self,
                 grounding_dino_model_id="IDEA-Research/grounding-dino-base",
                 sam2_checkpoint_path="./checkpoints/sam2.1_hiera_large.pt",
                 sam2_model_config_path="configs/sam2.1/sam2.1_hiera_l.yaml",
                 default_text_prompt="assembled lego structure.",
                 camera_name="cam1",
                 output_dir="outputs/lego_segmentation",
                 device="cuda",
                 dump_json_results=True,
                 top_k_detections=1,
                 dino_box_threshold=0.3,
                 dino_text_threshold=0.2):

        self.grounding_dino_model_id = grounding_dino_model_id
        self.sam2_checkpoint_path = sam2_checkpoint_path
        self.sam2_model_config_path = sam2_model_config_path
        self.default_text_prompt = default_text_prompt
        self.camera_name = camera_name
        self.output_dir = Path(output_dir)
        self.device = device
        self.dump_json_results = dump_json_results
        self.top_k_detections = top_k_detections
        self.dino_box_threshold = dino_box_threshold
        self.dino_text_threshold = dino_text_threshold

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Environment settings
        if self.device == "cuda":
            torch.autocast(device_type=self.device, dtype=torch.bfloat16).__enter__()
            if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        # Build SAM2 image predictor
        sam2_model = build_sam2(self.sam2_model_config_path, self.sam2_checkpoint_path, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        # Build Grounding DINO from Hugging Face
        self.dino_processor = AutoProcessor.from_pretrained(self.grounding_dino_model_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.grounding_dino_model_id).to(self.device)

        # Crop settings
        self.crop_box_dict = {
            "cam1": (1350, 550, 2000, 1200),  # (x1, y1, x2, y2)
            "cam2": (1600, 600, 2250, 1250),
            "sim_cam1": (1400, 500, 2050, 1150),
            "sim_cam2": (1550, 500, 2200, 1150),
            "default": None
        }
        # self.crop_box_dict = {
        #     "cam1": (2050, 900, 2600, 1550),  # (x1, y1, x2, y2)
        #     "cam2": (1150, 1000, 1700, 1600),
        #     "sim_cam1": (2200, 900, 2750, 1550),
        #     "sim_cam2": (1150, 900, 1700, 1550),
        #     "default": None
        # }
        self.crop_box = self.crop_box_dict.get(self.camera_name, self.crop_box_dict["default"])

        # Annotators
        self.box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        self.label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        # self.mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))


    def _prepare_text_prompt(self, text_prompt):
        processed_prompt = text_prompt.lower()
        if not processed_prompt.endswith("."):
            processed_prompt += "."
        return processed_prompt

    def load_and_preprocess_image(self, image_source):
        """
        Loads an image from a path, PIL image, or NumPy array, and preprocesses it.
        Preprocessing includes cropping based on camera_name.
        Returns a PIL Image object in RGB format.
        """
        if isinstance(image_source, str): # Path
            image = Image.open(image_source)
        elif isinstance(image_source, Image.Image): # PIL Image
            image = image_source
        elif isinstance(image_source, np.ndarray): # NumPy array
            image = Image.fromarray(image_source)
        else:
            raise ValueError("image_source must be a path string, PIL Image, or NumPy array.")

        if self.crop_box:
            image = image.crop(self.crop_box)
        
        return image.convert("RGB")

    def detect_with_grounding_dino(self, image_pil, text_prompt=None):
        """
        Performs object detection using Grounding DINO.
        Args:
            image_pil (PIL.Image): Preprocessed PIL Image.
            text_prompt (str, optional): Text prompt for detection. Uses default if None.
        Returns:
            tuple: (boxes_xyxy, scores, labels) or (None, None, None) if no detections.
                   boxes_xyxy is a torch.Tensor.
        """
        if text_prompt is None:
            text_prompt = self.default_text_prompt
        processed_text_prompt = self._prepare_text_prompt(text_prompt)

        inputs = self.dino_processor(images=image_pil, text=processed_text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)

        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.dino_box_threshold,
            text_threshold=self.dino_text_threshold,
            target_sizes=[image_pil.size[::-1]] # (height, width)
        )

        if not results or len(results[0]["boxes"]) == 0:
            print("No detections found.")
            return None, None, None

        # Sort by adjusted score (confidence * area factor) and select top_k
        adjusted_scores = []
        image_area = image_pil.size[0] * image_pil.size[1]
        for box, score in zip(results[0]["boxes"].cpu(), results[0]["scores"].cpu()):
            x1, y1, x2, y2 = box
            box_area = (x2 - x1) * (y2 - y1)
            area_ratio = box_area / image_area
            # Score peaks when area_ratio is 0.2, decreases otherwise
            area_factor = 1 - abs(area_ratio - 0.2) if image_area > 0 else 1.0
            adjusted_score = score * area_factor
            adjusted_scores.append(adjusted_score)
        # sort the top k boxes by adjusted score and adjust the results
        top_k = 1
        top_k_indices = np.argsort(adjusted_scores)[-top_k:]
        top_k_boxes = [results[0]["boxes"][i] for i in top_k_indices]
        top_k_scores = [results[0]["scores"][i] for i in top_k_indices]
        top_k_labels = [results[0]["labels"][i] for i in top_k_indices]
        dino_boxes = torch.stack(top_k_boxes)
        dino_score = torch.tensor(top_k_scores)
        dino_labels = top_k_labels
         
        
        return dino_boxes, dino_score, dino_labels


    def segment_with_sam2(self, image_pil, input_boxes_xyxy):
        """
        Performs segmentation using SAM2 based on input boxes.
        Args:
            image_pil (PIL.Image): Preprocessed PIL Image.
            input_boxes_xyxy (np.ndarray or torch.Tensor): Bounding boxes in (N, 4) xyxy format.
        Returns:
            tuple: (masks, scores)
                   masks is a torch.Tensor.
        """
        self.sam2_predictor.set_image(np.array(image_pil))
        
        if isinstance(input_boxes_xyxy, torch.Tensor):
            input_boxes_np = input_boxes_xyxy.cpu().numpy()
        else:
            input_boxes_np = input_boxes_xyxy

        masks, scores, _ = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes_np,
            multimask_output=False,
        )
        # Ensure masks is (N, H, W)
        if masks.ndim == 4 and masks.shape[1] == 1: # (N, 1, H, W)
            masks = masks.squeeze(1)
        return masks, scores
    
    def remove_background(self, img_cv_cropped, mask):
        """
        remove all non-red masks with hsv color space
        
        """
        bool_mask = mask[0].astype(bool)
        cutout = np.zeros((img_cv_cropped.shape[0], img_cv_cropped.shape[1], 4), dtype=np.uint8)
        cutout[bool_mask, :3] = img_cv_cropped[bool_mask]
        cutout[bool_mask, 3] = 255 # Alpha channel

        # Convert to HSV color space
        hsv = cv2.cvtColor(cutout, cv2.COLOR_BGR2HSV)
        # Define the range for red color in HSV
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        # Create a mask for red color
        red_mask = cv2.inRange(hsv, lower_red, upper_red)
        # update the mask
        mask[0][red_mask == 0] = 0
        return mask

    def generate_single_mask_from_data(self, image_data_np: np.ndarray, text_prompt: str = None, save_id: int = None): # -> np.ndarray | None:
        """
        Processes image data (NumPy array) in memory to get a single segmentation mask.
        Returns a 2D NumPy array (uint8, 0 or 255) for the mask, or None.
        The mask is relative to the (potentially) camera-cropped image.
        Assumes self.top_k_detections is effectively 1 for DINO part.
        """
        # 1. Load and preprocess image (handles camera crop if configured)
        image_pil = self.load_and_preprocess_image(image_data_np) # Returns PIL RGB

        # 2. Detect objects with Grounding DINO
        # The detect_with_grounding_dino method already implements top_k=1 logic internally.
        dino_boxes, dino_scores, dino_labels = self.detect_with_grounding_dino(image_pil, text_prompt)

        if dino_boxes is None or len(dino_boxes) == 0:
            print(f"LegoSegmenter: No objects detected by DINO for in-memory image ({self.camera_name}).")
            return None

        # 3. Segment objects with SAM2
        # sam_masks_tensor is (N, H, W) tensor. For top_k=1 from DINO, N should be 1.
        sam_masks_tensor, sam_scores = self.segment_with_sam2(image_pil, dino_boxes)

        if sam_masks_tensor is None or sam_masks_tensor.shape[0] == 0:
            print(f"LegoSegmenter: SAM2 did not produce any masks ({self.camera_name}).")
            return None

        # Assuming DINO returned one box and SAM one mask for it
        primary_mask_tensor = sam_masks_tensor[0] # Shape (H, W)
        
        # Convert to NumPy array
        primary_mask_np_bool =  primary_mask_tensor.astype(bool) # Boolean mask for indexing

        # Convert PIL image to NumPy array to apply the mask
        image_np_rgb = np.array(image_pil) # image_pil is RGB
        
        # Create RGBA cutout
        # image_pil.size is (width, height), NumPy shape is (height, width, channels)
        cutout_rgba = np.zeros((image_pil.size[1], image_pil.size[0], 4), dtype=np.uint8)
        
        # Apply mask to RGB channels
        cutout_rgba[primary_mask_np_bool, :3] = image_np_rgb[primary_mask_np_bool]
        
        # Set alpha channel based on mask
        cutout_rgba[primary_mask_np_bool, 3] = 255

        if save_id is not None:
            image_np_bgr = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR)
            save_name = f'{save_id:06d}'
            self.save_dino_boxes(dino_boxes, sam_masks_tensor, dino_scores, dino_labels, image_np_bgr, save_name)
            
        return cutout_rgba

    def _single_mask_to_rle(self, mask_tensor):
        mask_np = mask_tensor.astype(np.uint8)
        rle = mask_util.encode(np.array(mask_np[:, :, None], order="F"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    def _save_cutouts(self, original_image_cv_cropped, masks_np, output_subdir, base_name):
        if masks_np.ndim == 2: # Single mask
            masks_np = np.expand_dims(masks_np, axis=0)

        for i, mask_H_W in enumerate(masks_np):
            bool_mask = mask_H_W.astype(bool)
            cutout = np.zeros((original_image_cv_cropped.shape[0], original_image_cv_cropped.shape[1], 4), dtype=np.uint8)
            cutout[bool_mask, :3] = original_image_cv_cropped[bool_mask]
            cutout[bool_mask, 3] = 255 # Alpha channel

            cutout_filename = output_subdir / f"cutout_{base_name}_mask_{i}.png"
            cv2.imwrite(str(cutout_filename), cutout)
            print(f"Saved cutout to {cutout_filename}")

    def save_dino_boxes(self, dino_boxes, sam_masks, dino_scores, dino_labels, img_cv_cropped, output_image_name):
        detections = sv.Detections(
            xyxy=dino_boxes.cpu().numpy(),
            mask=sam_masks.astype(bool),
            class_id=np.arange(len(dino_labels)), # Use index as class_id for coloring
            confidence=dino_scores.cpu().numpy() # Store DINO scores with detections
        )

        # Create labels for visualization
        viz_labels = [
            f"{conf:.2f}"
            for name, conf in zip(dino_labels, dino_scores.cpu().numpy())
        ]

        annotated_frame = self.box_annotator.annotate(scene=img_cv_cropped.copy(), detections=detections)
        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=viz_labels)
        
        # Save annotated DINO prompts for SAM
        cv2.imwrite(str(self.output_dir / f"sam_prompt_{output_image_name}.jpg"), annotated_frame)

    def process_image_full_pipeline(self, image_path, output_image_name=None):
        """
        Processes a single image through the full pipeline: load, preprocess, detect, segment, visualize, and save.
        """
        if output_image_name is None:
            output_image_name = Path(image_path).stem
        
        print(f"Processing image: {image_path}")
        
        # Create a subdirectory in the output_dir for this specific image's artifacts
        # This behavior might differ from original script's output structure, adjust if needed.
        # For now, all outputs for one image go into self.output_dir directly or a sub-folder named after image.
        # Let's put them in self.output_dir for simplicity matching single image case.
        # If multiple images, main can create subdirs.
        current_output_dir = self.output_dir
        current_output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load and preprocess image for models
        image_pil = self.load_and_preprocess_image(image_path)

        # 2. Detect objects with Grounding DINO
        dino_boxes, dino_scores, dino_labels = self.detect_with_grounding_dino(image_pil)

        if dino_boxes is None or len(dino_boxes) == 0:
            print(f"No objects detected in {image_path}")
            # Save the (cropped) input image for reference if no detections
            img_cv_original = cv2.imread(str(image_path))
            if self.crop_box:
                x1, y1, x2, y2 = self.crop_box
                img_cv_cropped = img_cv_original[y1:y2, x1:x2]
            else:
                img_cv_cropped = img_cv_original
            cv2.imwrite(str(current_output_dir / f"no_detection_{output_image_name}.jpg"), img_cv_cropped)
            return

        # 3. Segment objects with SAM2
        sam_masks, sam_scores = self.segment_with_sam2(image_pil, dino_boxes)
        

        # 4. Visualization and Saving
        # Load original image for visualization (and crop it)
        img_cv_original = cv2.imread(str(image_path))
        if self.crop_box:
            x1, y1, x2, y2 = self.crop_box
            img_cv_cropped = img_cv_original[y1:y2, x1:x2].copy() # Use copy for annotation
        else:
            img_cv_cropped = img_cv_original.copy()

        # remove background
        if self.camera_name == "sim_cam1" or self.camera_name == "sim_cam2":
            sam_masks = self.remove_background(img_cv_cropped, sam_masks)

        self.save_dino_boxes(dino_boxes, sam_masks, dino_scores, dino_labels, img_cv_cropped, output_image_name)

        # Save cutouts
        self._save_cutouts(img_cv_cropped, sam_masks, current_output_dir, output_image_name)

        if self.dump_json_results:
            mask_rles = [self._single_mask_to_rle(mask) for mask in sam_masks]
            json_results = {
                "image_path": str(image_path),
                "annotations": [
                    {
                        "bbox": box.tolist(),
                        "segmentation": rle,
                        "dino_score": d_score.item(), # Grounding DINO score for the box
                        "sam_score": s_score.item(),  # SAM score for the mask
                        "label": label,
                    }
                    for box, rle, d_score, s_score, label in zip(
                        dino_boxes.cpu(), mask_rles, dino_scores.cpu(), sam_scores, dino_labels
                    )
                ],
                "box_format": "xyxy",
                "img_width": image_pil.width, # Cropped image width
                "img_height": image_pil.height, # Cropped image height
                "crop_box_applied": self.crop_box
            }
            with open(current_output_dir / f"results_{output_image_name}.json", "w") as f:
                json.dump(json_results, f, indent=4)
            print(f"Saved JSON results to {current_output_dir / f'results_{output_image_name}.json'}")


def main():
    parser = argparse.ArgumentParser(description="Lego Segmentation using Grounding DINO and SAM2.")
    parser.add_argument('--grounding-model', default="IDEA-Research/grounding-dino-base", help="Grounding DINO model ID.")
    parser.add_argument("--text-prompt", default="assembled lego structure in center.", help="Default text prompt for detection.")
    parser.add_argument("--img-path", default="failure_images/cam1/camera1_0031_20250320_180547.png", help="Path to a single image.")
    parser.add_argument("--img-folder", help="Path to a folder of images. Overrides --img-path.")
    parser.add_argument("--camera-name", default="cam1", help="Camera name for crop settings (e.g., cam1, sim_cam2).")
    parser.add_argument("--sam2-checkpoint", default="./checkpoints/sam2.1_hiera_large.pt", help="Path to SAM2 checkpoint.")
    parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml", help="Path to SAM2 model config.")
    parser.add_argument("--output-dir", default="outputs/lego_segmentation_results", help="Base directory for output.")
    parser.add_argument("--dump-json", action="store_true", help="Save results in JSON format.")
    parser.add_argument("--force-cpu", action="store_true", help="Force use of CPU even if CUDA is available.")
    parser.add_argument("--top-k", type=int, default=1, help="Number of top detections to process from Grounding DINO.")

    # Example argument
    # python lego_segmenter.py --img-folder "sim_images/$task/cam1" --output-dir "outputs/sim_cam1/$task" --camera-name sim_cam1
  
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    print(f"Using device: {device}")

    # Determine the final output directory path
    # If processing a folder, create a subdirectory named after the camera_name or folder name
    # If processing a single image, results go into output_dir directly or a subdir for that image.
    # The class's process_image_full_pipeline already creates a subdir for each image.
    # So, the main output_dir passed to the class is the root for all these.
    
    final_output_dir = Path(args.output_dir)
    if args.img_folder:
         # Optional: could add camera_name or folder name to path.
         # e.g. final_output_dir = Path(args.output_dir) / Path(args.img_folder).name
         # For now, class handles image-specific subdirs within the provided args.output_dir
         pass


    segmenter = LegoSegmenter(
        grounding_dino_model_id=args.grounding_model,
        sam2_checkpoint_path=args.sam2_checkpoint,
        sam2_model_config_path=args.sam2_model_config,
        default_text_prompt=args.text_prompt,
        camera_name=args.camera_name,
        output_dir=final_output_dir,
        device=device,
        dump_json_results=args.dump_json,
        top_k_detections=args.top_k
    )

    if args.img_folder:
        img_folder_path = Path(args.img_folder)
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_paths = [p for p in sorted(img_folder_path.iterdir()) if p.is_file() and p.suffix.lower() in supported_extensions]

        if not image_paths:
            print(f"No supported images found in folder: {img_folder_path}")
        else:
            print(f"Found {len(image_paths)} images to process in {img_folder_path}.")
            for img_p in image_paths:
                segmenter.process_image_full_pipeline(str(img_p)) # output_image_name will be img_p.stem
            print(f"All images processed. Results saved to {final_output_dir}")
    else:
        if not Path(args.img_path).exists():
            print(f"Error: Image path does not exist: {args.img_path}")
            return
        segmenter.process_image_full_pipeline(args.img_path)
        print(f"Image processed. Results saved to {final_output_dir}")

if __name__ == "__main__":
    main()