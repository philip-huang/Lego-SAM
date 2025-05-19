import argparse
import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from supervision.draw.color import ColorPalette
from utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

"""
Hyper parameters
"""
parser = argparse.ArgumentParser()
parser.add_argument('--grounding-model', default="IDEA-Research/grounding-dino-base")
parser.add_argument("--text-prompt", default="assembled lego structure. ")
parser.add_argument("--img-path", default="failure_images/cam1/camera1_0031_20250320_180547.png")
parser.add_argument("--img-folder", action="store_true")
parser.add_argument("--camera-name", default="cam1")
parser.add_argument("--sam2-checkpoint", default="./checkpoints/sam2.1_hiera_large.pt")
parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
parser.add_argument("--output-dir", default="outputs/test_sam2.1")
parser.add_argument("--no-dump-json", action="store_true")
parser.add_argument("--force-cpu", action="store_true")
args = parser.parse_args()

GROUNDING_MODEL = args.grounding_model
TEXT_PROMPT = args.text_prompt
IMG_PATH = args.img_path
CAMERA_NAME = args.camera_name
IMG_FOLDER = args.img_folder
SAM2_CHECKPOINT = args.sam2_checkpoint
SAM2_MODEL_CONFIG = args.sam2_model_config
DEVICE = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
OUTPUT_DIR = Path(args.output_dir)
if IMG_FOLDER:
    OUTPUT_DIR = Path("outputs") / CAMERA_NAME
DUMP_JSON_RESULTS = not args.no_dump_json

# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# environment settings
# use bfloat16
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)


# build grounding dino from huggingface
model_id = GROUNDING_MODEL
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)


# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = TEXT_PROMPT

def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def save_cutouts(img, masks, input_boxes, img_output_dir, out_img_name):
    """Saves cutouts based on masks to the specified directory."""
    # Ensure masks is a numpy array and has the expected dimensions (n, H, W)
    if isinstance(masks, torch.Tensor):
        masks_np = masks.cpu().numpy()
    else:
        masks_np = masks

    # Ensure masks_np is 3D (n, H, W) even if n=1
    if masks_np.ndim == 2:
        masks_np = np.expand_dims(masks_np, axis=0)

    for i, mask in enumerate(masks_np):
        # Ensure mask is boolean
        bool_mask = mask.astype(bool)

        # Create a 4-channel image (BGRA) for transparency
        cutout = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)

        # Copy original image pixels where the mask is True
        cutout[bool_mask, :3] = img[bool_mask]

        # Set alpha channel to 255 (opaque) where the mask is True
        cutout[bool_mask, 3] = 255

        # # crop the image to the input_box and only save that part
        # x1, y1, x2, y2 = input_boxes[i]
        # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # cutout = cutout[y1:y2, x1:x2]

        # Save the cutout
        cutout_filename = os.path.join(img_output_dir, f"cutout_{out_img_name}_mask_{i}.png") # Use PNG for transparency
        cv2.imwrite(cutout_filename, cutout)
        print(f"Saved cutout to {cutout_filename}")


def process_image(img_path, out_img_name="image"):
    print(f"Processing image: {img_path}")
    
    # Create a subdirectory in the output dir for this specific image
    img_name = Path(img_path).stem
    img_output_dir = OUTPUT_DIR
    img_output_dir.mkdir(exist_ok=True)
    
    image = Image.open(img_path)

    # crop the image to input_box first
    if CAMERA_NAME == "cam1":
        crop_size = np.array([[1350, 600, 2000, 1250]], dtype=np.int32)  # (1, 4)
    if CAMERA_NAME == "cam2":
        crop_size = np.array([[1300, 700, 1950, 1350]], dtype=np.int32)  # (1, 4)
    if CAMERA_NAME == "sim_cam1":
        crop_size = np.array([[1400, 500, 2050, 1150]], dtype=np.int32)
    if CAMERA_NAME == "sim_cam2":
        crop_size = np.array([[1550, 550, 2200, 1200]], dtype=np.int32)
    
    image = image.crop((crop_size[0][0], crop_size[0][1], crop_size[0][2], crop_size[0][3]))
    image = image.resize((crop_size[0][2] - crop_size[0][0], crop_size[0][3] - crop_size[0][1]))
    image = image.convert("RGB")

    sam2_predictor.set_image(np.array(image))

    inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.3,
        text_threshold=0.2,
        target_sizes=[image.size[::-1]]
    )

    # Skip if no detections
    if len(results[0]["boxes"]) == 0:
        print(f"No objects detected in {img_path}")
        return

    # sort the detected boxes by confidence and size of the box 
    # by computing an adjusted score based on area and confidence
    # the adjust score scales with confidence, but maxed out when area is 0.25 of total image area and decreases both ways
    adjusted_scores = []
    image_area = image.size[0] * image.size[1]
    for box, score in zip(results[0]["boxes"].cpu(), results[0]["scores"].cpu()):
        x1, y1, x2, y2 = box
        box_area = (x2 - x1) * (y2 - y1)
        area_ratio = box_area / image_area
        adjusted_score = score * (1 - abs(area_ratio - 0.25))
        adjusted_scores.append(adjusted_score)
    # sort the top k boxes by adjusted score and adjust the results
    top_k = 1
    top_k_indices = np.argsort(adjusted_scores)[-top_k:]
    top_k_boxes = [results[0]["boxes"][i] for i in top_k_indices]
    top_k_scores = [results[0]["scores"][i] for i in top_k_indices]
    top_k_labels = [results[0]["labels"][i] for i in top_k_indices]
    results[0]["boxes"] = torch.stack(top_k_boxes)
    results[0]["scores"] = torch.tensor(top_k_scores)
    results[0]["labels"] = top_k_labels
    

    # get the box prompt for SAM 2
    input_boxes = results[0]["boxes"].cpu().numpy()

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    confidences = results[0]["scores"].cpu().numpy().tolist()
    class_names = results[0]["labels"]
    class_ids = np.array(list(range(len(class_names))))

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]

    # Visualize image with supervision useful API
    img = cv2.imread(img_path)
    # crop the img
    img = img[crop_size[0][1]:crop_size[0][3], crop_size[0][0]:crop_size[0][2]]
    detections = sv.Detections(
        xyxy=input_boxes,  # (n, 4)
        mask=masks.astype(bool),  # (n, h, w)
        class_id=class_ids
    )

    box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    cv2.imwrite(os.path.join(img_output_dir, f"sam_prompt_{out_img_name}.jpg"), annotated_frame)

    # mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    # annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    # cv2.imwrite(os.path.join(img_output_dir, f"grounded_sam2_annotated_{out_img_name}.jpg"), annotated_frame)

    save_cutouts(img, masks, input_boxes, img_output_dir, out_img_name)

    if DUMP_JSON_RESULTS:
        # convert mask into rle format
        mask_rles = [single_mask_to_rle(mask) for mask in masks]

        input_boxes_list = input_boxes.tolist()
        scores_list = scores.tolist()
        
        # save the results in standard format
        json_results = {
            "image_path": img_path,
            "annotations": [
                {
                    "bbox": box,
                    "segmentation": mask_rle,
                    "score": score,
                }
                for box, mask_rle, score in zip(input_boxes_list, mask_rles, scores_list)
            ],
            "box_format": "xyxy",
            "img_width": image.width,
            "img_height": image.height,
        }
        
        with open(os.path.join(img_output_dir, "results.json"), "w") as f:
            json.dump(json_results, f, indent=4)

# Process either a single image or all images in a folder
if IMG_FOLDER:
    # Get a list of supported image file extensions
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    # Process all images in the folder
    img_folder_path = Path("failure_images") / CAMERA_NAME
    img_paths = [str(f) for f in sorted(img_folder_path.iterdir()) 
                if f.is_file() and f.suffix.lower() in supported_extensions]
    
    if not img_paths:
        print(f"No supported images found in folder: {IMG_FOLDER}")
    else:
        print(f"Found {len(img_paths)} images to process")
        for img_path in img_paths:
            process_image(img_path, out_img_name=Path(img_path).stem)
        
        print(f"All images processed. Results saved to {OUTPUT_DIR}")
else:
    # Process single image
    process_image(IMG_PATH)