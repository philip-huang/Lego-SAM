import torch
import os
import json
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from torch.nn.functional import cosine_similarity
import glob
import argparse
import pandas as pd
import torchvision.transforms
from collections import Counter

# --- Configuration ---
DEFAULT_CLASS_JSON_PATH = "outputs/class.json"
DEFAULT_BASE_OUTPUT_DIR = "outputs"
DEFAULT_RESULTS_FILE = "evaluation_results.csv"
MODEL_NAME = "facebook/dinov2-with-registers-large" # Using a large model, ensure enough VRAM if on GPU
IMAGE_EXTENSIONS = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif')
SCALES_SHORTEST_EDGE = [256] # Define scales for multi-scale processing

# --- Global Variables (initialized later) ---
device = None
processor = None
model = None

# --- Helper Function to Setup Device, Model, and Processor ---
def setup_system():
    global device, processor, model
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")

    print(f"Loading model: {MODEL_NAME}...")
    try:
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME).to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

# --- Helper Function to Preprocess and Get Embedding ---
def get_image_embedding(image_path, scale_shortest_edge): # Added scale_shortest_edge parameter
    global processor, model, device
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None

    try:
        inputs = processor(images=image, return_tensors="pt", size={"shortest_edge": scale_shortest_edge}).to(device) # Use scale_shortest_edge
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        if hasattr(outputs, 'last_hidden_state'):
            num_registers = getattr(model.config, 'num_registers', 4) 
            embedding = outputs.last_hidden_state[:, 1+num_registers:].reshape(1, -1)
        elif hasattr(outputs, 'pooler_output'):
            embedding = outputs.pooler_output
        else:
            print(f"Warning: Could not determine standard embedding output for {image_path} at scale {scale_shortest_edge}.")
            return None
        return embedding
    except Exception as e:
        print(f"Error processing image {image_path} for embedding at scale {scale_shortest_edge}: {e}")
        return None


# --- Helper Function to Extract Sim ID from Filename ---
def extract_sim_id_from_filename(filename):
    # Expected format: cutout_{sim_id_formatted}_{cam_suffix}_mask_0.png
    # e.g., cutout_0088_cam1_mask_0.png
    try:
        parts = os.path.splitext(filename)[0].split('_')
        if len(parts) >= 2 and parts[0] == "cutout":
            return int(parts[1]) # The ID is the second part
    except ValueError:
        print(f"Warning: Could not parse sim ID from filename: {filename}")
    except IndexError:
        print(f"Warning: Filename format unexpected for sim ID extraction: {filename}")
    return None


# --- Main Evaluation Logic ---
def run_evaluation(keys_to_eval, class_json_path, base_output_dir, results_file_path, aggregation_method):
    global device, processor, model
    if not model:
        setup_system()

    try:
        with open(class_json_path, 'r') as f:
            class_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: class.json not found at {class_json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {class_json_path}")
        return

    all_results = []
    
    if not keys_to_eval: 
        keys_to_eval = list(class_data.keys())
    
    print(f"Starting evaluation for keys: {', '.join(keys_to_eval)}")
    print(f"Using scales (shortest edge): {SCALES_SHORTEST_EDGE}")
    print(f"Using aggregation method: {aggregation_method}")

    for key in keys_to_eval:
        if key not in class_data:
            print(f"Warning: Key '{key}' not found in class.json. Skipping.")
            continue
        
        data_for_key = class_data[key]
        # sim_ids_for_key = data_for_key.get("sim", []) # Not directly used in this revised logic for matching
        real_data_for_key = data_for_key.get("real", {})

        if not real_data_for_key:
            print(f"Warning: No 'real' data defined for key '{key}'. Skipping real processing for this key.")
            continue

        # --- 1. Precompute Simulator Reference Embeddings for this key, all scales, and both cameras ---
        sim_embeddings_this_key_cam_scale = {1: {}, 2: {}} # {cam_idx: {scale: {sim_path: embedding}}}
        print(f"\nPreprocessing simulator embeddings for key '{key}'...")
        for cam_idx in [1, 2]:
            for scale in SCALES_SHORTEST_EDGE:
                sim_embeddings_this_key_cam_scale[cam_idx][scale] = {}

            sim_cam_folder_name = f"sim_cam{cam_idx}"
            sim_ref_dir = os.path.join(base_output_dir, sim_cam_folder_name, f"cutout_{key}")

            if not os.path.isdir(sim_ref_dir):
                print(f"  Warning: Simulator reference directory not found: {sim_ref_dir}. No sim embeddings for key '{key}', cam{cam_idx}.")
                continue

            sim_files_in_dir = []
            for ext in IMAGE_EXTENSIONS:
                sim_files_in_dir.extend(glob.glob(os.path.join(sim_ref_dir, ext)))

            if not sim_files_in_dir:
                print(f"  Warning: No simulator images found in {sim_ref_dir} for key '{key}', cam{cam_idx}.")
                continue
            
            print(f"  Processing {len(sim_files_in_dir)} sim references for key '{key}', cam{cam_idx} across {len(SCALES_SHORTEST_EDGE)} scales...")
            for sim_path in sim_files_in_dir:
                for scale_val in SCALES_SHORTEST_EDGE:
                    embedding = get_image_embedding(sim_path, scale_val)
                    if embedding is not None:
                        sim_embeddings_this_key_cam_scale[cam_idx][scale_val][sim_path] = embedding
        
        key_has_any_sim_embeddings = any(
            sim_embeddings_this_key_cam_scale[c][s] 
            for c in [1,2] for s in SCALES_SHORTEST_EDGE if s in sim_embeddings_this_key_cam_scale[c]
        )
        if not key_has_any_sim_embeddings:
            print(f"Warning: No simulator embeddings could be generated for key '{key}' across any camera/scale. Skipping real processing for this key.")
            continue

        # --- 2. Process Real Images for this key, compare with sim references, and perform majority vote ---
        print(f"Processing real images for key '{key}'...")
        for real_id_str, true_sim_id_int in real_data_for_key.items():
            real_id_formatted = real_id_str.zfill(4)
            
            # Stores tuples of (predicted_sim_id, similarity_score, cam_idx, scale, sim_path_basename)
            individual_predictions_for_real_id = [] 
            
            print(f"  Processing real_id: {real_id_str} (True Sim ID: {true_sim_id_int})")
            for cam_idx in [1, 2]:
                real_cam_folder_name = f"cam{cam_idx}"
                real_image_base_dir_for_cam = os.path.join(base_output_dir, real_cam_folder_name, f"cutout_{key}")
                
                real_filename_pattern = f"cutout_camera{cam_idx}_{real_id_formatted}_*_mask_0.png"
                search_path_pattern = os.path.join(real_image_base_dir_for_cam, real_filename_pattern)
                found_real_files = glob.glob(search_path_pattern)

                if not found_real_files:
                    # print(f"    Info: No real image found for key '{key}', cam '{real_cam_folder_name}', ID '{real_id_str}' (pattern: {search_path_pattern})")
                    continue
                
                for real_image_path in found_real_files: # Typically one, but glob returns a list
                    # print(f"    Real image: {os.path.basename(real_image_path)}")
                    for scale_val in SCALES_SHORTEST_EDGE:
                        current_sim_embeddings_for_cam_scale = sim_embeddings_this_key_cam_scale.get(cam_idx, {}).get(scale_val, {})
                        if not current_sim_embeddings_for_cam_scale:
                            # print(f"      Skipping scale {scale_val} for {os.path.basename(real_image_path)}, no sim embeddings for cam{cam_idx} at this scale.")
                            continue

                        real_embedding = get_image_embedding(real_image_path, scale_val)
                        if real_embedding is None:
                            # print(f"      Skipping scale {scale_val} for {os.path.basename(real_image_path)} due to real embedding error.")
                            continue

                        best_match_sim_path_at_scale_cam = None
                        highest_similarity_at_scale_cam = -2.0

                        for sim_path, sim_embedding in current_sim_embeddings_for_cam_scale.items():
                            similarity = cosine_similarity(real_embedding, sim_embedding, dim=1).item()
                            if similarity > highest_similarity_at_scale_cam:
                                highest_similarity_at_scale_cam = similarity
                                best_match_sim_path_at_scale_cam = sim_path
                        
                        if best_match_sim_path_at_scale_cam:
                            predicted_sim_filename = os.path.basename(best_match_sim_path_at_scale_cam)
                            predicted_sim_id_at_scale_cam = extract_sim_id_from_filename(predicted_sim_filename)
                            if predicted_sim_id_at_scale_cam is not None:
                                individual_predictions_for_real_id.append(
                                    (predicted_sim_id_at_scale_cam, highest_similarity_at_scale_cam, 
                                     cam_idx, scale_val, predicted_sim_filename)
                                )
                                # print(f"      Cam{cam_idx}, Scale {scale_val}: Match {predicted_sim_id_at_scale_cam} (Sim: {highest_similarity_at_scale_cam:.4f})")
            
            # --- 3. Majority Vote for this real_id_str ---
            final_predicted_sim_id = None
            is_correct = False
            best_overall_similarity_for_pred = -2.0
            aggregation_details_str = "N/A"
            num_contributions_for_pred = 0

            if individual_predictions_for_real_id:
                if aggregation_method == "vote":
                    votes = [pred[0] for pred in individual_predictions_for_real_id]
                    if votes:
                        vote_counts = Counter(votes)
                        aggregation_details_str = str(dict(vote_counts))
                        most_common_preds = vote_counts.most_common()
                        
                        if most_common_preds:
                            final_predicted_sim_id = most_common_preds[0][0]
                            num_contributions_for_pred = most_common_preds[0][1]

                            # Tie-breaking
                            if len(most_common_preds) > 1 and most_common_preds[0][1] == most_common_preds[1][1]:
                                tied_vote_count = most_common_preds[0][1]
                                candidates_in_tie = [item[0] for item in most_common_preds if item[1] == tied_vote_count]
                                
                                max_sim_for_tied_candidate = -3.0
                                best_tied_candidate_id = final_predicted_sim_id # Default to first one

                                for tied_candidate_id in candidates_in_tie:
                                    current_max_sim_for_this_candidate = -3.0
                                    for pred_tuple in individual_predictions_for_real_id:
                                        if pred_tuple[0] == tied_candidate_id: # pred_tuple[0] is predicted_sim_id
                                            if pred_tuple[1] > current_max_sim_for_this_candidate: # pred_tuple[1] is similarity
                                                current_max_sim_for_this_candidate = pred_tuple[1]
                                    
                                    if current_max_sim_for_this_candidate > max_sim_for_tied_candidate:
                                        max_sim_for_tied_candidate = current_max_sim_for_this_candidate
                                        best_tied_candidate_id = tied_candidate_id
                                
                                final_predicted_sim_id = best_tied_candidate_id
                                print(f"      Vote Tie occurred for real_id {real_id_str}. Candidates: {candidates_in_tie}. Votes: {tied_vote_count}. Selected: {final_predicted_sim_id} (max_sim among tied: {max_sim_for_tied_candidate:.4f}).")
                            
                            # Determine the best similarity score associated with the final_predicted_sim_id
                            if final_predicted_sim_id is not None:
                                for pred_tuple in individual_predictions_for_real_id:
                                    if pred_tuple[0] == final_predicted_sim_id:
                                        if pred_tuple[1] > best_overall_similarity_for_pred:
                                            best_overall_similarity_for_pred = pred_tuple[1]
                                is_correct = (final_predicted_sim_id == true_sim_id_int)
                elif aggregation_method == "average":
                    scores_by_sim_id = {} # {sim_id: {'sum_score': float, 'count': int, 'max_individual_score': float}}
                    for pred_sim_id, sim_score, _, _, _ in individual_predictions_for_real_id:
                        if pred_sim_id not in scores_by_sim_id:
                            scores_by_sim_id[pred_sim_id] = {'sum_score': 0.0, 'count': 0, 'max_individual_score': -3.0}
                        scores_by_sim_id[pred_sim_id]['sum_score'] += sim_score
                        scores_by_sim_id[pred_sim_id]['count'] += 1
                        if sim_score > scores_by_sim_id[pred_sim_id]['max_individual_score']:
                            scores_by_sim_id[pred_sim_id]['max_individual_score'] = sim_score
                    
                    aggregation_details_str = str({sid: data['sum_score'] for sid, data in scores_by_sim_id.items()})

                    if scores_by_sim_id:
                        best_sum_score = -float('inf')
                        candidate_sim_ids_for_avg = []

                        for sim_id, data in scores_by_sim_id.items():
                            if data['sum_score'] > best_sum_score:
                                best_sum_score = data['sum_score']
                                candidate_sim_ids_for_avg = [sim_id]
                            elif data['sum_score'] == best_sum_score:
                                candidate_sim_ids_for_avg.append(sim_id)
                        
                        if candidate_sim_ids_for_avg:
                            if len(candidate_sim_ids_for_avg) == 1:
                                final_predicted_sim_id = candidate_sim_ids_for_avg[0]
                            else: # Tie-breaking for averaging based on max individual score, then count
                                print(f"      Score Sum Tie for real_id {real_id_str}. Candidates: {candidate_sim_ids_for_avg} with sum_score: {best_sum_score:.4f}. Applying tie-breaking...")
                                best_max_individual_score_in_tie = -3.0
                                best_count_in_tie = 0
                                final_predicted_sim_id = candidate_sim_ids_for_avg[0] # Default

                                tied_candidates_with_metrics = []
                                for sim_id_in_tie in candidate_sim_ids_for_avg:
                                    tied_candidates_with_metrics.append(
                                        (sim_id_in_tie, 
                                         scores_by_sim_id[sim_id_in_tie]['max_individual_score'],
                                         scores_by_sim_id[sim_id_in_tie]['count'])
                                    )
                                
                                # Sort by max_individual_score (desc), then by count (desc)
                                tied_candidates_with_metrics.sort(key=lambda x: (x[1], x[2]), reverse=True)
                                final_predicted_sim_id = tied_candidates_with_metrics[0][0]
                                print(f"        Selected by tie-breaking (max_indiv_score, then count): {final_predicted_sim_id} (IndivScore: {tied_candidates_with_metrics[0][1]:.4f}, Count: {tied_candidates_with_metrics[0][2]})")

                            if final_predicted_sim_id is not None:
                                best_overall_similarity_for_pred = scores_by_sim_id[final_predicted_sim_id]['sum_score']
                                num_contributions_for_pred = scores_by_sim_id[final_predicted_sim_id]['count']
                    if final_predicted_sim_id is not None:
                        is_correct = (final_predicted_sim_id == true_sim_id_int)
            if final_predicted_sim_id is None: 
                final_predicted_sim_id = "Error/None"
                best_overall_similarity_for_pred = "N/A"
            elif isinstance(best_overall_similarity_for_pred, float):
                best_overall_similarity_for_pred = f"{best_overall_similarity_for_pred:.4f}"

            # For representative path, just pick one if available, e.g., cam1's first match
            representative_real_image_path = "N/A"
            real_filename_pattern_cam1 = f"cutout_camera1_{real_id_formatted}_*_mask_0.png"
            search_path_pattern_cam1 = os.path.join(base_output_dir, f"cam1/cutout_{key}", real_filename_pattern_cam1)
            found_real_files_cam1 = glob.glob(search_path_pattern_cam1)
            if found_real_files_cam1:
                representative_real_image_path = os.path.basename(found_real_files_cam1[0])

            result_entry = {
                "key": key,
                "real_image_id_str": real_id_str,
                "representative_real_image_path": representative_real_image_path,
                "true_sim_id": true_sim_id_int,
                "final_predicted_sim_id": final_predicted_sim_id,
                "aggregation_method": aggregation_method,
                "num_contributions_for_prediction": num_contributions_for_pred if final_predicted_sim_id != "Error/None" else 0,
                "total_individual_predictions": len(individual_predictions_for_real_id),
                "aggregation_details": aggregation_details_str, # Renamed from vote_details
                "metric_for_prediction": best_overall_similarity_for_pred, # Renamed from best_similarity_for_pred
                "is_correct": is_correct
            }
            all_results.append(result_entry)
            print(f"    -> Final for RealID {real_id_str} (Method: {aggregation_method}): Pred={final_predicted_sim_id}, True={true_sim_id_int}, Correct={is_correct}, Contributions={num_contributions_for_pred}/{len(individual_predictions_for_real_id)}, Details: {aggregation_details_str}")

    # --- 4. Save results and calculate accuracy ---
    if not all_results:
        print("No results to save. Evaluation might have encountered issues or no matching data was found.")
        return

    results_df = pd.DataFrame(all_results)
    try:
        # Create output directory if it doesn't exist for the results file
        os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
        results_df.to_csv(results_file_path, index=False)
        print(f"\nEvaluation results saved to: {results_file_path}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

    if not results_df.empty and 'is_correct' in results_df.columns:
        valid_predictions_df = results_df[results_df['final_predicted_sim_id'] != "Error/None"]
        if not valid_predictions_df.empty:
            total_predictions = len(valid_predictions_df)
            correct_predictions = valid_predictions_df['is_correct'].sum()
            accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
            
            print("\n--- Accuracy Summary (Majority Vote & Multi-Scale) ---")
            print(f"Total Valid Comparisons (Real Entries with Votes): {total_predictions}")
            print(f"Correct Predictions: {correct_predictions}")
            print(f"Overall Accuracy: {accuracy:.2f}%")

            print("\nAccuracy per Key:")
            for key_val, group in valid_predictions_df.groupby('key'):
                key_total = len(group)
                key_correct = group['is_correct'].sum()
                key_accuracy = (key_correct / key_total) * 100 if key_total > 0 else 0
                print(f"  Key '{key_val}': {key_correct}/{key_total} = {key_accuracy:.2f}%")
        else:
            print("No valid predictions (Error/None) were made to calculate accuracy.")
    else:
        print("No 'is_correct' column found or DataFrame is empty, cannot calculate accuracy.")
    
    print("\nEvaluation finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate real image to simulator image matching based on class.json.")
    parser.add_argument(
        "--key", 
        type=str, 
        nargs='*', # Allows zero or more keys
        help="Specific key(s) from class.json to evaluate (e.g., S R stairs_branched). If not provided, all keys will be evaluated."
    )
    parser.add_argument(
        "--class_json", 
        type=str, 
        default=DEFAULT_CLASS_JSON_PATH,
        help=f"Path to the class.json file (default: {DEFAULT_CLASS_JSON_PATH})"
    )
    parser.add_argument(
        "--base_dir", 
        type=str, 
        default=DEFAULT_BASE_OUTPUT_DIR,
        help=f"Base directory for 'outputs' containing camX and sim_camX subfolders (default: {DEFAULT_BASE_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default=DEFAULT_RESULTS_FILE,
        help=f"Path to save the CSV results file (default: {DEFAULT_RESULTS_FILE})"
    )
    parser.add_argument(
        "--aggregation_method",
        type=str,
        default="vote",
        choices=["vote", "average"],
        help="Method to aggregate predictions from multiple scales/cameras: 'vote' for majority voting, 'average' for sum of scores (default: vote)"
    )

    args = parser.parse_args()
    
    # Ensure the script can find the model and other resources if run from a different directory
    # by making paths relative to the script's location if they are not absolute.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    class_json_path = args.class_json if os.path.isabs(args.class_json) else os.path.join(script_dir, args.class_json)
    base_output_dir = args.base_dir if os.path.isabs(args.base_dir) else os.path.join(script_dir, args.base_dir)
    results_file_path = args.output_file if os.path.isabs(args.output_file) else os.path.join(script_dir, args.output_file)

    run_evaluation(args.key, class_json_path, base_output_dir, results_file_path, args.aggregation_method)
