#!/bin/bash

# Define the list of tasks
tasks=("S" "R" "fish_high" "stairs_rotated" "cliff" "faucet")

# Iterate over the tasks
for task in "${tasks[@]}"
do
  echo "Processing task: $task"

  # Command for cam1
  python lego_segmenter.py --img-folder "sim_images/$task/cam1" --output-dir "outputs/sim_cam1/$task" --camera-name sim_cam1
  
  # Command for cam2
  python lego_segmenter.py --img-folder "sim_images/$task/cam2" --output-dir "outputs/sim_cam2/$task" --camera-name sim_cam2
  
  echo "Finished processing task: $task"
  echo "------------------------------------"
done

echo "All tasks processed."