#!/bin/bash

# Check if a task is provided as an argument
if [ $# -eq 1 ]; then
  # Use the provided task
  tasks=("$1")
  echo "Using provided task: $1"
else
  # Define the default list of tasks
  tasks=("S" "R" "fish_high" "stairs_rotated" "cliff" "faucet")
  echo "No task provided, using default tasks: ${tasks[@]}"
fi

SIM_BASE_FOLDER="$(dirname "$0")/../Robot_Digital_Twin/gazebo/outputs"

# Iterate over the tasks
for task in "${tasks[@]}"
do
  echo "Processing task: $task"

  # Command for cam1
  uv run lego_segmenter.py --img-folder "${SIM_BASE_FOLDER}/$task/cam1" --output-dir "outputs/sim_cam1/$task" --camera-name sim_cam1
  
  # Command for cam2
  uv run lego_segmenter.py --img-folder "${SIM_BASE_FOLDER}/$task/cam2" --output-dir "outputs/sim_cam2/$task" --camera-name sim_cam2
  
  echo "Finished processing task: $task"
  echo "------------------------------------"
done

echo "All tasks processed."