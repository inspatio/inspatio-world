#!/bin/bash
# Run full inference pipeline on the example video with cyclic orbit trajectory.
# Usage: bash run_example.sh
set -e

bash run_test_pipeline.sh \
  --input_dir ./test/example \
  --traj_txt_path ./traj/x_y_circle_cycle.txt 

bash run_test_pipeline.sh \
  --input_dir ./test/example2 \
  --traj_txt_path ./traj/x_y_circle_cycle.txt 

bash run_test_pipeline.sh \
  --input_dir ./test/example2 \
  --traj_txt_path ./traj/zoom_out_in.txt

bash run_test_pipeline.sh \
  --input_dir ./test/example \
  --traj_txt_path ./traj/x_y_circle_cycle.txt \
  --config_path ./configs/inference_1.3b.yaml \
  --checkpoint_path ./checkpoints/InSpatio-World-1.3B/InSpatio-World-1.3B.safetensors 
  
bash run_test_pipeline.sh \
  --input_dir ./test/example2 \
  --traj_txt_path ./traj/x_y_circle_cycle.txt \
  --config_path ./configs/inference_1.3b.yaml \
  --checkpoint_path ./checkpoints/InSpatio-World-1.3B/InSpatio-World-1.3B.safetensors

bash run_test_pipeline.sh \
  --input_dir ./test/example2 \
  --traj_txt_path ./traj/zoom_out_in.txt \
  --config_path ./configs/inference_1.3b.yaml \
  --checkpoint_path ./checkpoints/InSpatio-World-1.3B/InSpatio-World-1.3B.safetensors


bash run_test_pipeline.sh \
  --input_dir ./test/example3 \
  --traj_txt_path ./traj/x_y_circle_cycle.txt \
  --config_path ./configs/inference_1.3b.yaml \
  --checkpoint_path ./checkpoints/InSpatio-World-1.3B/InSpatio-World-1.3B.safetensors \
  --step1_gpus 6,7 \
  --step2_gpus 6,7 \
  --step3_gpus 6,7 \
  --step3_nproc 2 \
  --output_folder ./output/example_multitest 