#!/bin/bash
# Run full inference pipeline on the example video with cyclic orbit trajectory.
# Usage: bash run_example.sh
set -e

bash run_test_pipeline.sh \
  --input_dir ./test/example \
  --traj_txt_path ./traj/x_y_circle_cycle.txt \
  --disable_adaptive_frame 

bash run_test_pipeline.sh \
  --input_dir ./test/example \
  --traj_txt_path ./traj/x_y_circle_cycle.txt \
  --config_path ./configs/inference_1.3b.yaml \
  --checkpoint_path ./checkpoints/InSpatio-World-1.3B/InSpatio-World-1.3B.safetensors \
  --skip_step1 --skip_step2 \
  --disable_adaptive_frame 

bash run_test_pipeline.sh \
  --input_dir ./test/example \
  --traj_txt_path ./traj/x_y_circle_cycle.txt \
  --freeze_repeat 150 \
  --skip_step1 --skip_step2 \
  --output_folder ./output/example_freeze_repeat_150 \
  --disable_adaptive_frame 

bash run_test_pipeline.sh \
  --input_dir ./test/example \
  --traj_txt_path ./traj/x_y_circle_cycle.txt \
  --skip_step1 --skip_step2 \
  --use_tae --compile_dit \
  --config_path ./configs/inference_1.3b.yaml  \
  --checkpoint_path ./checkpoints/InSpatio-World-1.3B/InSpatio-World-1.3B.safetensors \
  --disable_adaptive_frame 

bash run_test_pipeline.sh \
  --input_dir ./test/example2 \
  --traj_txt_path ./traj/x_y_circle_cycle.txt 

bash run_test_pipeline.sh \
  --input_dir ./test/example2 \
  --traj_txt_path ./traj/zoom_out_in.txt \
  --skip_step1 --skip_step2
  
bash run_test_pipeline.sh \
  --input_dir ./test/example2 \
  --traj_txt_path ./traj/x_y_circle_cycle.txt \
  --config_path ./configs/inference_1.3b.yaml \
  --checkpoint_path ./checkpoints/InSpatio-World-1.3B/InSpatio-World-1.3B.safetensors \
  --skip_step1 --skip_step2

bash run_test_pipeline.sh \
  --input_dir ./test/example3 \
  --traj_txt_path ./traj/x_y_circle_cycle.txt \
  --relative_to_source \
  --rotation_only \
  --disable_adaptive_frame 