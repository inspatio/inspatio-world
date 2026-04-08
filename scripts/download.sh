# 1. Create and enter the target directory
mkdir -p checkpoints
cd checkpoints

# --- Download the InSpatio-World series (needs to be split into two folders) ---
# Clone the entire repository first
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/inspatio/world
cd world
git lfs pull
cd ..

# Organize the InSpatio-World structure
# Create the 1.3B folder and move the corresponding weights into it
mkdir -p InSpatio-World-1.3B
mv world/InSpatio-World-1.3B.safetensors InSpatio-World-1.3B/

# Leave the remaining weights in the InSpatio-World folder (rename the original repository folder)
mv world/InSpatio-World.safetensors world/InSpatio-World.safetensors.tmp # Prevent name conflicts
mv world InSpatio-World
mv InSpatio-World/InSpatio-World.safetensors.tmp InSpatio-World/InSpatio-World.safetensors

# --- Download the Wan-AI series ---
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
cd Wan2.1-T2V-1.3B && git lfs pull && cd ..

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P
cd Wan2.1-I2V-14B-480P && git lfs pull && cd ..

# --- Download DA3 ---
# Note: The repository name you provided is depth-anything/DA3NESTED-GIANT-LARGE 
# To match the directory name DA3, we specify the target folder name directly when cloning
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/depth-anything/Depth-Anything-V2-Large DA3
cd DA3 && git lfs pull && cd ..

# --- Download Florence-2 ---
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/microsoft/Florence-2-large
cd Florence-2-large && git lfs pull && cd ..

# optional
git clone https://github.com/madebyollin/taehv.git
cd ..

echo "All models have been downloaded and organized into the checkpoints/ directory."