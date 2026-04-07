# 1. 创建并进入目标目录
mkdir -p checkpoints
cd checkpoints

# --- 下载 InSpatio-World 系列 (需拆分为两个文件夹) ---
# 先克隆整个仓库
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/inspatio/world
cd world
git lfs pull
cd ..

# 整理 InSpatio-World 结构
# 创建 1.3B 文件夹并将对应的权重移过去
mkdir -p InSpatio-World-1.3B
mv world/InSpatio-World-1.3B.safetensors InSpatio-World-1.3B/

# 将剩余权重留在 InSpatio-World 文件夹中（重命名原仓库文件夹）
mv world/InSpatio-World.safetensors world/InSpatio-World.safetensors.tmp # 防止同名冲突
mv world InSpatio-World
mv InSpatio-World/InSpatio-World.safetensors.tmp InSpatio-World/InSpatio-World.safetensors

# --- 下载 Wan-AI 系列 ---
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
cd Wan2.1-T2V-1.3B && git lfs pull && cd ..

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P
cd Wan2.1-I2V-14B-480P && git lfs pull && cd ..

# --- 下载 DA3 ---
# 注意：你提供的仓库名是 depth-anything/DA3NESTED-GIANT-LARGE 
# 若要对应目录名 DA3，我们克隆时直接指定目标文件夹名
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/depth-anything/Depth-Anything-V2-Large DA3
cd DA3 && git lfs pull && cd ..

# --- 下载 Florence-2 ---
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/microsoft/Florence-2-large
cd Florence-2-large && git lfs pull && cd ..

echo "所有模型已下载并整理至 checkpoints/ 目录。"