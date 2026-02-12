#!/bin/bash
# ============================================
# Install Official 3DGS Implementation
# ============================================
# This script clones and sets up the official
# 3D Gaussian Splatting repository.
# ============================================

set -e

echo "============================================"
echo "  Installing Official 3DGS"
echo "============================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPOS_DIR="$PROJECT_DIR/repos"

mkdir -p "$REPOS_DIR"
cd "$REPOS_DIR"

# Clone the official repository
if [ -d "gaussian-splatting" ]; then
    echo "Official 3DGS repository already exists."
    read -p "Do you want to update it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd gaussian-splatting
        git pull
        cd ..
    fi
else
    echo "Cloning official 3DGS repository..."
    git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive
fi

# Create dedicated conda environment
ENV_NAME="gaussian_splatting"

echo ""
echo "Setting up environment for official 3DGS..."

# Check if environment exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' already exists."
else
    echo "Creating conda environment: $ENV_NAME"
    conda create -n $ENV_NAME python=3.9 -y
fi

# Activate and install
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

cd gaussian-splatting

# Install PyTorch with CUDA
echo "Installing PyTorch..."
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Compile CUDA modules
echo "Compiling CUDA rasterization module..."
pip install submodules/diff-gaussian-rasterization

echo "Compiling simple-knn module..."
pip install submodules/simple-knn

echo ""
echo "============================================"
echo "  Installation Complete!"
echo "============================================"
echo ""
echo "To use the official 3DGS:"
echo "  conda activate gaussian_splatting"
echo "  cd $REPOS_DIR/gaussian-splatting"
echo ""
echo "Train on your data:"
echo "  python train.py -s <path_to_data>"
echo ""
echo "Render results:"
echo "  python render.py -m <path_to_model>"
