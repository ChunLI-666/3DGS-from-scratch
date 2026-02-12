#!/bin/bash
# ============================================
# 3DGS-from-scratch Environment Setup Script
# ============================================
# This script sets up the complete development environment
# for the 3DGS tutorial series.
#
# Usage: bash scripts/setup_env.sh
# ============================================

set -e

echo "============================================"
echo "  3DGS-from-scratch Environment Setup"
echo "============================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if conda is available
check_conda() {
    if command -v conda &> /dev/null; then
        print_status "Conda found: $(conda --version)"
        return 0
    else
        print_error "Conda not found. Please install Miniconda or Anaconda first."
        echo "  Visit: https://docs.conda.io/en/latest/miniconda.html"
        return 1
    fi
}

# Check CUDA availability
check_cuda() {
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
        print_status "CUDA found: $CUDA_VERSION"
        return 0
    else
        print_warning "CUDA not found. GPU acceleration will not be available."
        return 1
    fi
}

# Check GPU
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
        GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n1)
        print_status "GPU found: $GPU_NAME ($GPU_MEMORY)"
        return 0
    else
        print_warning "nvidia-smi not found. GPU may not be available."
        return 1
    fi
}

# Create conda environment
create_env() {
    ENV_NAME="3dgs-tutorial"

    # Check if environment already exists
    if conda env list | grep -q "^$ENV_NAME "; then
        print_warning "Environment '$ENV_NAME' already exists."
        read -p "Do you want to recreate it? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            conda env remove -n $ENV_NAME -y
        else
            print_status "Using existing environment."
            return 0
        fi
    fi

    echo ""
    echo "Creating conda environment: $ENV_NAME"
    conda create -n $ENV_NAME python=3.10 -y

    print_status "Conda environment created."
}

# Install dependencies
install_deps() {
    ENV_NAME="3dgs-tutorial"

    echo ""
    echo "Installing dependencies..."

    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate $ENV_NAME

    # Determine CUDA version for PyTorch
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
        CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d'.' -f1)
        CUDA_MINOR=$(echo $CUDA_VERSION | cut -d'.' -f2)

        if [[ "$CUDA_MAJOR" == "12" ]]; then
            PYTORCH_CUDA="cu121"
        elif [[ "$CUDA_MAJOR" == "11" && "$CUDA_MINOR" -ge "8" ]]; then
            PYTORCH_CUDA="cu118"
        else
            PYTORCH_CUDA="cu117"
        fi

        echo "Installing PyTorch with CUDA support ($PYTORCH_CUDA)..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/$PYTORCH_CUDA
    else
        echo "Installing PyTorch (CPU only)..."
        pip install torch torchvision
    fi

    # Install other requirements
    pip install -r requirements.txt

    # Install the package in development mode
    pip install -e .

    print_status "Dependencies installed."
}

# Setup Jupyter kernel
setup_jupyter() {
    ENV_NAME="3dgs-tutorial"

    echo ""
    echo "Setting up Jupyter kernel..."

    eval "$(conda shell.bash hook)"
    conda activate $ENV_NAME

    python -m ipykernel install --user --name $ENV_NAME --display-name "Python (3DGS Tutorial)"

    print_status "Jupyter kernel installed."
}

# Main setup flow
main() {
    echo "Checking prerequisites..."
    echo ""

    check_conda || exit 1
    check_cuda
    check_gpu

    echo ""
    echo "============================================"
    echo "  Starting Installation"
    echo "============================================"

    create_env
    install_deps
    setup_jupyter

    echo ""
    echo "============================================"
    echo "  Setup Complete!"
    echo "============================================"
    echo ""
    print_status "Environment: 3dgs-tutorial"
    print_status "Jupyter kernel: Python (3DGS Tutorial)"
    echo ""
    echo "To activate the environment:"
    echo "  conda activate 3dgs-tutorial"
    echo ""
    echo "To start Jupyter Lab:"
    echo "  jupyter lab"
    echo ""
    echo "Happy learning!"
}

main "$@"
