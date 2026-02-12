#!/bin/bash
# ============================================
# Download Sample Data for 3DGS Tutorials
# ============================================

set -e

echo "============================================"
echo "  Downloading Sample Data"
echo "============================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"

mkdir -p "$DATA_DIR/sample_scenes"
mkdir -p "$DATA_DIR/sample_images"

cd "$DATA_DIR"

# Download a small sample scene (Truck scene from Tanks and Temples)
echo ""
echo "Downloading sample scene..."

# Create a simple synthetic dataset for quick testing
echo "Creating synthetic test data..."

python3 << 'EOF'
import numpy as np
import os
from PIL import Image

# Create sample images directory
os.makedirs('sample_images', exist_ok=True)

# Create some simple colored gradient images
for i in range(5):
    # Create a gradient image
    img = np.zeros((256, 256, 3), dtype=np.uint8)

    # Add some color variation
    for y in range(256):
        for x in range(256):
            img[y, x, 0] = int((x / 255.0) * 255)  # Red gradient
            img[y, x, 1] = int((y / 255.0) * 255)  # Green gradient
            img[y, x, 2] = int(((x + y) / 510.0 + i * 0.1) * 255) % 256  # Blue varies

    Image.fromarray(img).save(f'sample_images/image_{i:03d}.png')

print("Created 5 sample gradient images")

# Create a simple point cloud
points = np.random.randn(1000, 3) * 0.5
colors = np.random.rand(1000, 3)

np.savez('sample_scenes/simple_pointcloud.npz',
         points=points.astype(np.float32),
         colors=colors.astype(np.float32))

print("Created sample point cloud with 1000 points")
EOF

echo ""
echo "============================================"
echo "  Download Complete!"
echo "============================================"
echo ""
echo "Sample data location: $DATA_DIR"
echo "  - sample_images/: Test images"
echo "  - sample_scenes/: Point cloud data"
