"""
DUSt3R module: Dense point map and global alignment for 3D reconstruction.

Provides educational implementations of key DUSt3R concepts:
- PointMap: Dense 3D point representation (H×W×3 coordinate maps)
- GlobalAligner: Multi-view pose estimation and scene alignment
"""

from .pointmap import PointMap
from .alignment import GlobalAligner

__all__ = [
    'PointMap',
    'GlobalAligner',
]
