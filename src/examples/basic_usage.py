"""
Basic usage example for Viviani4D framework.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.viviani_surface_4d import VivianiSurface4D
import numpy as np

def main():
    """Demonstrate basic functionality of Viviani4D."""
    
    print("=== Viviani4D Basic Usage Example ===\n")
    
    # 1. Create a 4D Viviani surface
    print("1. Creating 4D Viviani surface with radius=2.0")
    viviani = VivianiSurface4D(radius=2.0)
    
    # 2. Build the NURBS representation
    print("2. Building exact NURBS representation")
    viviani.build_nurbs_surface()
    
    # 3. Show control points information
    print(f"3. Control points shape: {viviani.control_points_4d.shape}")
    print(f"   Weights shape: {viviani.weights_4d.shape}")
    print(f"   Knot vectors: u-direction ({len(viviani.knot_vectors[0])} knots), "
          f"v-direction ({len(viviani.knot_vectors[1])} knots)")
    
    # 4. Project to 3D
    print("4. Projecting to 3D using standard basis")
    projected = viviani.project_to_3d()
    
    # 5. Show projection results
    print(f"5. Projected shape: {projected.shape}")
    print(f"   Min coordinates: [{projected[:,:,0].min():.3f}, "
          f"{projected[:,:,1].min():.3f}, {projected[:,:,2].min():.3f}]")
    print(f"   Max coordinates: [{projected[:,:,0].max():.3f}, "
          f"{projected[:,:,1].max():.3f}, {projected[:,:,2].max():.3f}]")
    
    print("\n=== Example completed successfully ===")
    print("\nTo visualize the surface, run:")
    print("  viviani.visualize()")
    print("\nOr with custom projection:")
    print("  rotation_matrix = np.eye(4)[:3, :]  # Customize this")
    print("  viviani.visualize(projection_basis=rotation_matrix)")

if __name__ == "__main__":
    main()
