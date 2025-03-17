#!/usr/bin/env python3

"""
Demo script for converting an image to a clean 3D mesh using pointcloud_to_mesh.py
"""

import os
import argparse
from pointcloud_to_mesh import process_image_to_mesh

def main():
    parser = argparse.ArgumentParser(description="Demo: Convert an image to a clean 3D mesh")
    
    # Input/output arguments
    parser.add_argument("--image_path", type=str, required=True,
                       help="Path to the input image")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Path to save the output mesh (default: based on input filename)")
    
    # Performance parameters
    parser.add_argument("--downscale_factor", type=float, default=0.5,
                       help="Downscale input image for faster processing (default: 0.5)")
    
    # Processing parameters
    parser.add_argument("--fast", action="store_true",
                       help="Use faster processing with less detail")
    parser.add_argument("--high_quality", action="store_true",
                       help="Use higher quality processing with more detail (slower)")
    parser.add_argument("--mesh_method", type=str, 
                       choices=["poisson", "ball_pivoting", "simple"], 
                       default="poisson",
                       help="Method to use for mesh creation: poisson (smooth), ball_pivoting (more detail), simple (uniform triangles)")
    
    args = parser.parse_args()
    
    # Set parameters based on quality settings
    if args.high_quality:
        print("Using high quality settings (slower processing)")
        voxel_size = 0.02  # Smaller voxel size for more detail
        depth = 9          # Higher depth for better reconstruction
        downscale_factor = 1.0 if args.downscale_factor == 0.5 else args.downscale_factor
        mesh_method = args.mesh_method
    elif args.fast:
        print("Using fast settings (quicker processing)")
        voxel_size = 0.1   # Larger voxel size for faster processing
        depth = 7          # Lower depth for faster reconstruction
        downscale_factor = 0.3 if args.downscale_factor == 0.5 else args.downscale_factor
        mesh_method = "simple" if args.mesh_method == "poisson" else args.mesh_method  # Simple is faster than Poisson
    else:
        # Default balanced settings
        voxel_size = 0.05  # Medium voxel size
        depth = 8          # Medium depth
        downscale_factor = args.downscale_factor
        mesh_method = args.mesh_method
    
    # Process the image to create a mesh
    mesh = process_image_to_mesh(
        image_path=args.image_path,
        output_path=args.output_path,
        voxel_size=voxel_size,
        depth=depth,
        remove_strays=True,
        clean_shadows_flag=True,
        downscale_factor=downscale_factor,
        grid_size=20,
        ground_percentile=5,
        mesh_method=mesh_method
    )
    
    if mesh is not None:
        print("Mesh creation successful!")
        if args.output_path:
            print(f"Mesh saved to: {args.output_path}")
        else:
            base_name = os.path.splitext(os.path.basename(args.image_path))[0]
            print(f"Mesh saved to: {base_name}_mesh.obj")
            print(f"Preview saved to: {base_name}_mesh_preview.png")
    else:
        print("Mesh creation failed.")

if __name__ == "__main__":
    main() 