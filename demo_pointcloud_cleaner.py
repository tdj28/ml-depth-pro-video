#!/usr/bin/env python3

"""
Demo script for creating clean 3D point clouds from images using pointcloud_cleaner.py
"""

import os
import argparse
from pointcloud_cleaner import process_image_to_clean_pointcloud

def main():
    parser = argparse.ArgumentParser(description="Demo: Create a clean 3D point cloud from an image")
    
    # Input/output arguments
    parser.add_argument("--image_path", type=str, required=True,
                       help="Path to the input image")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Path to save the output point cloud (default: based on input filename)")
    
    # Performance parameters
    parser.add_argument("--downscale_factor", type=float, default=0.5,
                       help="Downscale input image for faster processing (default: 0.5)")
    
    # Processing parameters
    parser.add_argument("--fast", action="store_true",
                       help="Use faster processing with less detail")
    parser.add_argument("--high_quality", action="store_true",
                       help="Use higher quality processing with more detail (slower)")
    
    # Visualization parameters
    parser.add_argument("--view_preset", type=str, 
                       choices=["front", "top", "side", "isometric"], 
                       default="isometric",
                       help="View preset for visualization (default: isometric)")
    parser.add_argument("--no_visualization", action="store_true",
                       help="Skip generating a preview image")
    
    args = parser.parse_args()
    
    # Set parameters based on quality settings
    if args.high_quality:
        print("Using high quality settings (slower processing)")
        downscale_factor = 1.0 if args.downscale_factor == 0.5 else args.downscale_factor
        # High quality settings retain more points
        radius = 0.05      # Smaller radius for stray point detection
        nb_points = 10     # Fewer points required to keep a point
    elif args.fast:
        print("Using fast settings (quicker processing)")
        downscale_factor = 0.3 if args.downscale_factor == 0.5 else args.downscale_factor
        # Fast settings may be more aggressive in removing points
        radius = 0.15      # Larger radius for stray point detection
        nb_points = 25     # More points required to keep a point
    else:
        # Default balanced settings
        downscale_factor = args.downscale_factor
        radius = 0.1       # Medium radius for stray point detection
        nb_points = 20     # Medium threshold for keeping points
    
    # Process the image to create a clean point cloud
    pcd = process_image_to_clean_pointcloud(
        image_path=args.image_path,
        output_path=args.output_path,
        remove_strays=True,
        clean_shadows_flag=True,
        downscale_factor=downscale_factor,
        grid_size=20,
        ground_percentile=5,
        visualize=not args.no_visualization,
        view_preset=args.view_preset
    )
    
    if pcd is not None:
        num_points = len(pcd.points)
        print(f"Point cloud cleaning successful! Generated {num_points} points.")
        if args.output_path:
            print(f"Cleaned point cloud saved to: {args.output_path}")
        else:
            base_name = os.path.splitext(os.path.basename(args.image_path))[0]
            print(f"Cleaned point cloud saved to: {base_name}_clean.ply")
            if not args.no_visualization:
                print(f"Preview saved to: {base_name}_clean_preview.png")
    else:
        print("Point cloud creation failed.")

if __name__ == "__main__":
    main() 