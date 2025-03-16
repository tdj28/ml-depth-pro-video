#!/usr/bin/env python3
"""
Create a 2D floor plan from a depth map by projecting objects onto the ground plane.
This script uses the ground plane detection from mesh_from_depth.py to create a top-down
floor plan view of the scene.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import depth_pro
from mesh_from_depth import detect_ground_plane, remove_depth_shadows, apply_rotation_to_plane, create_floor_plan

def main():
    parser = argparse.ArgumentParser(description="Create a 2D floor plan from a depth map")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_path", type=str, default="floor_plan.png", help="Path to save the floor plan image")
    parser.add_argument("--height_threshold", type=float, default=0.5,
                        help="Height threshold in meters above ground plane (objects above this will be included)")
    parser.add_argument("--no_visualize", action="store_false", dest="visualize", 
                        help="Disable visualization")
    parser.add_argument("--keep_shadows", action="store_false", dest="remove_shadows",
                        help="Keep shadow artifacts (don't remove them)")
    parser.add_argument("--no_ground_interp", action="store_false", dest="interpolate_ground",
                        help="Don't interpolate ground in shadow areas")
    parser.add_argument("--no_saved_ground", action="store_false", dest="use_saved_ground",
                        help="Don't use saved ground plane parameters (always detect)")
    parser.add_argument("--ground_params_dir", type=str,
                        help="Directory to save/load ground plane parameters (defaults to image directory)")
    parser.add_argument("--rot_x", type=float, default=0.0,
                        help="Rotation around X axis in degrees (can be negative)")
    parser.add_argument("--rot_y", type=float, default=0.0,
                        help="Rotation around Y axis in degrees (can be negative)")
    parser.add_argument("--rot_z", type=float, default=0.0,
                        help="Rotation around Z axis in degrees (can be negative)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for the output image")
    
    # Add new floor plan options
    parser.add_argument("--simplified", action="store_true",
                        help="Create a simplified floor plan with cleaner outlines")
    parser.add_argument("--color_by_height", action="store_true",
                        help="Color the floor plan by height")
    parser.add_argument("--max_height", type=float, default=2.5,
                        help="Maximum height in meters for color mapping")
    parser.add_argument("--show_text", action="store_true",
                        help="Show text labels on the floor plan")
    
    args = parser.parse_args()
    
    # Create rotation offset from individual axis rotations
    rotation_offset = [args.rot_x, args.rot_y, args.rot_z]
    
    # Only use rotation if at least one axis has a non-zero rotation
    if rotation_offset == [0.0, 0.0, 0.0]:
        rotation_offset = None
    else:
        print(f"Using rotation offset: {rotation_offset[0]}, {rotation_offset[1]}, {rotation_offset[2]} degrees")
    
    # Load model and preprocessing transform
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()

    # Load and preprocess an image
    image, _, f_px = depth_pro.load_rgb(args.image_path)
    image_tensor = transform(image)

    # Run inference
    prediction = model.infer(image_tensor, f_px=f_px)
    depth = prediction["depth"]  # Depth in [m]
    focallength_px = prediction["focallength_px"]  # Focal length in pixels
    
    # Convert depth to numpy array
    depth_np = depth.detach().cpu().numpy()
    
    # Remove shadow artifacts if requested
    ground_mask = None
    ground_model = None
    if args.remove_shadows:
        depth_np, ground_mask, ground_model = remove_depth_shadows(
            depth_np, np.array(image), 
            interpolate_ground=args.interpolate_ground,
            image_path=args.image_path,
            use_saved_ground=args.use_saved_ground,
            rotation_offset=rotation_offset,
            ground_params_dir=args.ground_params_dir
        )
    
    # If we don't have a ground model yet, detect it
    if ground_model is None:
        print("Detecting ground plane...")
        ground_mask, ground_model = detect_ground_plane(depth_np)
        
        # Apply rotation offset if provided
        if rotation_offset is not None:
            ground_model = apply_rotation_to_plane(ground_model, rotation_offset)
    
    # Create the floor plan
    floor_plan = create_floor_plan(
        depth_np, 
        np.array(image), 
        ground_model, 
        focallength_px.item(),
        height_threshold=args.height_threshold,
        output_path=args.output_path,
        visualize=args.visualize,
        dpi=args.dpi,
        simplified=args.simplified,
        color_by_height=args.color_by_height,
        max_height=args.max_height,
        show_text=args.show_text
    )
    
    if floor_plan is None:
        print("Failed to create floor plan")
        return 1
    
    print(f"Floor plan created successfully and saved to {args.output_path}")
    return 0

if __name__ == "__main__":
    exit(main()) 