from PIL import Image
import depth_pro
import numpy as np
import torch
import cv2
import os
import argparse
from tqdm import tqdm

def create_parallax_effect(image_path, output_path, duration=5.0, fps=30, amplitude=0.05, 
                          motion_type="circle", resolution_scale=1.0):
    """
    Create a parallax/3D effect video from a single image using depth information.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the output video
        duration: Duration of the video in seconds
        fps: Frames per second
        amplitude: Amplitude of the camera motion
        motion_type: Type of camera motion ("circle", "zoom", "swing")
        resolution_scale: Scale factor for the output resolution (1.0 = original size)
    """
    # Load model and preprocessing transform
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()

    # Load and preprocess an image
    image, _, f_px = depth_pro.load_rgb(image_path)
    image_tensor = transform(image)

    # Run inference
    prediction = model.infer(image_tensor, f_px=f_px)
    depth = prediction["depth"]  # Depth in [m]
    
    # Convert to numpy
    depth_np = depth.detach().cpu().numpy()
    
    # Convert image to numpy array if it's not already
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # Resize if needed
    if resolution_scale != 1.0:
        h, w = image.shape[:2]
        new_h, new_w = int(h * resolution_scale), int(w * resolution_scale)
        image = cv2.resize(image, (new_w, new_h))
        depth_np = cv2.resize(depth_np, (new_w, new_h))
    
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Calculate total frames
    total_frames = int(duration * fps)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    # Normalize depth for better visualization
    depth_min = np.min(depth_np)
    depth_max = np.max(depth_np)
    depth_norm = (depth_np - depth_min) / (depth_max - depth_min)
    
    # Create x and y grids for the original image
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    print(f"Creating parallax effect video with {total_frames} frames...")
    for frame in tqdm(range(total_frames)):
        # Calculate the position in the animation cycle (0 to 2π)
        t = 2 * np.pi * frame / total_frames
        
        # Calculate the displacement based on the motion type
        if motion_type == "circle":
            dx = amplitude * w * np.cos(t)
            dy = amplitude * h * np.sin(t)
        elif motion_type == "zoom":
            # Zoom effect (in and out)
            zoom_factor = 1.0 + amplitude * np.sin(t)
            dx = (1 - zoom_factor) * (x_coords - w/2)
            dy = (1 - zoom_factor) * (y_coords - h/2)
        elif motion_type == "swing":
            # Swinging effect (left to right)
            dx = amplitude * w * np.sin(t)
            dy = 0
        else:
            raise ValueError(f"Unknown motion type: {motion_type}")
        
        # Apply depth-dependent displacement
        if motion_type == "zoom":
            # For zoom, we apply the calculated dx and dy directly
            map_x = x_coords + dx
            map_y = y_coords + dy
        else:
            # For other motions, scale displacement by depth
            map_x = x_coords + dx * (1 - depth_norm)
            map_y = y_coords + dy * (1 - depth_norm)
        
        # Ensure coordinates are within bounds
        map_x = np.clip(map_x, 0, w-1).astype(np.float32)
        map_y = np.clip(map_y, 0, h-1).astype(np.float32)
        
        # Remap the image using the calculated coordinates
        frame_image = cv2.remap(image, map_x.T, map_y.T, cv2.INTER_LINEAR)
        
        # Write the frame to the video
        video.write(frame_image)
    
    # Release the video writer
    video.release()
    print(f"Video saved to {output_path}")

def create_3d_anaglyph(image_path, output_path, separation=0.05):
    """
    Create a red-cyan anaglyph 3D image from a single image using depth information.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the output anaglyph image
        separation: Amount of separation between left and right views
    """
    # Load model and preprocessing transform
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()

    # Load and preprocess an image
    image, _, f_px = depth_pro.load_rgb(image_path)
    image_tensor = transform(image)

    # Run inference
    prediction = model.infer(image_tensor, f_px=f_px)
    depth = prediction["depth"]  # Depth in [m]
    
    # Convert to numpy
    depth_np = depth.detach().cpu().numpy()
    
    # Convert image to numpy array if it's not already
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Normalize depth for better visualization
    depth_min = np.min(depth_np)
    depth_max = np.max(depth_np)
    depth_norm = (depth_np - depth_min) / (depth_max - depth_min)
    
    # Create x and y grids for the original image
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Create left and right views
    dx = separation * w
    
    # Left view (shifted right)
    map_x_left = x_coords + dx * (1 - depth_norm)
    map_y_left = y_coords
    
    # Right view (shifted left)
    map_x_right = x_coords - dx * (1 - depth_norm)
    map_y_right = y_coords
    
    # Ensure coordinates are within bounds
    map_x_left = np.clip(map_x_left, 0, w-1).astype(np.float32)
    map_y_left = np.clip(map_y_left, 0, h-1).astype(np.float32)
    map_x_right = np.clip(map_x_right, 0, w-1).astype(np.float32)
    map_y_right = np.clip(map_y_right, 0, h-1).astype(np.float32)
    
    # Remap the image using the calculated coordinates
    left_image = cv2.remap(image, map_x_left.T, map_y_left.T, cv2.INTER_LINEAR)
    right_image = cv2.remap(image, map_x_right.T, map_y_right.T, cv2.INTER_LINEAR)
    
    # Create anaglyph image (red-cyan)
    anaglyph = np.zeros_like(image)
    anaglyph[:,:,0] = left_image[:,:,0]  # Red channel from left image
    anaglyph[:,:,1] = right_image[:,:,1]  # Green channel from right image
    anaglyph[:,:,2] = right_image[:,:,2]  # Blue channel from right image
    
    # Save the anaglyph image
    cv2.imwrite(output_path, anaglyph)
    print(f"Anaglyph image saved to {output_path}")
    
    return anaglyph

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create 3D effects from a single image using depth")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output")
    parser.add_argument("--effect", type=str, default="parallax", 
                        choices=["parallax", "anaglyph"], 
                        help="Type of 3D effect to create")
    
    # Parallax video parameters
    parser.add_argument("--duration", type=float, default=5.0, help="Duration of video in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--amplitude", type=float, default=0.05, 
                        help="Amplitude of camera motion (0.01-0.2 recommended)")
    parser.add_argument("--motion", type=str, default="circle", 
                        choices=["circle", "zoom", "swing"], 
                        help="Type of camera motion")
    parser.add_argument("--scale", type=float, default=1.0, 
                        help="Scale factor for output resolution")
    
    # Anaglyph parameters
    parser.add_argument("--separation", type=float, default=0.05, 
                        help="Separation between left and right views (0.01-0.1 recommended)")
    
    args = parser.parse_args()
    
    if args.effect == "parallax":
        create_parallax_effect(
            args.image_path, 
            args.output_path, 
            duration=args.duration,
            fps=args.fps,
            amplitude=args.amplitude,
            motion_type=args.motion,
            resolution_scale=args.scale
        )
    elif args.effect == "anaglyph":
        create_3d_anaglyph(
            args.image_path,
            args.output_path,
            separation=args.separation
        ) 