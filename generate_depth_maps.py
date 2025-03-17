#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Import depth estimation model
import depth_pro
from img_to_normalized_pointcloud import get_torch_device, is_apple_silicon

def colorize_depth(depth, min_depth=None, max_depth=None, cmap='turbo'):
    """
    Colorize a depth map using a colormap.
    
    Args:
        depth: Depth map (HxW)
        min_depth: Minimum depth value for colormap (default: auto)
        max_depth: Maximum depth value for colormap (default: auto)
        cmap: Colormap name (default: 'turbo')
        
    Returns:
        Colorized depth map (HxWx3)
    """
    if min_depth is None:
        min_depth = np.nanmin(depth)
    if max_depth is None:
        max_depth = np.nanmax(depth)
    
    # Normalize depth to 0-1 range
    depth_norm = (depth - min_depth) / (max_depth - min_depth)
    depth_norm = np.clip(depth_norm, 0, 1)
    
    # Apply colormap
    mapper = plt.get_cmap(cmap)
    colored_depth = mapper(depth_norm)[:, :, :3]  # Remove alpha channel
    
    # Convert to 0-255 range for saving as image
    colored_depth = (colored_depth * 255).astype(np.uint8)
    
    return colored_depth

def generate_depth_map(image_path, output_path=None, downscale_factor=1.0, half_precision=False, colored=True, cmap='turbo'):
    """
    Generate a depth map from an input image and save it.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the output depth map (default: derived from image_path)
        downscale_factor: Downscale factor for image processing
        half_precision: Use half precision for computation
        colored: Whether to save a colored depth map
        cmap: Colormap for depth visualization
        
    Returns:
        Path to the saved depth map
    """
    try:
        # Default output path if not specified
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"{base_name}_depth.png"
        
        # Get the optimal device for the system
        device = get_torch_device()
        print(f"Using device: {device}")
        
        # Set precision
        precision = torch.float16 if (half_precision or device.type == 'mps') else torch.float32
        print(f"Using {precision} precision for computation")
        
        # Load model and preprocessing transform
        model, transform = depth_pro.create_model_and_transforms(
            device=device,
            precision=precision
        )
        model.eval()

        # Enable torch compile for PyTorch 2.0+ if available
        if hasattr(torch, 'compile') and device.type != 'cpu':
            try:
                model = torch.compile(model)
                print("Using torch.compile() for model acceleration")
            except Exception as e:
                print(f"Warning: Unable to use torch.compile(): {e}")
        
        # Load the image
        image, _, f_px = depth_pro.load_rgb(image_path)
        
        # Downscale the image if requested
        orig_size = None
        if downscale_factor != 1.0 and downscale_factor > 0:
            orig_size = image.shape[:2]  # (height, width)
            new_height = int(orig_size[0] * downscale_factor)
            new_width = int(orig_size[1] * downscale_factor)
            print(f"Downscaling image from {orig_size[1]}x{orig_size[0]} to {new_width}x{new_height}")
            
            # Resize the image
            image = cv2.resize(
                image, 
                (new_width, new_height), 
                interpolation=cv2.INTER_AREA if downscale_factor < 1.0 else cv2.INTER_LINEAR
            )
            
            # Adjust focal length
            if f_px is not None:
                f_px = f_px * downscale_factor
        
        # Apply transform and run inference
        image_tensor = transform(image)

        # Run inference
        with torch.no_grad():
            prediction = model.infer(image_tensor, f_px=f_px)
            depth = prediction["depth"]  # Depth in [m]
            
        # Convert depth to numpy array
        depth_np = depth.detach().cpu().numpy()
        
        # Get image dimensions
        h, w = depth_np.shape
        print(f"Depth map dimensions: {w}x{h}")
        
        # Save depth map
        if colored:
            # Colorize depth map
            colored_depth = colorize_depth(depth_np, cmap=cmap)
            
            # Save colored depth map
            cv2.imwrite(output_path, cv2.cvtColor(colored_depth, cv2.COLOR_RGB2BGR))
            print(f"Saved colored depth map to {output_path}")
        else:
            # Normalize depth values to 0-65535 range (16-bit grayscale)
            min_depth = np.nanmin(depth_np)
            max_depth = np.nanmax(depth_np)
            depth_norm = ((depth_np - min_depth) / (max_depth - min_depth) * 65535).astype(np.uint16)
            
            # Save raw depth map
            cv2.imwrite(output_path, depth_norm)
            print(f"Saved raw depth map to {output_path}")
        
        return output_path
    
    except Exception as e:
        print(f"Error generating depth map for {image_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def batch_generate_depth_maps(input_dir, output_dir, pattern="*.png", downscale_factor=1.0, half_precision=False, colored=True, cmap='turbo'):
    """
    Generate depth maps for all matching images in a directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save output depth maps
        pattern: Glob pattern to match input images
        downscale_factor: Downscale factor for image processing
        half_precision: Use half precision for computation
        colored: Whether to save colored depth maps
        cmap: Colormap for depth visualization
        
    Returns:
        Number of successfully processed images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all matching images
    input_pattern = os.path.join(input_dir, pattern)
    image_paths = sorted(glob.glob(input_pattern))
    
    if not image_paths:
        print(f"No images found matching pattern {input_pattern}")
        return 0
    
    print(f"Found {len(image_paths)} images to process")
    
    # Process each image
    successful = 0
    
    for i, image_path in enumerate(image_paths):
        # Construct output path
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_depth.png")
        
        print(f"[{i+1}/{len(image_paths)}] Processing {base_name}")
        
        # Generate depth map
        result = generate_depth_map(
            image_path=image_path,
            output_path=output_path,
            downscale_factor=downscale_factor,
            half_precision=half_precision,
            colored=colored,
            cmap=cmap
        )
        
        if result is not None:
            successful += 1
    
    print(f"Processing complete: {successful}/{len(image_paths)} images successfully processed")
    return successful

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate depth maps from input images")
    
    # Input/output arguments
    parser.add_argument("--input_dir", type=str, default="./TEMP/FRAMES",
                        help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="./TMP/DEPTH",
                        help="Directory to save output depth maps")
    parser.add_argument("--pattern", type=str, default="*.png",
                        help="Glob pattern to match input images")
    
    # Processing parameters
    parser.add_argument("--downscale_factor", type=float, default=1.0,
                        help="Downscale input images for faster processing")
    parser.add_argument("--half_precision", action="store_true",
                        help="Use float16 for faster computation")
    
    # Visualization parameters
    parser.add_argument("--raw", action="store_true",
                        help="Save raw depth maps (grayscale) instead of colored ones")
    parser.add_argument("--colormap", type=str, default="turbo",
                        choices=["turbo", "viridis", "plasma", "inferno", "magma", "cividis", "jet"],
                        help="Colormap for depth visualization")
    
    args = parser.parse_args()
    
    # Check if using Apple Silicon
    if is_apple_silicon():
        print("Apple Silicon detected, enabling optimizations...")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        # Half precision is better for MPS
        if args.half_precision:
            print("Using half precision for better performance on Apple Silicon")
    
    # Process all images
    batch_generate_depth_maps(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        downscale_factor=args.downscale_factor,
        half_precision=args.half_precision,
        colored=not args.raw,
        cmap=args.colormap
    ) 