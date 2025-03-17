#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import torch
import time
import json
from pathlib import Path
import multiprocessing
import queue
import threading
import signal
import sys

# Import functions from the three scripts
from img_to_normalized_pointcloud import (
    create_normalized_pointcloud,
    load_ground_plane_params,
    save_ground_plane_params,
    get_torch_device,
    render_point_cloud_to_image,
    is_apple_silicon
)

from pointcloud_cleaner import (
    get_normalized_pointcloud,
    remove_stray_points,
    clean_shadows
)

from simple_pointcloud_viewer import (
    fit_shapes_to_clusters,
    export_shape_data
)

# Global variable for graceful shutdown
stop_processing = False

def signal_handler(sig, frame):
    """Handle Ctrl+C for graceful shutdown"""
    global stop_processing
    print("\nInterrupted! Finishing current tasks and shutting down...")
    stop_processing = True

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

def in_memory_pointcloud_visualization(pcd, output_path, height_threshold=None, 
                                     point_size=2, dpi=300, max_points=50000,
                                     tight_crop=True, fit_shapes=False, cluster_eps=0.2, 
                                     min_cluster_size=5, circularity_threshold=0.85,
                                     output_all_files=True):
    """
    A version of simple_pointcloud_visualization that works with in-memory pointcloud objects
    
    Args:
        pcd: Open3D point cloud object
        output_path: Path to save the output image
        height_threshold: Only show points above this height (meters)
        point_size: Size of points in visualization
        dpi: DPI for output image
        max_points: Maximum number of points to visualize (downsamples if more)
        tight_crop: Whether to crop view tightly around visible points
        fit_shapes: Whether to fit rectangles and circles to point clusters
        cluster_eps: DBSCAN epsilon parameter (max distance between points in a cluster)
        min_cluster_size: Minimum points for a valid cluster
        circularity_threshold: Threshold for circularity (0-1)
        output_all_files: Whether to output additional files (shapes, floor plan)
    
    Returns:
        True if successful, False otherwise
    """
    import numpy as np
    import open3d as o3d
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import cv2
    from sklearn.cluster import DBSCAN
    
    print(f"Creating floor plan visualization at {output_path}...")
    
    # Get points and colors from the pointcloud
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    if len(points) < min_cluster_size:
        print(f"Not enough points to create visualization (only {len(points)})")
        return False
    
    # Assume the pointcloud has been normalized, so floor is at y=0
    normalized_points = points
    
    # Filter points by height if threshold is provided
    if height_threshold is not None:
        print(f"Filtering points above {height_threshold}m from floor")
        height_mask = normalized_points[:, 1] >= height_threshold
        vis_points = normalized_points[height_mask]
        vis_colors = colors[height_mask] if len(colors) > 0 else None
        title = f"Points Above {height_threshold}m From Floor"
    else:
        vis_points = normalized_points
        vis_colors = colors if len(colors) > 0 else None
        title = "All Points (Top-Down View)"
    
    print(f"Visualizing {len(vis_points)} points")
    
    # Extract X and Z coordinates (top-down view)
    x = vis_points[:, 0].copy()
    z = vis_points[:, 2].copy()
    
    # Flip X coordinates to match original PLY orientation
    x = -x  # Flip the X axis to match expected orientation
    
    # Create 2D point array for shape fitting
    points_2d = np.column_stack((x, z))
    
    # Fit shapes if requested
    rectangles = []
    circles = []
    if fit_shapes:
        # For shape fitting, use all points 
        # (but still reasonable to limit to prevent slowdown)
        shape_points = points_2d
        if len(shape_points) > 100000:
            # Use more points for shape fitting than visualization
            indices = np.random.choice(len(shape_points), 100000, replace=False)
            shape_points = shape_points[indices]
        
        print("Fitting shapes to point clusters...")
        rectangles, circles = fit_shapes_to_clusters(
            shape_points, 
            eps=cluster_eps, 
            min_samples=min_cluster_size,
            circularity_threshold=circularity_threshold
        )
        print(f"Fitted {len(rectangles)} rectangles and {len(circles)} circles")
        
        # Append shape info to title
        title += f" ({len(rectangles)} rectangles, {len(circles)} circles)"
    
    # Downsample for visualization if needed
    if len(vis_points) > max_points:
        # Sample a subset of points
        print(f"Downsampling from {len(vis_points)} to {max_points} points for visualization")
        indices = np.random.choice(len(vis_points), max_points, replace=False)
        vis_points = vis_points[indices]
        vis_colors = vis_colors[indices] if vis_colors is not None else None
        # Recalculate x, z after downsampling
        x = -vis_points[:, 0]
        z = vis_points[:, 2]
    
    # Create visualization (top-down view)
    plt.figure(figsize=(12, 12))
    
    # Color by height or use original colors
    if len(vis_colors) > 0:
        # Use original colors
        rgba_colors = np.zeros((len(vis_colors), 4))
        rgba_colors[:, 0:3] = vis_colors
        rgba_colors[:, 3] = 0.5 if fit_shapes else 0.8  # More transparent if showing shapes
        plt.scatter(x, z, s=point_size, c=rgba_colors)
    else:
        # Color by height with high contrast
        plt.scatter(x, z, s=point_size, c=vis_points[:, 1], cmap='jet', 
                   alpha=0.5 if fit_shapes else 0.8)
        plt.colorbar(label='Height (m)')
    
    # Draw fitted shapes if requested
    if fit_shapes:
        ax = plt.gca()
        
        # Use a nice color palette for different shapes
        rectangle_colors = ['#4285F4', '#34A853', '#FBBC05', '#EA4335', 
                           '#8E44AD', '#16A085', '#D35400', '#7F8C8D']
        circle_colors = ['#3498DB', '#2ECC71', '#F1C40F', '#E74C3C', 
                        '#9B59B6', '#1ABC9C', '#E67E22', '#95A5A6']
        
        # Draw rectangles
        for i, rect in enumerate(rectangles):
            color_idx = i % len(rectangle_colors)
            center_x, center_y, width, height, angle = rect
            rect_patch = patches.Rectangle(
                (0, 0), width, height, 
                angle=angle,
                linewidth=2,
                edgecolor=rectangle_colors[color_idx],
                facecolor='none',
                alpha=0.9
            )
            
            # Move rectangle to center position
            transform = (
                patches.transforms.Affine2D()
                .rotate_deg(-angle)
                .translate(-width/2, -height/2)
                .rotate_deg(angle)
                .translate(center_x, center_y)
                + ax.transData
            )
            rect_patch.set_transform(transform)
            ax.add_patch(rect_patch)
            
            # Add a numerical label to the shape
            plt.text(center_x, center_y, str(i+1), 
                   ha='center', va='center', 
                   color=rectangle_colors[color_idx], fontsize=10, fontweight='bold',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
        
        # Draw circles
        for i, circle in enumerate(circles):
            color_idx = i % len(circle_colors)
            center_x, center_y, radius = circle
            circle_patch = patches.Circle(
                (center_x, center_y), radius,
                linewidth=2,
                edgecolor=circle_colors[color_idx],
                facecolor='none',
                alpha=0.9
            )
            ax.add_patch(circle_patch)
            
            # Add a numerical label to the shape
            label_num = len(rectangles) + i + 1
            plt.text(center_x, center_y, str(label_num), 
                   ha='center', va='center', 
                   color=circle_colors[color_idx], fontsize=10, fontweight='bold',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
    
    # Set title and labels
    plt.title(title)
    plt.xlabel('X (meters)')
    plt.ylabel('Z (meters)')
    
    # Set equal aspect ratio and grid
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add a border to make the points stand out
    plt.gca().set_facecolor('#f0f0f0')  # Light gray background
    
    # Set tight crop around visible points if requested
    if tight_crop:
        # Add a small padding (10% of range)
        padding_x = 0.1 * (np.max(x) - np.min(x))
        padding_z = 0.1 * (np.max(z) - np.min(z))
        plt.xlim(np.min(x) - padding_x, np.max(x) + padding_x)
        plt.ylim(np.min(z) - padding_z, np.max(z) + padding_z)
    
    # Save the image
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    
    # Save additional files only if requested
    if fit_shapes and output_all_files:
        # Create a clean floor plan with just the shapes
        shape_output_path = os.path.splitext(output_path)[0] + "_shapes.png"
        plt.figure(figsize=(12, 12))
        ax = plt.gca()
        
        # Use a nice color palette for different shapes
        rectangle_colors = ['#4285F4', '#34A853', '#FBBC05', '#EA4335', 
                           '#8E44AD', '#16A085', '#D35400', '#7F8C8D']
        circle_colors = ['#3498DB', '#2ECC71', '#F1C40F', '#E74C3C', 
                        '#9B59B6', '#1ABC9C', '#E67E22', '#95A5A6']
        
        # Draw rectangles
        for i, rect in enumerate(rectangles):
            color_idx = i % len(rectangle_colors)
            center_x, center_y, width, height, angle = rect
            rect_patch = patches.Rectangle(
                (0, 0), width, height, 
                angle=angle,
                linewidth=2,
                edgecolor='black',
                facecolor=rectangle_colors[color_idx],
                alpha=0.7
            )
            
            # Move rectangle to center position
            transform = (
                patches.transforms.Affine2D()
                .rotate_deg(-angle)
                .translate(-width/2, -height/2)
                .rotate_deg(angle)
                .translate(center_x, center_y)
                + ax.transData
            )
            rect_patch.set_transform(transform)
            ax.add_patch(rect_patch)
            
            # Add a numerical label to the shape
            plt.text(center_x, center_y, str(i+1), 
                    ha='center', va='center', 
                    color='white', fontsize=10, fontweight='bold')
            
            # Add dimensions if large enough to display
            if width > 0.3 and height > 0.3:  # Only label larger rectangles
                size_text = f"{width:.2f}×{height:.2f}m"
                plt.text(center_x, center_y + 0.15, size_text, 
                        ha='center', va='center', 
                        color='white', fontsize=8)
        
        # Draw circles
        for i, circle in enumerate(circles):
            color_idx = i % len(circle_colors)
            center_x, center_y, radius = circle
            circle_patch = patches.Circle(
                (center_x, center_y), radius,
                linewidth=2,
                edgecolor='black',
                facecolor=circle_colors[color_idx],
                alpha=0.7
            )
            ax.add_patch(circle_patch)
            
            # Add a numerical label to the shape
            label_num = len(rectangles) + i + 1
            plt.text(center_x, center_y, str(label_num), 
                    ha='center', va='center', 
                    color='white', fontsize=10, fontweight='bold')
            
            # Add radius if large enough to display
            if radius > 0.2:  # Only label larger circles
                size_text = f"r={radius:.2f}m"
                plt.text(center_x, center_y + 0.1, size_text, 
                        ha='center', va='center', 
                        color='white', fontsize=8)
        
        # Calculate total area covered by shapes
        total_rect_area = sum(rect[2] * rect[3] for rect in rectangles)
        total_circle_area = sum(np.pi * circ[2]**2 for circ in circles)
        total_area = total_rect_area + total_circle_area
            
        # Set title and labels
        plt.title(f"Floor Plan - {len(rectangles)} Rectangles, {len(circles)} Circles (Total Area: {total_area:.2f}m²)")
        plt.xlabel('X (meters)')
        plt.ylabel('Z (meters)')
        
        # Set equal aspect ratio and grid
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.4)
        
        # Use the same axis limits as the point visualization
        if tight_crop:
            plt.xlim(np.min(x) - padding_x, np.max(x) + padding_x)
            plt.ylim(np.min(z) - padding_z, np.max(z) + padding_z)
        
        # Set a light background color
        ax.set_facecolor('#f8f9fa')
        
        # Add a scale bar (1 meter)
        bar_x = np.min(x) + padding_x * 2
        bar_y = np.min(z) + padding_z * 2
        plt.plot([bar_x, bar_x + 1.0], [bar_y, bar_y], 'k-', linewidth=3)
        plt.text(bar_x + 0.5, bar_y - 0.1, '1 meter', ha='center', va='top')
        
        # Save the shape-only image
        plt.tight_layout()
        plt.savefig(shape_output_path, dpi=dpi)
        plt.close()
        
        # Also create a simplified floor plan with filled shapes
        floor_plan_path = os.path.splitext(output_path)[0] + "_floor_plan.png"
        fig, ax = plt.subplots(figsize=(12, 12), facecolor='white')
        
        # Set a white background
        ax.set_facecolor('white')
        
        # Draw rectangles with solid fill
        for i, rect in enumerate(rectangles):
            center_x, center_y, width, height, angle = rect
            rect_patch = patches.Rectangle(
                (0, 0), width, height, 
                angle=angle,
                linewidth=1.5,
                edgecolor='black',
                facecolor='lightgray',
                alpha=1.0
            )
            
            # Move rectangle to center position
            transform = (
                patches.transforms.Affine2D()
                .rotate_deg(-angle)
                .translate(-width/2, -height/2)
                .rotate_deg(angle)
                .translate(center_x, center_y)
                + ax.transData
            )
            rect_patch.set_transform(transform)
            ax.add_patch(rect_patch)
        
        # Draw circles with solid fill
        for circle in circles:
            center_x, center_y, radius = circle
            circle_patch = patches.Circle(
                (center_x, center_y), radius,
                linewidth=1.5,
                edgecolor='black',
                facecolor='lightgray',
                alpha=1.0
            )
            ax.add_patch(circle_patch)
        
        # Set equal aspect ratio
        plt.axis('equal')
        plt.axis('off')  # Hide axes
        
        # Use the same axis limits as the point visualization
        if tight_crop:
            plt.xlim(np.min(x) - padding_x, np.max(x) + padding_x)
            plt.ylim(np.min(z) - padding_z, np.max(z) + padding_z)
        
        # Save the floor plan image
        plt.tight_layout()
        plt.savefig(floor_plan_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"Floor plan saved to: {floor_plan_path}")
        
        # Export shape data to a text file
        data_output_path = os.path.splitext(output_path)[0] + "_shapes.txt"
        export_shape_data(rectangles, circles, data_output_path)
        
        print(f"Shape-only floor plan saved to: {shape_output_path}")
    
    print(f"Visualization saved to: {output_path}")
    return True

def process_single_frame(
    image_path, 
    output_dir, 
    ground_model,
    ground_params_dir,
    height_threshold=1.3,
    point_size=10,
    downscale_factor=1.0,
    half_precision=False,
    fit_shapes=True,
    visualize_3d=False,
    simple_output=False,
    output_all_files=True
):
    """
    Process a single frame from image to floor plan
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save output files
        ground_model: Pre-loaded ground model or None
        ground_params_dir: Directory containing ground.json
        height_threshold: Only show points above this height (meters)
        point_size: Size of points in visualization
        downscale_factor: Downscale factor for image processing
        half_precision: Use float16 for faster computation
        fit_shapes: Fit shapes to the point cloud
        visualize_3d: Create 3D point cloud visualizations
        simple_output: Output simple visualization without shapes or labels
        output_all_files: Whether to output additional files (shapes, floor plan)
        
    Returns:
        success: True if processing was successful, False otherwise
        processing_time: Time taken to process the frame in seconds
    """
    start_time = time.time()
    
    try:
        base_name = os.path.basename(image_path)
        frame_name = os.path.splitext(base_name)[0]
        
        # STEP 1: Generate normalized point cloud using the ground model
        normalized_pcd, colors = get_normalized_pointcloud(
            image_path=image_path,
            grid_size=20,
            ground_percentile=5,
            downscale_factor=downscale_factor,
            half_precision=half_precision,
            rotation_offset=None,
            ground_params_dir=ground_params_dir  # Will use cached ground.json
        )
        
        if normalized_pcd is None:
            print(f"Failed to generate point cloud for {base_name}. Skipping.")
            return False, time.time() - start_time
        
        # STEP 2: Clean the point cloud
        print(f"[{frame_name}] Removing stray points...")
        cleaned_pcd = remove_stray_points(normalized_pcd, nb_points=20, radius=0.1)
        
        print(f"[{frame_name}] Cleaning shadow artifacts...")
        cleaned_pcd = clean_shadows(cleaned_pcd)
        
        # Generate 3D visualization if requested
        if visualize_3d:
            viz_output_path = os.path.join(output_dir, f"{frame_name}_pcd_preview.png")
            render_point_cloud_to_image(
                cleaned_pcd, 
                viz_output_path, 
                width=1280, 
                height=720, 
                background_color=[1, 1, 1], 
                view_preset="front"
            )
        
        # STEP 3: Create floor plan visualization
        floor_plan_output = os.path.join(output_dir, f"{frame_name}_clean_simple_view.png")
        
        # Use our in-memory function directly
        print(f"[{frame_name}] Creating floor plan...")
        in_memory_pointcloud_visualization(
            pcd=cleaned_pcd,
            output_path=floor_plan_output,
            height_threshold=height_threshold,
            point_size=point_size,
            dpi=300,
            max_points=50000,
            tight_crop=True,
            fit_shapes=False if simple_output else fit_shapes,
            cluster_eps=0.2,
            min_cluster_size=5,
            circularity_threshold=0.85,
            output_all_files=output_all_files
        )
        
        return True, time.time() - start_time
    
    except Exception as e:
        import traceback
        print(f"Error processing {os.path.basename(image_path)}: {str(e)}")
        traceback.print_exc()
        return False, time.time() - start_time

def worker_process(task_queue, result_queue, ground_params_dir, args):
    """
    Worker process to handle frame processing
    
    Args:
        task_queue: Queue of image paths to process
        result_queue: Queue to return results
        ground_params_dir: Directory containing ground.json
        args: Command line arguments
    """
    # Initialize device
    device = get_torch_device()
    print(f"Worker initialized with device: {device}")
    
    # Load ground model (shared across all frames)
    try:
        ground_model = load_ground_plane_params("dummy_path.png", ground_params_dir)
        print("Ground plane loaded in worker")
    except Exception as e:
        print(f"Error loading ground plane in worker: {e}")
        ground_model = None
    
    # Process frames from the queue
    while True:
        try:
            # Get task from queue with timeout to allow checking for stop signal
            try:
                image_path, frame_index, total_frames = task_queue.get(timeout=1)
            except queue.Empty:
                # Check if we should stop
                if stop_processing:
                    break
                continue
            
            if stop_processing:
                break
                
            # Process the frame
            print(f"[{frame_index+1}/{total_frames}] Processing {os.path.basename(image_path)}")
            success, processing_time = process_single_frame(
                image_path=image_path,
                output_dir=args.output_dir,
                ground_model=ground_model,
                ground_params_dir=ground_params_dir,
                height_threshold=args.height_threshold,
                point_size=args.point_size,
                downscale_factor=args.downscale_factor,
                half_precision=args.half_precision,
                fit_shapes=args.fit_shapes,
                visualize_3d=args.visualize_3d,
                simple_output=args.simple_output,
                output_all_files=args.output_all_files
            )
            
            # Send result back
            result_queue.put((frame_index, image_path, success, processing_time))
            
            # Mark task as done
            task_queue.task_done()
            
        except Exception as e:
            import traceback
            print(f"Error in worker: {str(e)}")
            traceback.print_exc()
            break
    
    print("Worker process exiting")

def process_images_to_floor_plans(
    frames_dir,
    output_dir=None,
    height_threshold=1.3,
    point_size=10,
    downscale_factor=1.0,
    half_precision=False,
    start_frame=None,
    end_frame=None,
    pattern="output_*.png",
    fit_shapes=True,
    visualize_3d=False,
    num_workers=1,
    simple_output=False,
    output_all_files=True
):
    """
    Process a directory of image frames into floor plans using the most efficient pipeline
    
    Args:
        frames_dir: Directory containing image frames
        output_dir: Directory to save output files (defaults to frames_dir)
        height_threshold: Only show points above this height (meters)
        point_size: Size of points in visualization
        downscale_factor: Downscale factor for image processing
        half_precision: Use float16 for faster computation
        start_frame: First frame to process (number in filename)
        end_frame: Last frame to process (number in filename)
        pattern: Pattern to match image files
        fit_shapes: Fit rectangular and circular shapes to the point cloud
        visualize_3d: Whether to create 3D point cloud visualizations
        num_workers: Number of parallel workers (0 for sequential processing)
        simple_output: Output simple visualization without shapes or labels
        output_all_files: Whether to output additional files (shapes, floor plan)
    """
    global stop_processing
    stop_processing = False
    
    # Set default output directory
    if output_dir is None:
        output_dir = frames_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all image files in the frames directory
    image_path_pattern = os.path.join(frames_dir, pattern)
    image_paths = sorted(glob.glob(image_path_pattern))
    
    if not image_paths:
        print(f"No images found matching pattern {image_path_pattern}")
        return
    
    # Filter by frame range if specified
    if start_frame is not None or end_frame is not None:
        filtered_paths = []
        for path in image_paths:
            # Extract frame number from filename
            base_name = os.path.basename(path)
            try:
                frame_num = int(''.join(filter(str.isdigit, base_name)))
                
                # Check if frame is in specified range
                if (start_frame is None or frame_num >= start_frame) and \
                (end_frame is None or frame_num <= end_frame):
                    filtered_paths.append(path)
            except ValueError:
                # Skip files that don't have a numeric part in the name
                continue
        
        image_paths = filtered_paths
    
    if not image_paths:
        print("No images to process after applying frame range filters")
        return
    
    total_frames = len(image_paths)
    print(f"Processing {total_frames} images")
    
    # Check for existing ground plane parameters
    ground_params_path = os.path.join(frames_dir, "ground.json")
    ground_model = None
    
    if os.path.exists(ground_params_path):
        try:
            print(f"Loading ground plane parameters from {ground_params_path}")
            ground_model = load_ground_plane_params(image_paths[0], frames_dir)
            print("Ground plane loaded successfully")
        except Exception as e:
            print(f"Error loading ground plane: {e}")
            ground_model = None
    
    # If no ground model exists, calculate it from the first frame
    if ground_model is None:
        print("Calculating ground plane from first frame...")
        # Create normalized point cloud and save ground model
        first_image = image_paths[0]
        normalized_pcd, colors = get_normalized_pointcloud(
            image_path=first_image,
            grid_size=20,
            ground_percentile=5,
            downscale_factor=downscale_factor,
            half_precision=half_precision,
            rotation_offset=None,
            ground_params_dir=frames_dir  # This will save ground.json
        )
        
        # Load the ground model we just saved
        ground_model = load_ground_plane_params(first_image, frames_dir)
    
    # Process frames
    start_time = time.time()
    
    # Decide whether to use parallel or sequential processing
    if num_workers > 1 and total_frames > 1:
        print(f"Using {num_workers} parallel workers")
        
        # Create queues for tasks and results
        task_queue = multiprocessing.JoinableQueue()
        result_queue = multiprocessing.Queue()
        
        # Create and start worker processes
        workers = []
        for _ in range(num_workers):
            p = multiprocessing.Process(
                target=worker_process,
                args=(task_queue, result_queue, frames_dir, argparse.Namespace(
                    output_dir=output_dir,
                    height_threshold=height_threshold,
                    point_size=point_size,
                    downscale_factor=downscale_factor,
                    half_precision=half_precision,
                    fit_shapes=fit_shapes,
                    visualize_3d=visualize_3d,
                    simple_output=simple_output,
                    output_all_files=output_all_files
                ))
            )
            p.daemon = True
            p.start()
            workers.append(p)
        
        # Fill task queue with image paths
        for i, image_path in enumerate(image_paths):
            task_queue.put((image_path, i, total_frames))
        
        # Setup progress tracking
        completed_frames = 0
        processing_times = []
        
        # Process results as they come in
        try:
            while completed_frames < total_frames and not stop_processing:
                try:
                    # Get result with timeout to allow checking for stop signal
                    frame_index, image_path, success, processing_time = result_queue.get(timeout=1)
                    completed_frames += 1
                    
                    # Update progress
                    if success:
                        processing_times.append(processing_time)
                        avg_time = sum(processing_times) / len(processing_times)
                        remaining_frames = total_frames - completed_frames
                        estimated_remaining_time = avg_time * remaining_frames
                        
                        print(f"Progress: {completed_frames}/{total_frames} frames, " 
                            f"Avg: {avg_time:.2f}s/frame, "
                            f"Remaining: {estimated_remaining_time/60:.1f} minutes")
                except queue.Empty:
                    continue
        
        except KeyboardInterrupt:
            stop_processing = True
            print("\nInterrupted! Finishing current tasks and shutting down...")
        
        # Close down
        for worker in workers:
            worker.join(timeout=5)
        
        # Terminate any remaining workers
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
    
    else:
        # Sequential processing (better for debugging)
        print("Using sequential processing")
        
        for idx, image_path in enumerate(image_paths):
            if stop_processing:
                break
                
            base_name = os.path.basename(image_path)
            print(f"[{idx+1}/{total_frames}] Processing {base_name}")
            frame_start_time = time.time()
            
            success, processing_time = process_single_frame(
                image_path=image_path,
                output_dir=output_dir,
                ground_model=ground_model,
                ground_params_dir=frames_dir,
                height_threshold=height_threshold,
                point_size=point_size,
                downscale_factor=downscale_factor,
                half_precision=half_precision,
                fit_shapes=fit_shapes,
                visualize_3d=visualize_3d,
                simple_output=simple_output,
                output_all_files=output_all_files
            )
            
            # Estimate remaining time
            elapsed_time = time.time() - start_time
            frames_processed = idx + 1
            avg_time_per_frame = elapsed_time / frames_processed
            remaining_frames = total_frames - frames_processed
            estimated_remaining_time = avg_time_per_frame * remaining_frames
            
            print(f"Progress: {frames_processed}/{total_frames} frames, " 
                f"Avg: {avg_time_per_frame:.2f}s/frame, "
                f"Remaining: {estimated_remaining_time/60:.1f} minutes")
    
    total_time = time.time() - start_time
    print(f"Processing complete! {total_frames} frames processed in {total_time/60:.2f} minutes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process image frames to floor plans in a single pipeline")
    
    # Input/output arguments
    parser.add_argument("--frames_dir", type=str, default="./TEMP/FRAMES",
                       help="Directory containing image frames")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save output files (defaults to frames_dir)")
                       
    # Frame selection arguments
    parser.add_argument("--pattern", type=str, default="output_*.png",
                       help="Pattern to match image files")
    parser.add_argument("--start_frame", type=int, default=None,
                       help="First frame to process (number in filename)")
    parser.add_argument("--end_frame", type=int, default=None,
                       help="Last frame to process (number in filename)")
    
    # Visualization parameters
    parser.add_argument("--height_threshold", type=float, default=1.3,
                       help="Only show points above this height (meters)")
    parser.add_argument("--point_size", type=int, default=10,
                       help="Size of points in floor plan visualization")
    parser.add_argument("--fit_shapes", action="store_true", default=True,
                       help="Fit rectangular and circular shapes to the point cloud")
    parser.add_argument("--visualize_3d", action="store_true",
                       help="Create 3D point cloud visualizations")
    
    # Output options
    parser.add_argument("--simple_output", action="store_true", 
                       help="Output simple visualization without shapes or labels")
    parser.add_argument("--output_main_only", action="store_true",
                       help="Only output the main visualization file (no shapes or floor plan files)")
    
    # Processing parameters
    parser.add_argument("--downscale_factor", type=float, default=1.0,
                       help="Downscale input images for faster processing")
    parser.add_argument("--half_precision", action="store_true",
                       help="Use float16 for faster computation")
    parser.add_argument("--num_threads", type=int, default=None,
                       help="Number of threads to use for processing (defaults to number of CPU cores)")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Number of parallel worker processes (0 for sequential processing)")
    
    args = parser.parse_args()
    
    # Set number of threads for numpy and OpenMP if specified
    if args.num_threads is not None:
        try:
            import os
            os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
            os.environ["OPENBLAS_NUM_THREADS"] = str(args.num_threads)
            os.environ["MKL_NUM_THREADS"] = str(args.num_threads)
            os.environ["VECLIB_MAXIMUM_THREADS"] = str(args.num_threads)
            os.environ["NUMEXPR_NUM_THREADS"] = str(args.num_threads)
            print(f"Set thread limit to {args.num_threads}")
        except Exception as e:
            print(f"Warning: Failed to set thread limit: {e}")
    
    try:
        # Using Apple Silicon MPS or other accelerators requires setting this early
        if is_apple_silicon():
            print("Apple Silicon detected, enabling optimizations...")
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            # Force use of MPS for matrices on M1/M2 Macs
            if args.half_precision:
                # Half precision is better for MPS
                print("Using half precision for better performance on Apple Silicon")
                
        process_images_to_floor_plans(
            frames_dir=args.frames_dir,
            output_dir=args.output_dir,
            height_threshold=args.height_threshold,
            point_size=args.point_size,
            downscale_factor=args.downscale_factor,
            half_precision=args.half_precision,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            pattern=args.pattern,
            fit_shapes=False if args.simple_output else args.fit_shapes,
            visualize_3d=args.visualize_3d,
            num_workers=args.num_workers,
            simple_output=args.simple_output,
            output_all_files=not args.output_main_only
        )
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    except Exception as e:
        import traceback
        print(f"Error during processing: {e}")
        traceback.print_exc() 