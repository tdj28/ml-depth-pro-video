from PIL import Image
import depth_pro
import numpy as np
import torch
import open3d as o3d
import cv2
import copy
import json
import os
from scipy import ndimage
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

def save_ground_plane_params(ground_model, image_path, output_dir=None):
    """
    Save ground plane parameters to a JSON file.
    
    Args:
        ground_model: Dictionary containing ground plane parameters
        image_path: Path to the input image (used to generate the output filename)
        output_dir: Directory to save the JSON file (defaults to same directory as image)
        
    Returns:
        json_path: Path to the saved JSON file
    """
    if ground_model is None:
        print("No ground model to save")
        return None
    
    # Create a serializable version of the ground model
    serializable_model = {
        'normal': ground_model['normal'].tolist() if isinstance(ground_model['normal'], np.ndarray) else ground_model['normal'],
        'd': float(ground_model['d']),
        'origin': ground_model['origin'].tolist() if isinstance(ground_model['origin'], np.ndarray) else ground_model['origin']
    }
    
    # Generate output filename based on input image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output path
    json_path = os.path.join(output_dir, f"{base_name}_ground_plane.json")
    
    # Save to JSON file
    with open(json_path, 'w') as f:
        json.dump(serializable_model, f, indent=4)
    
    print(f"Ground plane parameters saved to {json_path}")
    return json_path

def load_ground_plane_params(image_path, input_dir=None):
    """
    Load ground plane parameters from a JSON file.
    
    Args:
        image_path: Path to the input image (used to find the JSON file)
        input_dir: Directory to look for the JSON file (defaults to same directory as image)
        
    Returns:
        ground_model: Dictionary containing ground plane parameters, or None if file not found
    """
    # Generate input filename based on input image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    if input_dir is None:
        input_dir = os.path.dirname(image_path)
    
    # Create input path
    json_path = os.path.join(input_dir, f"{base_name}_ground_plane.json")
    
    # Check if file exists
    if not os.path.exists(json_path):
        print(f"No saved ground plane parameters found at {json_path}")
        return None
    
    # Load from JSON file
    try:
        with open(json_path, 'r') as f:
            loaded_model = json.load(f)
        
        # Convert lists back to numpy arrays
        ground_model = {
            'normal': np.array(loaded_model['normal']),
            'd': loaded_model['d'],
            'origin': np.array(loaded_model['origin']),
            'inliers_3d': None  # We don't save inliers
        }
        
        print(f"Ground plane parameters loaded from {json_path}")
        return ground_model
    except Exception as e:
        print(f"Error loading ground plane parameters: {e}")
        return None

def apply_rotation_to_plane(ground_model, rotation_offset):
    """
    Apply rotation offsets to the ground plane normal.
    
    Args:
        ground_model: Dictionary containing ground plane parameters
        rotation_offset: List of rotation angles in degrees [x_rot, y_rot, z_rot]
        
    Returns:
        updated_model: Dictionary with updated ground plane parameters
    """
    if ground_model is None:
        return None
    
    # Convert rotation angles from degrees to radians
    x_rot, y_rot, z_rot = np.radians(rotation_offset)
    
    # Create rotation matrices for each axis
    # Rotation around X axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(x_rot), -np.sin(x_rot)],
        [0, np.sin(x_rot), np.cos(x_rot)]
    ])
    
    # Rotation around Y axis
    Ry = np.array([
        [np.cos(y_rot), 0, np.sin(y_rot)],
        [0, 1, 0],
        [-np.sin(y_rot), 0, np.cos(y_rot)]
    ])
    
    # Rotation around Z axis
    Rz = np.array([
        [np.cos(z_rot), -np.sin(z_rot), 0],
        [np.sin(z_rot), np.cos(z_rot), 0],
        [0, 0, 1]
    ])
    
    # Combine rotations
    R = Rz @ Ry @ Rx
    
    # Apply rotation to normal vector
    normal = ground_model['normal']
    rotated_normal = R @ normal
    
    # Normalize the rotated normal
    rotated_normal = rotated_normal / np.linalg.norm(rotated_normal)
    
    # Create updated model
    updated_model = ground_model.copy()
    updated_model['normal'] = rotated_normal
    
    # Recalculate d in plane equation ax + by + cz + d = 0
    # d = -n·p where p is a point on the plane
    updated_model['d'] = -np.dot(rotated_normal, ground_model['origin'])
    
    print(f"Applied rotation offset: {rotation_offset} degrees")
    print(f"Updated normal: [{rotated_normal[0]:.4f}, {rotated_normal[1]:.4f}, {rotated_normal[2]:.4f}]")
    
    return updated_model

def optimize_ground_plane(points_3d, ground_model, min_points_above=0.95, visualize=False):
    """
    Optimize the ground plane orientation to ensure no points lie below the ground
    and the distance from the ground to the nearest points is minimized.
    
    Args:
        points_3d: Nx3 array of 3D points
        ground_model: Dictionary containing ground plane parameters
        min_points_above: Minimum fraction of points that must be above the ground
        visualize: Whether to visualize the optimization process
        
    Returns:
        optimized_model: Dictionary with optimized ground plane parameters
    """
    if ground_model is None or len(points_3d) == 0:
        print("No ground model or points available for optimization")
        return ground_model
    
    print(f"Optimizing ground plane with {len(points_3d)} points...")
    
    # Get initial parameters
    initial_normal = ground_model['normal']
    initial_d = ground_model['d']
    initial_origin = ground_model['origin']
    
    # Ensure normal is normalized
    initial_normal = initial_normal / np.linalg.norm(initial_normal)
    
    # First, identify likely ground points
    # These are points that are close to the initial ground plane
    initial_distances = np.abs(np.dot(points_3d, initial_normal) + initial_d)
    
    # Points within a small distance of the plane are likely ground points
    # Use a distance threshold based on the scene scale (median depth)
    median_depth = np.median(points_3d[:, 2])
    ground_threshold = 0.05 * median_depth  # 5% of median depth
    likely_ground_indices = initial_distances < ground_threshold
    
    # Also consider points in the lower part of the scene as likely ground
    # Sort points by Y coordinate (height in camera space)
    y_sorted_indices = np.argsort(points_3d[:, 1])
    
    # Take the lowest 20% of points as potential ground points
    lowest_count = max(int(0.2 * len(points_3d)), 100)
    lowest_indices = y_sorted_indices[:lowest_count]
    
    # Combine both criteria
    ground_candidate_indices = np.union1d(
        np.where(likely_ground_indices)[0],
        lowest_indices
    )
    
    # Get the ground candidate points
    ground_candidates = points_3d[ground_candidate_indices]
    
    print(f"Identified {len(ground_candidates)} potential ground points out of {len(points_3d)} total points")
    
    # Convert normal to spherical coordinates for optimization
    # This ensures the normal remains unit length during optimization
    theta = np.arccos(initial_normal[1])  # Angle from Y axis
    phi = np.arctan2(initial_normal[2], initial_normal[0])  # Angle in XZ plane
    
    # Initial parameters for optimization: [theta, phi, d_offset]
    # d_offset is an offset to the plane distance
    initial_params = [theta, phi, 0.0]
    
    # Function to convert optimization parameters to plane parameters
    def params_to_plane(params):
        theta, phi, d_offset = params
        
        # Convert spherical to Cartesian coordinates for normal
        normal = np.array([
            np.sin(theta) * np.cos(phi),
            np.cos(theta),
            np.sin(theta) * np.sin(phi)
        ])
        
        # Ensure normal points "up" (-Y in camera space)
        if normal[1] > 0:
            normal = -normal
            
        # Calculate new d value with offset
        # We use the original d as a base and add the offset
        d = initial_d + d_offset
        
        return normal, d
    
    # Function to calculate distances from points to plane
    def point_plane_distances(normal, d, points):
        # Calculate signed distances (positive is above the plane)
        return np.dot(points, normal) + d
    
    # Objective function to minimize
    def objective(params):
        normal, d = params_to_plane(params)
        
        # Calculate signed distances for all points
        all_distances = point_plane_distances(normal, d, points_3d)
        
        # Count points below the plane (negative distances)
        points_below = np.sum(all_distances < 0)
        fraction_below = points_below / len(points_3d)
        
        # If too many points are below the plane, heavily penalize
        if fraction_below > (1 - min_points_above):
            penalty = 1000 * fraction_below
        else:
            penalty = 0
        
        # Calculate distances for ground candidate points
        ground_distances = point_plane_distances(normal, d, ground_candidates)
        
        # We want ground points to be very close to the plane, but slightly above it
        # Penalize both negative distances (below plane) and large positive distances
        ground_penalty = 0
        
        # Penalize ground points that are below the plane (should be 0 or slightly above)
        below_ground = ground_distances < 0
        if np.any(below_ground):
            ground_penalty += 10 * np.sum(np.abs(ground_distances[below_ground]))
        
        # Penalize ground points that are too far above the plane
        above_threshold = 0.1 * median_depth  # 10% of median depth
        too_high = (ground_distances > 0) & (ground_distances > above_threshold)
        if np.any(too_high):
            ground_penalty += 5 * np.sum(ground_distances[too_high] - above_threshold)
        
        # Calculate variance of ground point distances - we want them to be consistent
        if len(ground_distances) > 1:
            # Only consider points that are close to the plane
            close_ground = np.abs(ground_distances) < above_threshold
            if np.sum(close_ground) > 1:
                variance_penalty = 10 * np.var(ground_distances[close_ground])
            else:
                variance_penalty = 0
        else:
            variance_penalty = 0
        
        # Prefer planes that are more horizontal (Y-up in world space)
        # Calculate angle with horizontal plane
        horizontal = np.array([0, -1, 0])  # Y-up in world space
        cos_angle = np.abs(np.dot(normal, horizontal))
        angle_penalty = 2 * (1 - cos_angle)  # 0 when perfectly horizontal, 2 when vertical
        
        # Combine all penalties
        total_objective = penalty + ground_penalty + variance_penalty + angle_penalty
        
        return total_objective
    
    # Constraint: allow more angular change to find the correct orientation
    max_angle_change = np.radians(30)  # Max 30 degrees change
    
    # Bounds for parameters
    bounds = [
        (max(0, theta - max_angle_change), min(np.pi, theta + max_angle_change)),  # theta
        (phi - max_angle_change, phi + max_angle_change),      # phi
        (-0.5, 0.5)                                           # d_offset (in meters)
    ]
    
    # Run optimization
    result = minimize(
        objective, 
        initial_params, 
        method='L-BFGS-B', 
        bounds=bounds,
        options={'maxiter': 200}
    )
    
    # Get optimized parameters
    optimized_normal, optimized_d = params_to_plane(result.x)
    
    # Create optimized model
    optimized_model = ground_model.copy()
    optimized_model['normal'] = optimized_normal
    optimized_model['d'] = optimized_d
    
    # Calculate statistics for reporting
    initial_distances = point_plane_distances(initial_normal, initial_d, points_3d)
    optimized_distances = point_plane_distances(optimized_normal, optimized_d, points_3d)
    
    initial_below = np.sum(initial_distances < 0) / len(points_3d)
    optimized_below = np.sum(optimized_distances < 0) / len(points_3d)
    
    # Calculate angle between initial and optimized normals
    angle = np.arccos(np.clip(np.dot(initial_normal, optimized_normal), -1.0, 1.0)) * 180 / np.pi
    
    # Calculate angle with horizontal for both planes
    horizontal = np.array([0, -1, 0])  # Y-up in world space
    initial_horizontal_angle = np.arccos(np.clip(np.dot(initial_normal, horizontal), -1.0, 1.0)) * 180 / np.pi
    optimized_horizontal_angle = np.arccos(np.clip(np.dot(optimized_normal, horizontal), -1.0, 1.0)) * 180 / np.pi
    
    print(f"Ground plane optimization results:")
    print(f"  Initial points below: {initial_below:.2%}")
    print(f"  Optimized points below: {optimized_below:.2%}")
    print(f"  Angle change: {angle:.2f} degrees")
    print(f"  Initial angle with horizontal: {initial_horizontal_angle:.2f} degrees")
    print(f"  Optimized angle with horizontal: {optimized_horizontal_angle:.2f} degrees")
    print(f"  Distance parameter change: {optimized_d - initial_d:.4f} meters")
    
    # Visualize if requested
    if visualize:
        plt.figure(figsize=(15, 10))
        
        # Plot distance histograms
        plt.subplot(221)
        plt.hist(initial_distances, bins=50, alpha=0.5, label='Initial')
        plt.hist(optimized_distances, bins=50, alpha=0.5, label='Optimized')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Signed Distance to Plane (m)')
        plt.ylabel('Count')
        plt.legend()
        plt.title('Distance Distribution Before/After Optimization')
        
        # Plot ground candidate points distances
        plt.subplot(222)
        initial_ground_distances = point_plane_distances(initial_normal, initial_d, ground_candidates)
        optimized_ground_distances = point_plane_distances(optimized_normal, optimized_d, ground_candidates)
        
        plt.hist(initial_ground_distances, bins=50, alpha=0.5, label='Initial')
        plt.hist(optimized_ground_distances, bins=50, alpha=0.5, label='Optimized')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Signed Distance to Plane (m)')
        plt.ylabel('Count')
        plt.legend()
        plt.title('Ground Candidate Points Distance Distribution')
        
        # Plot 3D visualization of planes - side view
        ax = plt.subplot(223, projection='3d')
        
        # Sample points for visualization
        sample_indices = np.random.choice(len(points_3d), min(1000, len(points_3d)), replace=False)
        sample_points = points_3d[sample_indices]
        
        # Also include ground candidate points
        ground_sample_indices = np.random.choice(len(ground_candidates), min(500, len(ground_candidates)), replace=False)
        ground_sample_points = ground_candidates[ground_sample_indices]
        
        # Plot sample points
        ax.scatter(sample_points[:, 0], sample_points[:, 2], -sample_points[:, 1], 
                  c='blue', alpha=0.3, s=1, label='All Points')
        
        # Plot ground candidate points
        ax.scatter(ground_sample_points[:, 0], ground_sample_points[:, 2], -ground_sample_points[:, 1], 
                  c='green', alpha=0.5, s=3, label='Ground Candidates')
        
        # Create a grid for the planes
        x_range = np.linspace(np.min(sample_points[:, 0]), np.max(sample_points[:, 0]), 10)
        z_range = np.linspace(np.min(sample_points[:, 2]), np.max(sample_points[:, 2]), 10)
        xx, zz = np.meshgrid(x_range, z_range)
        
        # Calculate y coordinates for initial plane
        if abs(initial_normal[1]) > 1e-6:
            yy_initial = (-initial_normal[0] * xx - initial_normal[2] * zz - initial_d) / initial_normal[1]
        else:
            yy_initial = np.zeros_like(xx)
        
        # Calculate y coordinates for optimized plane
        if abs(optimized_normal[1]) > 1e-6:
            yy_optimized = (-optimized_normal[0] * xx - optimized_normal[2] * zz - optimized_d) / optimized_normal[1]
        else:
            yy_optimized = np.zeros_like(xx)
        
        # Plot the planes
        ax.plot_surface(xx, zz, -yy_initial, alpha=0.3, color='r')
        ax.plot_surface(xx, zz, -yy_optimized, alpha=0.3, color='g')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('-Y')
        ax.set_title('Initial (red) vs Optimized (green) Ground Plane - Side View')
        ax.legend()
        
        # Plot 3D visualization of planes - top-down view
        ax = plt.subplot(224, projection='3d')
        
        # Set a fixed viewing angle for top-down perspective
        ax.view_init(elev=90, azim=-90)
        
        # Plot sample points
        ax.scatter(sample_points[:, 0], sample_points[:, 2], -sample_points[:, 1], 
                  c='blue', alpha=0.3, s=1)
        
        # Plot ground candidate points
        ax.scatter(ground_sample_points[:, 0], ground_sample_points[:, 2], -ground_sample_points[:, 1], 
                  c='green', alpha=0.5, s=3)
        
        # Plot the planes
        ax.plot_surface(xx, zz, -yy_initial, alpha=0.3, color='r')
        ax.plot_surface(xx, zz, -yy_optimized, alpha=0.3, color='g')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('-Y')
        ax.set_title('Initial (red) vs Optimized (green) Ground Plane - Top View')
        
        plt.tight_layout()
        plt.show()
    
    return optimized_model

def detect_ground_plane(depth_map, threshold=0.05, max_iterations=500, lower_region_height=0.4, visualize_steps=False, optimize=True):
    """
    Detect the ground plane in a depth map using optimized RANSAC in 3D space.
    
    Args:
        depth_map: Numpy array containing the depth map
        threshold: Threshold for RANSAC inlier detection
        max_iterations: Maximum number of RANSAC iterations
        lower_region_height: Proportion of the image height to consider as the lower region
        visualize_steps: Whether to visualize intermediate steps
        optimize: Whether to optimize the ground plane after detection
        
    Returns:
        ground_mask: Boolean mask of ground pixels
        ground_model: Dictionary containing ground plane parameters
    """
    h, w = depth_map.shape
    
    # First convert the depth map to 3D points in world space
    # For optimization, we'll sample points rather than using all pixels
    
    # Define sampling rate - use more points for smaller images, fewer for larger ones
    if h * w > 1000000:  # > 1MP
        sample_rate = 0.1  # Use 10% of pixels
    elif h * w > 500000:  # > 0.5MP
        sample_rate = 0.2  # Use 20% of pixels
    else:
        sample_rate = 0.5  # Use 50% of pixels for small images
    
    # For very large images, cap the maximum number of points
    max_points = 100000
    
    # Determine sampling stride (sample every nth pixel)
    stride = max(1, int(1.0 / sample_rate))
    
    # Sample pixels using stride
    y_coords, x_coords = np.meshgrid(
        np.arange(0, h, stride), 
        np.arange(0, w, stride), 
        indexing='ij'
    )
    y_coords = y_coords.flatten()
    x_coords = x_coords.flatten()
    
    # For converting to 3D coordinates
    cx, cy = w / 2, h / 2
    focal_length_px = max(w, h)
    
    # Get depth values at sampled points
    sampled_depths = depth_map[y_coords, x_coords]
    
    # Filter out invalid depths
    valid_mask = ~np.isnan(sampled_depths) & (sampled_depths > 0)
    y_img = y_coords[valid_mask]
    x_img = x_coords[valid_mask]
    z = sampled_depths[valid_mask]
    
    # If we still have too many points, randomly subsample
    if len(z) > max_points:
        subsample_indices = np.random.choice(len(z), max_points, replace=False)
        y_img = y_img[subsample_indices]
        x_img = x_img[subsample_indices]
        z = z[subsample_indices]
    
    # Convert to 3D world coordinates (vectorized)
    points_3d = np.zeros((len(z), 3))
    points_3d[:, 0] = (x_img - cx) * z / focal_length_px  # X
    points_3d[:, 1] = (y_img - cy) * z / focal_length_px  # Y
    points_3d[:, 2] = z  # Z
    
    # Create a map from 3D points back to image coordinates
    point_to_pixel = {}
    for i in range(len(y_img)):
        point_to_pixel[i] = (y_img[i], x_img[i])
    
    # Focus on lower part of image where ground is likely
    lower_region = int((1 - lower_region_height) * h)
    
    # Simple but effective weighting: higher weight for lower points
    weights = np.ones(len(points_3d))
    lower_points = y_img >= lower_region
    weights[lower_points] = 3.0  # Higher weight for lower region
    
    # Further prioritize the very bottom for faster ground detection
    bottom_points = y_img >= int(0.8 * h)
    weights[bottom_points] *= 2.0  # Even higher weight for bottom 20%
    
    # Normalize weights for sampling
    weights = weights / weights.sum()
    
    print(f"Finding ground plane with {len(points_3d)} sampled 3D points...")
    
    # Setup for RANSAC
    best_plane = None
    most_inliers = 0
    best_inlier_mask = None
    
    # Adapt threshold to point cloud scale
    # Base threshold on median Z to handle different scales automatically
    median_z = np.median(points_3d[:, 2])
    adaptive_threshold = threshold * median_z * 0.1  # 10% of threshold * median depth
    
    # Use adaptive thresholds
    thresholds = [adaptive_threshold, adaptive_threshold * 2]
    
    # Early stopping criteria
    min_inlier_ratio = 0.2  # Stop when we find a plane with at least 20% inliers
    early_stopping_angle = 20  # Max angle with horizontal in degrees
    
    # Try multiple approaches with different parameters
    # Start with optimized RANSAC
    for t in thresholds:
        # Run RANSAC iterations
        for _ in range(max_iterations):
            # Sample 3 random points efficiently
            sample_indices = np.random.choice(
                len(points_3d), 
                size=3, 
                p=weights,
                replace=False
            )
            
            # Get the 3 points
            p1, p2, p3 = points_3d[sample_indices]
            
            # Calculate plane normal using cross product
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            
            # Skip if normal is too small (points are collinear)
            normal_length = np.linalg.norm(normal)
            if normal_length < 1e-6:
                continue
            
            # Normalize the normal vector
            normal = normal / normal_length
            
            # Make sure normal points "up" (-Y in camera space)
            if normal[1] > 0:
                normal = -normal
            
            # Quick check if plane is roughly horizontal
            horizontal = np.array([0, -1, 0])  # Y-up in world space
            cos_angle = np.abs(np.dot(normal, horizontal))
            angle = np.arccos(cos_angle) * 180 / np.pi
            
            # Skip planes that are too vertical (> 45 degrees from horizontal)
            if angle > 45:
                continue
            
            # Calculate d in plane equation ax + by + cz + d = 0
            d = -np.dot(normal, p1)
            
            # Fast vectorized distance calculation
            distances = np.abs(np.dot(points_3d, normal) + d)
            
            # Count inliers
            inlier_mask = distances < t
            num_inliers = np.sum(inlier_mask)
            inlier_ratio = num_inliers / len(points_3d)
            
            # Skip if too few inliers
            if inlier_ratio < 0.05:  # At least 5% must be inliers
                continue
                
            # Check if this plane is better than current best
            if num_inliers > most_inliers:
                most_inliers = num_inliers
                best_plane = {
                    'normal': normal,
                    'd': d,
                    'origin': p1,
                    'inliers_3d': points_3d[inlier_mask]
                }
                best_inlier_mask = inlier_mask
            
            # Early stopping - if we found a good plane, stop iterating
            if inlier_ratio > min_inlier_ratio and angle < early_stopping_angle:
                break
        
        if best_plane is not None:
            inlier_ratio = most_inliers / len(points_3d)
            print(f"  Found plane with {most_inliers} inliers ({inlier_ratio:.1%})")
            
            # Very efficient early stopping - if good enough, don't try other thresholds
            if inlier_ratio > min_inlier_ratio:
                break
    
    # If first method failed, use simple segmentation-based approach
    if best_plane is None:
        print("Using height-based segmentation for ground plane...")
        
        # Simple heuristic: lowest 30% of points likely contain ground
        lower_mask = np.zeros_like(depth_map, dtype=bool)
        lower_mask[int(0.7 * h):, :] = True
        
        # Get median depth of the bottom region
        bottom_depths = depth_map[lower_mask & (depth_map > 0)]
        
        if len(bottom_depths) > 0:
            median_depth = np.median(bottom_depths)
            
            # Create a simple horizontal plane at this depth
            dummy_normal = np.array([0, -1, 0])  # Y-up in world space
            dummy_origin = np.array([0, 0, median_depth])
            best_plane = {
                'normal': dummy_normal,
                'd': np.dot(dummy_normal, dummy_origin),
                'origin': dummy_origin,
                'inliers_3d': None
            }
            
            # Create a basic ground mask
            ground_mask = lower_mask
            return ground_mask, best_plane
        else:
            # If all else fails, use a dummy plane
            print("No valid ground points found, using default plane")
            ground_mask = np.zeros_like(depth_map, dtype=bool)
            ground_mask[int(0.7 * h):, :] = True
            
            dummy_normal = np.array([0, -1, 0])  # Y-up in world space
            dummy_origin = np.array([0, 0, 1.0])  # Default 1 meter depth
            best_plane = {
                'normal': dummy_normal,
                'd': -1.0,
                'origin': dummy_origin,
                'inliers_3d': None
            }
            
            return ground_mask, best_plane
    
    # Optional: quick refinement of the plane using SVD if we have enough inliers
    if np.sum(best_inlier_mask) >= 10:  # Need at least some points for stability
        inlier_points = points_3d[best_inlier_mask]
        
        # Quick SVD refinement
        centroid = np.mean(inlier_points, axis=0)
        centered_points = inlier_points - centroid
        
        # SVD to find best-fit plane
        u, s, vh = np.linalg.svd(centered_points)
        
        # The normal is the last column of vh (smallest singular value)
        refined_normal = vh[2, :]
        
        # Make sure normal points "up"
        if refined_normal[1] > 0:
            refined_normal = -refined_normal
            
        # Calculate d in plane equation
        refined_d = -np.dot(refined_normal, centroid)
        
        # Update the best plane
        best_plane['normal'] = refined_normal
        best_plane['d'] = refined_d
        best_plane['origin'] = centroid
    
    # Create ground mask for the full image (this is needed for visualization)
    # For efficiency, we'll create the mask by projecting all points to the plane
    
    # Fast approach: create a coarse mask and then refine with morphology
    # First create a binary mask at the original stride
    coarse_mask = np.zeros((h // stride + 1, w // stride + 1), dtype=bool)
    
    if best_inlier_mask is not None:
        # Map inlier indices back to our sampled grid
        for i in np.where(best_inlier_mask)[0]:
            y, x = point_to_pixel[i]
            y_idx, x_idx = y // stride, x // stride
            if 0 <= y_idx < coarse_mask.shape[0] and 0 <= x_idx < coarse_mask.shape[1]:
                coarse_mask[y_idx, x_idx] = True
    
    # Resize the mask back to the original image size
    if stride > 1:
        # Use nearest neighbor interpolation for binary mask
        ground_mask = cv2.resize(
            coarse_mask.astype(np.uint8), 
            (w, h), 
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    else:
        ground_mask = coarse_mask
    
    # Get normal and d from best plane
    normal = best_plane['normal']
    d = best_plane['d']
    
    # Report plane orientation
    horizontal = np.array([0, -1, 0])  # Y-up in world space
    angle = np.arccos(np.clip(np.dot(normal, horizontal), -1.0, 1.0)) * 180 / np.pi
    
    print(f"Ground plane normal: [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}]")
    print(f"Angle with horizontal: {angle:.2f} degrees")
    
    # Fast post-processing of the mask
    # Dilate to ensure good coverage
    ground_mask = ndimage.binary_dilation(ground_mask, iterations=2)
    
    # Fill holes for a cleaner mask
    ground_mask = ndimage.binary_fill_holes(ground_mask)
    
    # Remove small disconnected regions
    if np.sum(ground_mask) > 100:  # Only if we have enough ground pixels
        labeled_mask, num_labels = ndimage.label(ground_mask)
        if num_labels > 1:
            component_sizes = np.bincount(labeled_mask.ravel())[1:]
            largest_component = np.argmax(component_sizes) + 1
            ground_mask = labeled_mask == largest_component
    
    # Visualize if requested (simplified for speed)
    if visualize_steps:
        plt.figure(figsize=(10, 8))
        
        plt.subplot(221)
        plt.title("Depth Map")
        plt.imshow(depth_map, cmap='viridis')
        
        plt.subplot(222)
        plt.title("Ground Mask")
        plt.imshow(ground_mask, cmap='gray')
        
        # Simple 3D plot of the ground plane
        ax = plt.subplot(223, projection='3d')
        plt.title("3D Ground Plane")
        
        # Create a grid for the plane
        x_range = np.linspace(-1, 1, 10) * median_z
        z_range = np.linspace(0, 2, 10) * median_z
        xx, zz = np.meshgrid(x_range, z_range)
        
        # Calculate y coordinates from plane equation
        if abs(normal[1]) > 1e-6:
            yy = (-normal[0] * xx - normal[2] * zz - d) / normal[1]
        else:
            yy = np.zeros_like(xx)
        
        # Plot the plane
        ax.plot_surface(xx, zz, -yy, alpha=0.5, color='b')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('-Y')
        
        plt.subplot(224)
        plt.title("Ground Overlay")
        plt.imshow(depth_map, cmap='gray')
        plt.imshow(np.stack([np.zeros_like(ground_mask), 
                            np.zeros_like(ground_mask), 
                            ground_mask.astype(float)], axis=2), 
                  alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    # Add optimization step if requested
    if optimize and best_plane is not None:
        # Get all 3D points for optimization
        h, w = depth_map.shape
        
        # Define sampling rate for optimization
        # Use more points for smaller images, fewer for larger ones
        if h * w > 1000000:  # > 1MP
            sample_rate = 0.05  # Use 5% of pixels for large images
        elif h * w > 500000:  # > 0.5MP
            sample_rate = 0.1  # Use 10% of pixels
        else:
            sample_rate = 0.2  # Use 20% for small images
        
        # For very large images, cap the maximum number of points
        max_points = 100000  # Increase from 50000 to get more points for better optimization
        
        # Determine sampling stride
        stride = max(1, int(1.0 / sample_rate))
        
        # Sample pixels using stride
        y_coords, x_coords = np.meshgrid(
            np.arange(0, h, stride), 
            np.arange(0, w, stride), 
            indexing='ij'
        )
        y_coords = y_coords.flatten()
        x_coords = x_coords.flatten()
        
        # For converting to 3D coordinates
        cx, cy = w / 2, h / 2
        focal_length_px = max(w, h)
        
        # Get depth values at sampled points
        sampled_depths = depth_map[y_coords, x_coords]
        
        # Filter out invalid depths
        valid_depth_mask = ~np.isnan(sampled_depths) & (sampled_depths > 0)
        y_img = y_coords[valid_depth_mask]
        x_img = x_coords[valid_depth_mask]
        z = sampled_depths[valid_depth_mask]
        
        # If we still have too many points, randomly subsample
        if len(z) > max_points:
            subsample_indices = np.random.choice(len(z), max_points, replace=False)
            y_img = y_img[subsample_indices]
            x_img = x_img[subsample_indices]
            z = z[subsample_indices]
        
        # Convert to 3D world coordinates
        points_3d = np.zeros((len(z), 3))
        points_3d[:, 0] = (x_img - cx) * z / focal_length_px  # X
        points_3d[:, 1] = (y_img - cy) * z / focal_length_px  # Y
        points_3d[:, 2] = z  # Z
        
        # Optimize the ground plane
        best_plane = optimize_ground_plane(points_3d, best_plane, visualize=visualize_steps)
    
    return ground_mask, best_plane

def create_ground_plane_mesh(depth_map, ground_mask, ground_model, focallength_px, scale_factor=2.0, rotation_offset=None):
    """
    Create a ground plane mesh based on the detected ground plane parameters.
    The mesh will correctly align with the actual ground in 3D space.
    
    Args:
        depth_map: Depth map
        ground_mask: Boolean mask of ground pixels
        ground_model: Dictionary containing ground plane parameters
        focallength_px: Focal length in pixels
        scale_factor: How much larger to make the plane than the detected ground area
        rotation_offset: Optional list of rotation angles in degrees [x_rot, y_rot, z_rot]
        
    Returns:
        ground_plane_mesh: Open3D mesh representing the ground plane
    """
    if ground_model is None or np.sum(ground_mask) == 0:
        print("No ground model available, skipping ground plane creation")
        return o3d.geometry.TriangleMesh()
    
    # Apply rotation offset if provided
    if rotation_offset is not None and len(rotation_offset) == 3:
        ground_model = apply_rotation_to_plane(ground_model, rotation_offset)
    
    h, w = depth_map.shape
    cx, cy = w / 2, h / 2
    
    # Get ground plane parameters
    normal = ground_model['normal']
    plane_d = ground_model['d']
    
    # Ensure normal is normalized
    normal = normal / np.linalg.norm(normal)
    
    # Check if the plane is close to horizontal in world space
    up_vector = np.array([0, -1, 0])  # Y points up in world space
    cos_angle = np.abs(np.dot(normal, up_vector))
    angle_with_horizontal = np.arccos(cos_angle) * 180 / np.pi
    
    print(f"Ground plane angle with horizontal: {angle_with_horizontal:.2f} degrees")
    
    # Extract 3D coordinates of the ground points
    ground_pixels = np.where(ground_mask)
    if len(ground_pixels[0]) == 0:
        print("No ground pixels found, skipping ground plane creation")
        return o3d.geometry.TriangleMesh()
        
    # Get a sample of ground points (limit for efficiency)
    max_samples = 5000
    if len(ground_pixels[0]) > max_samples:
        sample_indices = np.random.choice(len(ground_pixels[0]), max_samples, replace=False)
        ground_y = ground_pixels[0][sample_indices]
        ground_x = ground_pixels[1][sample_indices]
    else:
        ground_y = ground_pixels[0]
        ground_x = ground_pixels[1]
    
    # Get depth values for these points
    ground_z = depth_map[ground_y, ground_x]
    
    # Filter to valid depths
    valid_mask = ground_z > 0
    ground_x = ground_x[valid_mask]
    ground_y = ground_y[valid_mask]
    ground_z = ground_z[valid_mask]
    
    if len(ground_z) == 0:
        print("No valid ground depths found, skipping ground plane creation")
        return o3d.geometry.TriangleMesh()
    
    # Convert to 3D coordinates
    ground_points_3d = []
    for i in range(len(ground_x)):
        x_img = ground_x[i]
        y_img = ground_y[i]
        z = ground_z[i]
        
        # Convert to 3D world coordinates
        x = (x_img - cx) * z / focallength_px
        y = (y_img - cy) * z / focallength_px
        ground_points_3d.append([x, y, z])
    
    ground_points_3d = np.array(ground_points_3d)
    
    # Use the plane equation to generate a properly aligned ground plane
    # First, determine the bounds of the ground area in 3D
    if 'inliers_3d' in ground_model and ground_model['inliers_3d'] is not None and len(ground_model['inliers_3d']) > 0:
        # Prefer to use the inliers from the RANSAC model
        ground_3d = ground_model['inliers_3d']
    else:
        # Use the points we just calculated
        ground_3d = ground_points_3d
    
    # Determine bounds in X and Z dimensions
    min_x = np.min(ground_3d[:, 0])
    max_x = np.max(ground_3d[:, 0])
    min_z = np.min(ground_3d[:, 2])
    max_z = np.max(ground_3d[:, 2])
    
    # Extend bounds by scale factor
    center_x = (min_x + max_x) / 2
    center_z = (min_z + max_z) / 2
    width = (max_x - min_x) * scale_factor
    depth = (max_z - min_z) * scale_factor
    
    # Make sure we have a minimum size for the plane
    # Use 2 meters as a minimum if the detected area is very small
    min_dimension = np.median(ground_z) * 2
    width = max(width, min_dimension)
    depth = max(depth, min_dimension)
    
    # Set bounds
    min_x = center_x - width / 2
    max_x = center_x + width / 2
    min_z = center_z - depth / 2
    max_z = center_z + depth / 2
    
    # Determine the Y value for the plane
    # We'll place it at a suitable height based on the ground model
    # Calculate several points on the plane to determine Y
    y_values = []
    test_points = [
        [center_x, center_z],        # Center
        [min_x, min_z],              # Bottom-left
        [max_x, min_z],              # Bottom-right
        [max_x, max_z],              # Top-right
        [min_x, max_z]               # Top-left
    ]
    
    for x, z in test_points:
        # Solve ax + by + cz + d = 0 for y
        if abs(normal[1]) > 1e-6:
            y = (-normal[0] * x - normal[2] * z - plane_d) / normal[1]
            y_values.append(y)
    
    # Use the median Y value for the plane
    # or place it slightly below all ground points if no Y could be calculated
    if y_values:
        base_y = np.median(y_values)
    else:
        base_y = np.max(ground_3d[:, 1]) * 1.02  # Slightly below lowest point
    
    # Create a grid of vertices for the plane
    n_grid = 20  # 20x20 grid for the plane
    x_coords = np.linspace(min_x, max_x, n_grid)
    z_coords = np.linspace(min_z, max_z, n_grid)
    
    vertices = []
    
    # Calculate proper y-value for each grid point
    for z in z_coords:
        for x in x_coords:
            if abs(normal[1]) > 1e-6:
                # Use plane equation to find y for each (x,z)
                y = (-normal[0] * x - normal[2] * z - plane_d) / normal[1]
            else:
                # If normal[1] is near zero, use the base_y value
                y = base_y
            
            vertices.append([x, y, z])
    
    # Create triangles from the grid
    triangles = []
    for i in range(n_grid - 1):
        for j in range(n_grid - 1):
            # Define quad corners
            idx00 = i * n_grid + j
            idx01 = i * n_grid + (j + 1)
            idx10 = (i + 1) * n_grid + j
            idx11 = (i + 1) * n_grid + (j + 1)
            
            # Create two triangles for the quad
            triangles.append([idx00, idx01, idx11])
            triangles.append([idx00, idx11, idx10])
    
    # Create Open3D mesh
    ground_mesh = o3d.geometry.TriangleMesh()
    ground_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    ground_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    # Set a light blue color with transparency
    ground_mesh.paint_uniform_color([0.7, 0.9, 1.0])
    
    # Compute normals for proper rendering
    ground_mesh.compute_vertex_normals()
    
    print(f"Created ground plane of dimensions {width:.2f}m × {depth:.2f}m")
    print(f"Plane normal: [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}]")
    
    return ground_mesh

def remove_stray_points(points, colors, min_neighbors=3, radius=0.2):
    """
    Remove stray points that are not connected to the main structure.
    
    Args:
        points: Nx3 array of point coordinates
        colors: Nx3 array of point colors
        min_neighbors: Minimum number of neighbors required
        radius: Radius for neighbor search
        
    Returns:
        filtered_points: Points with stray points removed
        filtered_colors: Corresponding colors
    """
    if len(points) == 0:
        return points, colors
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Compute nearest neighbors
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    # Check each point for number of neighbors
    valid_indices = []
    for i in range(len(points)):
        [k, idx, _] = kdtree.search_radius_vector_3d(pcd.points[i], radius)
        if k > min_neighbors:
            valid_indices.append(i)
    
    # Filter points and colors
    filtered_points = points[valid_indices]
    filtered_colors = colors[valid_indices]
    
    print(f"Removed {len(points) - len(filtered_points)} stray points out of {len(points)} total points")
    
    return filtered_points, filtered_colors

def find_depth_shadows(depth_map, threshold_factor=0.2, min_region_size=100, image=None):
    """
    Find shadow regions in a depth map using depth discontinuity detection.
    
    Args:
        depth_map: Numpy array containing the depth map
        threshold_factor: Factor to determine depth discontinuity (0.1-0.5)
        min_region_size: Minimum size of regions to keep
        image: Optional RGB image for edge detection assistance
        
    Returns:
        shadow_mask: Boolean mask of shadow regions
    """
    if depth_map is None:
        return None
        
    # Calculate depth gradients
    depth_dx = ndimage.sobel(depth_map, axis=1)
    depth_dy = ndimage.sobel(depth_map, axis=0)
    depth_gradient_magnitude = np.sqrt(depth_dx**2 + depth_dy**2)
    
    # Normalize gradient magnitude
    if depth_gradient_magnitude.max() > 0:
        depth_gradient_magnitude = depth_gradient_magnitude / depth_gradient_magnitude.max()
    
    # Use image edges to help if image is provided
    if image is not None:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Detect edges in the image
        edges = cv2.Canny(gray, 50, 150)
        
        # Combine depth gradients with image edges
        combined_edges = np.maximum(depth_gradient_magnitude, edges / 255.0)
    else:
        combined_edges = depth_gradient_magnitude
    
    # Threshold to find significant depth discontinuities
    depth_threshold = threshold_factor * depth_map.mean()
    edge_mask = combined_edges > threshold_factor
    
    # Label connected regions
    labeled_regions, num_regions = ndimage.label(~edge_mask)
    
    # Find the region containing the foreground (usually the largest region)
    region_sizes = ndimage.sum(np.ones_like(labeled_regions), labeled_regions, range(1, num_regions + 1))
    
    # Keep only regions larger than min_region_size
    valid_regions = np.where(region_sizes >= min_region_size)[0] + 1
    
    # Create a mask for valid regions
    valid_mask = np.isin(labeled_regions, valid_regions)
    
    # Shadow mask is the inverse of valid regions
    shadow_mask = ~valid_mask
    
    return shadow_mask

def force_horizontal_ground(ground_model, max_angle=5.0, completely_flat=False):
    """
    Force the ground plane to be more horizontal by adjusting its normal vector.
    
    Args:
        ground_model: Dictionary containing ground plane parameters
        max_angle: Maximum allowed angle with horizontal plane in degrees
        completely_flat: If True, make the plane perfectly horizontal
        
    Returns:
        adjusted_model: Dictionary with adjusted ground plane parameters
    """
    if ground_model is None:
        print("No ground model to adjust")
        return None
    
    # Get ground plane parameters
    normal = ground_model['normal']
    origin = ground_model['origin']
    
    # Ensure normal is normalized
    normal = normal / np.linalg.norm(normal)
    
    # Calculate current angle with horizontal
    horizontal_vector = np.array([0, -1, 0])  # Y-up in world space
    cos_angle = np.abs(np.dot(normal, horizontal_vector))
    current_angle = np.arccos(cos_angle) * 180 / np.pi
    
    print(f"Current ground plane angle with horizontal: {current_angle:.2f} degrees")
    
    if completely_flat:
        # Make the plane perfectly horizontal
        new_normal = np.array([0, -1, 0])  # Y-up in world space
        print("Forcing ground plane to be perfectly horizontal")
    elif current_angle > max_angle:
        # Limit the angle to max_angle
        print(f"Adjusting ground plane angle from {current_angle:.2f} to {max_angle:.2f} degrees")
        
        # Calculate rotation axis (perpendicular to both normal and horizontal)
        rotation_axis = np.cross(normal, horizontal_vector)
        if np.linalg.norm(rotation_axis) < 1e-6:
            # Vectors are parallel, choose any perpendicular axis
            rotation_axis = np.array([1, 0, 0])
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        
        # Calculate rotation angle
        rotation_angle = np.radians(current_angle - max_angle)
        
        # Create rotation matrix using Rodrigues' rotation formula
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        rotation_matrix = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * (K @ K)
        
        # Apply rotation to normal
        new_normal = rotation_matrix @ normal
        
        # Ensure normal points "up" (-Y in camera space)
        if new_normal[1] > 0:
            new_normal = -new_normal
    else:
        # Ground is already sufficiently horizontal
        print("Ground plane already satisfies maximum angle constraint")
        return ground_model
    
    # Create adjusted model
    adjusted_model = ground_model.copy()
    adjusted_model['normal'] = new_normal
    
    # Recalculate d in plane equation ax + by + cz + d = 0
    # d = -n·p where p is a point on the plane
    adjusted_model['d'] = -np.dot(new_normal, origin)
    
    # Calculate new angle with horizontal
    cos_angle = np.abs(np.dot(new_normal, horizontal_vector))
    new_angle = np.arccos(cos_angle) * 180 / np.pi
    
    print(f"Adjusted ground plane normal: [{new_normal[0]:.4f}, {new_normal[1]:.4f}, {new_normal[2]:.4f}]")
    print(f"New angle with horizontal: {new_angle:.2f} degrees")
    
    return adjusted_model

def grid_based_ground_plane_fit(points_3d, initial_ground_model=None, grid_size=100, visualize=False, force_horizontal=False):
    """
    Ground plane fitting using Z-binning approach:
    1. Divide the depth (Z-axis) into bins
    2. For each Z-bin, find the lowest points (in Y-axis) and use them as ground trace
    3. Fit a plane to these ground trace points, preferring a horizontal solution
    
    Args:
        points_3d: Nx3 array of 3D points
        initial_ground_model: Optional initial ground plane model (ignored)
        grid_size: Number of bins for the Z direction
        visualize: Whether to visualize the fitting process
        force_horizontal: Whether to force the ground plane to be horizontal
        
    Returns:
        fitted_ground_model: Ground plane model fitted to the ground trace
    """
    # Ensure points_3d is a numpy array
    points_3d = np.asarray(points_3d)
    valid_points = ~np.isnan(points_3d).any(axis=1)
    points = points_3d[valid_points]
    
    print(f"Z-binning ground plane fitting with {len(points)} points...")
    
    # Extract coordinates
    x_coords = points[:, 0]
    y_coords = points[:, 1]  # Y is now up, fixed in depth_to_3d
    z_coords = points[:, 2]
    
    # Calculate Z range for binning
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    z_range = z_max - z_min
    
    # Create bins in the Z direction
    n_bins = grid_size
    bin_size = z_range / n_bins
    
    print(f"Z range: [{z_min:.2f}, {z_max:.2f}], using {n_bins} bins of size {bin_size:.2f}m")
    
    # Initialize arrays to store ground trace points
    ground_trace_points = []
    
    # For each Z bin, find the lowest Y points (ground level)
    for i in range(n_bins):
        # Calculate bin boundaries
        bin_z_min = z_min + i * bin_size
        bin_z_max = z_min + (i + 1) * bin_size
        
        # Find points in this Z bin
        bin_mask = (z_coords >= bin_z_min) & (z_coords < bin_z_max)
        bin_points = points[bin_mask]
        
        if len(bin_points) > 10:  # Enough points for analysis
            # Sort by Y coordinate (height)
            sorted_indices = np.argsort(bin_points[:, 1])
            
            # Take the lowest 5% of points in this bin
            n_lowest = max(1, int(0.05 * len(bin_points)))
            lowest_points = bin_points[sorted_indices[:n_lowest]]
            
            # Calculate the average of these lowest points
            avg_point = np.mean(lowest_points, axis=0)
            ground_trace_points.append(avg_point)
    
    ground_trace_points = np.array(ground_trace_points)
    print(f"Found {len(ground_trace_points)} ground trace points from Z bins")
    
    if len(ground_trace_points) < 10:
        print("Warning: Too few ground trace points. Using fallback approach.")
        # Fall back to lowest 5% of all points by Y
        sorted_indices = np.argsort(y_coords)
        n_lowest = max(10, int(0.05 * len(points)))
        ground_trace_points = points[sorted_indices[:n_lowest]]
    
    # Now we have ground trace points, let's decide how to fit them
    
    # STEP 1: First try to fit a plane to all ground trace points
    if force_horizontal:
        # For horizontal plane, just take the average Y value of ground trace points
        # as the height of the plane
        normal = np.array([0, 1, 0])  # Y-up
        avg_y = np.mean(ground_trace_points[:, 1])
        d_plane = -avg_y
        
        print(f"Forcing horizontal plane at Y = {avg_y:.4f}")
    else:
        # For sloped plane, try RANSAC first (more robust to outliers)
        # We'll find the best plane that follows the ground trace points
        
        # Prepare data for RANSAC: Z is independent, Y is target
        if len(ground_trace_points) >= 10:
            try:
                from sklearn.linear_model import RANSACRegressor
                
                # Extract X, Z, Y coordinates - Y is the target
                X_data = ground_trace_points[:, [0, 2]]  # X and Z coordinates
                y_data = ground_trace_points[:, 1]       # Y coordinates
                
                # Use RANSAC for robust fitting - find relationship Y = f(X,Z)
                ransac = RANSACRegressor(min_samples=10, max_trials=1000, residual_threshold=0.1)
                ransac.fit(X_data, y_data)
                
                # Extract coefficients - these define our plane
                # y = a*x + c*z + d, which means normal = [a, -1, c]
                a = ransac.estimator_.coef_[0]  # X coefficient
                c = ransac.estimator_.coef_[1]  # Z coefficient
                d = ransac.estimator_.intercept_  # Intercept
                
                # Convert to plane equation: ax + by + cz + d = 0
                # We have: y = a*x + c*z + d, which rearranges to: -a*x + y - c*z - d = 0
                normal = np.array([-a, 1, -c])
                d_plane = -d
                
                # Normalize the normal vector
                normal = normal / np.linalg.norm(normal)
                
                # Calculate the angle with the horizontal
                horizontal_vector = np.array([0, 1, 0])  # Y-up
                cos_angle = np.abs(np.dot(normal, horizontal_vector))
                angle_degrees = np.arccos(cos_angle) * 180 / np.pi
                
                print(f"RANSAC fit with angle {angle_degrees:.2f}° from horizontal")
                
                # If the angle is too steep (> 20 degrees), use a more horizontal approach
                if angle_degrees > 20:
                    print(f"Plane angle {angle_degrees:.2f}° exceeds threshold. Using fallback horizontal fit.")
                    normal = np.array([0, 1, 0])  # Y-up
                    avg_y = np.median(ground_trace_points[:, 1])
                    d_plane = -avg_y
            except Exception as e:
                print(f"RANSAC fitting error: {e}. Using simple least squares.")
                # Fallback to simple least squares fit
                try:
                    # Fit a plane using least squares: y = ax + cz + d
                    A = np.column_stack((ground_trace_points[:, 0], ground_trace_points[:, 2], np.ones(len(ground_trace_points))))
                    b = ground_trace_points[:, 1]
                    
                    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                    a, c, d = x
                    
                    # Convert to plane equation: ax + by + cz + d = 0
                    # We have: y = ax + cz + d, which rearranges to: -ax + y - cz - d = 0
                    normal = np.array([-a, 1, -c])
                    d_plane = -d
                    
                    # Normalize the normal vector
                    normal = normal / np.linalg.norm(normal)
                    
                    # Calculate the angle with the horizontal
                    horizontal_vector = np.array([0, 1, 0])  # Y-up
                    cos_angle = np.abs(np.dot(normal, horizontal_vector))
                    angle_degrees = np.arccos(cos_angle) * 180 / np.pi
                    
                    print(f"Least squares fit with angle {angle_degrees:.2f}° from horizontal")
                    
                    # If the angle is too steep (> 20 degrees), use a more horizontal approach
                    if angle_degrees > 20:
                        print(f"Plane angle {angle_degrees:.2f}° exceeds threshold. Using horizontal fit.")
                        normal = np.array([0, 1, 0])  # Y-up
                        avg_y = np.median(ground_trace_points[:, 1])
                        d_plane = -avg_y
                except np.linalg.LinAlgError:
                    print("Error in least squares fit. Using horizontal plane.")
                    normal = np.array([0, 1, 0])  # Y-up
                    avg_y = np.median(ground_trace_points[:, 1])
                    d_plane = -avg_y
        else:
            # Not enough points for RANSAC, use simple horizontal plane
            print("Too few ground trace points for robust fitting. Using horizontal plane.")
            normal = np.array([0, 1, 0])  # Y-up
            avg_y = np.median(ground_trace_points[:, 1])
            d_plane = -avg_y
    
    # Ensure normal points up (Y positive)
    if normal[1] < 0:
        normal = -normal
        d_plane = -d_plane
    
    # STEP 2: Ensure no points go below the plane
    # Calculate signed distances to the plane
    signed_distances = np.dot(points, normal) + d_plane
    
    # Check if there are points below the plane
    points_below = np.sum(signed_distances < 0)
    if points_below > 0:
        print(f"Found {points_below} points below the plane. Adjusting plane down.")
        
        # Find the minimum signed distance
        min_distance = np.min(signed_distances)
        
        # Adjust d to move the plane down
        d_plane -= (min_distance + 0.05)  # Add a small safety margin
        
        # Recalculate points below
        signed_distances = np.dot(points, normal) + d_plane
        points_below = np.sum(signed_distances < 0)
        print(f"After adjustment: {points_below} points below the plane")
    
    # Calculate statistics about the fit
    horizontal_vector = np.array([0, 1, 0])  # Y-up
    cos_angle = np.abs(np.dot(normal, horizontal_vector))
    angle_degrees = np.arccos(cos_angle) * 180 / np.pi
    
    # Calculate signed distances and inliers for reporting purposes
    signed_distances = np.dot(points, normal) + d_plane
    inliers = np.abs(signed_distances) < 0.1  # Within 10cm of the plane
    num_inliers = np.sum(inliers)
    
    print(f"Ground plane fitting results:")
    print(f"  Normal vector: [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}]")
    print(f"  Angle with horizontal: {angle_degrees:.2f} degrees")
    print(f"  Found {num_inliers} inliers ({num_inliers/len(points)*100:.1f}%)")
    
    # Create the final ground model
    fitted_model = {
        'normal': normal,
        'd': d_plane,
        'origin': np.array([0, -d_plane / normal[1] if normal[1] != 0 else 0, 0])  # A point on the plane
    }
    
    # Visualize if requested
    if visualize:
        print("SHOWING 3D VISUALIZATION WITH RED SPHERES - PLEASE WAIT...")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(15, 12))
        
        # 3D visualization - full scene
        ax1 = fig.add_subplot(221, projection='3d')
        
        # Sample points for visualization (max 5000)
        if len(points) > 5000:
            viz_indices = np.random.choice(len(points), 5000, replace=False)
            viz_points = points[viz_indices]
        else:
            viz_points = points
            
        # Visualize all points
        ax1.scatter(viz_points[:, 0], viz_points[:, 2], viz_points[:, 1], c='blue', alpha=0.1, s=1)
        
        # Highlight the ground trace points with RED SPHERES
        ax1.scatter(ground_trace_points[:, 0], ground_trace_points[:, 2], ground_trace_points[:, 1], 
                   c='red', alpha=1.0, s=100, marker='o')
        
        print(f"Added {len(ground_trace_points)} RED SPHERES to visualization")
        
        # Draw the ground plane
        min_x, max_x = np.min(viz_points[:, 0]), np.max(viz_points[:, 0])
        min_z, max_z = np.min(viz_points[:, 2]), np.max(viz_points[:, 2])
        
        xx, zz = np.meshgrid([min_x, max_x], [min_z, max_z])
        
        # Calculate Y values for the plane at each X,Z coordinate
        if abs(normal[1]) > 1e-6:  # Avoid division by zero
            yy = (-normal[0] * xx - normal[2] * zz - d_plane) / normal[1]
        else:
            yy = np.zeros_like(xx)
            
        ax1.plot_surface(xx, zz, yy, alpha=0.3, color='green')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Z (depth)')
        ax1.set_zlabel('Y (height)')
        ax1.set_title('Point Cloud with Fitted Ground Plane')
        
        # Close-up view of Z-bins and ground trace
        ax2 = fig.add_subplot(222, projection='3d')
        
        # Only show the lowest 30% of the scene in Y direction
        all_y = viz_points[:, 1]
        y_threshold = np.percentile(all_y, 30)  # Show bottom 30% of points
        bottom_mask = all_y < y_threshold
        bottom_points = viz_points[bottom_mask]
        
        # Display all bottom points lightly
        if len(bottom_points) > 0:
            ax2.scatter(bottom_points[:, 0], bottom_points[:, 2], bottom_points[:, 1], 
                      c='blue', alpha=0.1, s=1)
        
        # RED SPHERES for ground trace points
        ax2.scatter(ground_trace_points[:, 0], ground_trace_points[:, 2], ground_trace_points[:, 1], 
                  c='red', alpha=1.0, s=100, marker='o')
        
        # Draw the Z-bin boundaries
        for i in range(n_bins+1):
            z_line = z_min + i * bin_size
            if z_line <= z_max:
                # Draw a vertical line at this Z position
                if abs(normal[1]) > 1e-6:
                    y_at_points = (-normal[0] * np.array([min_x, max_x]) - normal[2] * z_line - d_plane) / normal[1]
                else:
                    y_at_points = np.array([0, 0])
                    
                ax2.plot([min_x, max_x], [z_line, z_line], y_at_points, 'k-', alpha=0.1)
        
        # Draw the ground plane
        ax2.plot_surface(xx, zz, yy, alpha=0.3, color='green')
        
        # Set a good viewpoint to see Z-bins and ground trace
        ax2.view_init(elev=35, azim=-60)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z (depth)')
        ax2.set_zlabel('Y (height)')
        ax2.set_title('RED SPHERES: Ground Trace Points')
        
        # Side view (XY plane)
        ax3 = fig.add_subplot(223)
        ax3.scatter(viz_points[:, 0], viz_points[:, 1], c='blue', alpha=0.1, s=1)
        ax3.scatter(ground_trace_points[:, 0], ground_trace_points[:, 1], 
                   c='red', alpha=0.8, s=15)
        
        # Draw the sloped ground plane in X-Y projection
        x_range = np.linspace(min_x, max_x, 100)
        if abs(normal[1]) > 1e-6:  # Avoid division by zero
            # For each X, get Y when Z=0 (middle of the scene)
            middle_z = (min_z + max_z) / 2
            y_line = (-normal[0] * x_range - normal[2] * middle_z - d_plane) / normal[1]
        else:
            y_line = np.full_like(x_range, fitted_model['origin'][1])
            
        ax3.plot(x_range, y_line, 'g-', linewidth=2, label=f'Fitted Plane ({angle_degrees:.1f}°)')
        
        ax3.set_title('Side View (XY Plane) with Ground Level')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y (height)')
        ax3.grid(True)
        ax3.legend()
        
        # Front view (ZY plane)
        ax4 = fig.add_subplot(224)
        ax4.scatter(viz_points[:, 2], viz_points[:, 1], c='blue', alpha=0.1, s=1)
        ax4.scatter(ground_trace_points[:, 2], ground_trace_points[:, 1], 
                   c='red', alpha=0.8, s=15)
        
        # Draw the sloped ground plane in Z-Y projection
        z_range = np.linspace(min_z, max_z, 100)
        if abs(normal[1]) > 1e-6:  # Avoid division by zero
            # For each Z, get Y when X=0 (middle of the scene)
            middle_x = (min_x + max_x) / 2
            y_line = (-normal[0] * middle_x - normal[2] * z_range - d_plane) / normal[1]
        else:
            y_line = np.full_like(z_range, fitted_model['origin'][1])
            
        ax4.plot(z_range, y_line, 'g-', linewidth=2, label=f'Fitted Plane ({angle_degrees:.1f}°)')
        
        ax4.set_title('Front View (ZY Plane) with Ground Level')
        ax4.set_xlabel('Z (depth)')
        ax4.set_ylabel('Y (height)')
        ax4.grid(True)
        ax4.legend()
        
        plt.tight_layout()
        plt.show()
    
    return fitted_model

def remove_depth_shadows(depth_map, image=None, threshold_factor=0.2, min_region_size=100, interpolate_ground=True, 
                     image_path=None, use_saved_ground=True, rotation_offset=None, ground_params_dir=None, 
                     optimize_ground=True, force_horizontal=False, max_ground_angle=5.0, perfectly_flat_ground=False,
                     use_smart_ground_fit=True, visualize_ground=False):
    """
    Remove shadow artifacts from depth maps by detecting abrupt depth changes.
    Optionally interpolate ground plane in shadow areas.
    
    Args:
        depth_map: Numpy array containing the depth map
        image: Optional RGB image for edge detection assistance
        threshold_factor: Factor to determine depth discontinuity (0.1-0.5)
        min_region_size: Minimum size of regions to keep
        interpolate_ground: Whether to interpolate ground plane in shadow areas
        image_path: Path to the input image (for loading saved ground plane)
        use_saved_ground: Whether to try loading saved ground plane parameters
        rotation_offset: Optional list of rotation angles in degrees [x_rot, y_rot, z_rot]
        ground_params_dir: Directory to save/load ground plane parameters
        optimize_ground: Whether to optimize the ground plane after detection
        force_horizontal: Whether to force the ground plane to be more horizontal
        max_ground_angle: Maximum allowed angle with horizontal plane in degrees
        perfectly_flat_ground: If True, make the plane perfectly horizontal
        use_smart_ground_fit: Whether to use the Z-binning ground plane fitting approach (default True)
        visualize_ground: Whether to visualize the ground plane fitting process
    
    Returns:
        filtered_depth: Depth map with shadows removed
        ground_mask: Boolean mask of ground pixels
        ground_model: Dictionary containing ground plane parameters
    """
    # Find depth discontinuities (shadows)
    if depth_map is None:
        return None, None, None

    # Make a copy of the depth map for processing
    processed_depth = depth_map.copy()
    
    # Find shadow regions using depth discontinuity detection
    shadow_mask = find_depth_shadows(depth_map, threshold_factor, min_region_size)
    
    # Handle ground plane interpolation if requested
    ground_mask = None
    ground_model = None
    
    if interpolate_ground:
        # Try to load saved ground plane parameters if requested
        if use_saved_ground and image_path is not None:
            ground_model = load_ground_plane_params(image_path, ground_params_dir)
            
            if ground_model is not None:
                # Apply rotation offset if provided
                if rotation_offset is not None and len(rotation_offset) == 3:
                    ground_model = apply_rotation_to_plane(ground_model, rotation_offset)
                
                # We need to create a ground mask since we don't save it
                # Use a simple heuristic: lowest 30% of points likely contain ground
                h, w = depth_map.shape
                ground_mask = np.zeros_like(depth_map, dtype=bool)
                ground_mask[int(0.7 * h):, :] = True
                
                print("Using saved ground plane parameters")
                
                if use_smart_ground_fit:
                    # Convert depth map to 3D points for smart ground fitting
                    h, w = depth_map.shape
                    h_indices, w_indices = np.mgrid[0:h, 0:w]
                    valid_depth_mask = (depth_map > 0) & (~np.isnan(depth_map))
                    x_indices = w_indices[valid_depth_mask].flatten()
                    y_indices = h_indices[valid_depth_mask].flatten()
                    depths = depth_map[valid_depth_mask].flatten()
                    
                    # Sample points to reduce computational load
                    num_points = len(depths)
                    sample_size = min(100000, num_points)
                    if num_points > sample_size:
                        sample_indices = np.random.choice(num_points, sample_size, replace=False)
                        x_indices = x_indices[sample_indices]
                        y_indices = y_indices[sample_indices]
                        depths = depths[sample_indices]
                    
                    # Calculate focal length
                    cx, cy = w / 2, h / 2
                    focallength_px = max(w, h)
                    
                    # Convert to 3D points
                    points_3d, _ = depth_to_3d(depth_map, focallength_px, w, h)
                    
                    # Apply grid-based ground plane fitting
                    print(f"Using grid-based ground plane fitting with {len(points_3d)} points...")
                    ground_model = grid_based_ground_plane_fit(points_3d, ground_model, visualize=visualize_ground, force_horizontal=force_horizontal)
                # Optimize ground plane if requested
                elif optimize_ground:
                    # Get 3D points for optimization
                    h, w = depth_map.shape
                    cx, cy = w / 2, h / 2
                    focal_length_px = max(w, h)
                    
                    # Sample points for optimization - use more points for better optimization
                    sample_rate = 0.2  # Use 20% of pixels
                    stride = max(1, int(1.0 / sample_rate))
                    
                    y_coords, x_coords = np.meshgrid(
                        np.arange(0, h, stride), 
                        np.arange(0, w, stride), 
                        indexing='ij'
                    )
                    y_coords = y_coords.flatten()
                    x_coords = x_coords.flatten()
                    
                    # Get depth values
                    sampled_depths = depth_map[y_coords, x_coords]
                    
                    # Filter out invalid depths
                    valid_depth_mask = ~np.isnan(sampled_depths) & (sampled_depths > 0)
                    y_img = y_coords[valid_depth_mask]
                    x_img = x_coords[valid_depth_mask]
                    z = sampled_depths[valid_depth_mask]
                    
                    # Limit number of points for optimization
                    max_points = 100000  # Increased from 50000
                    if len(z) > max_points:
                        subsample_indices = np.random.choice(len(z), max_points, replace=False)
                        y_img = y_img[subsample_indices]
                        x_img = x_img[subsample_indices]
                        z = z[subsample_indices]
                    
                    # Convert to 3D world coordinates
                    points_3d = np.zeros((len(z), 3))
                    points_3d[:, 0] = (x_img - cx) * z / focal_length_px  # X
                    points_3d[:, 1] = (y_img - cy) * z / focal_length_px  # Y
                    points_3d[:, 2] = z  # Z
                    
                    # Optimize the ground plane
                    ground_model = optimize_ground_plane(points_3d, ground_model)
                    
                # Force ground plane to be more horizontal if requested
                if force_horizontal:
                    ground_model = force_horizontal_ground(
                        ground_model, 
                        max_angle=max_ground_angle, 
                        completely_flat=perfectly_flat_ground
                    )
                    
                # Save the optimized ground plane parameters
                if image_path is not None and use_saved_ground:
                    save_ground_plane_params(ground_model, image_path, ground_params_dir)
            else:
                # If loading failed, detect ground plane
                ground_mask, ground_model = detect_ground_plane(depth_map, optimize=optimize_ground)
                
                if use_smart_ground_fit:
                    # Convert depth map to 3D points for smart ground fitting
                    h, w = depth_map.shape
                    h_indices, w_indices = np.mgrid[0:h, 0:w]
                    valid_depth_mask = (depth_map > 0) & (~np.isnan(depth_map))
                    x_indices = w_indices[valid_depth_mask].flatten()
                    y_indices = h_indices[valid_depth_mask].flatten()
                    depths = depth_map[valid_depth_mask].flatten()
                    
                    # Sample points to reduce computational load
                    num_points = len(depths)
                    sample_size = min(100000, num_points)
                    if num_points > sample_size:
                        sample_indices = np.random.choice(num_points, sample_size, replace=False)
                        x_indices = x_indices[sample_indices]
                        y_indices = y_indices[sample_indices]
                        depths = depths[sample_indices]
                    
                    # Calculate focal length
                    cx, cy = w / 2, h / 2
                    focallength_px = max(w, h)
                    
                    # Convert to 3D points
                    points_3d, _ = depth_to_3d(depth_map, focallength_px, w, h)
                    
                    # Apply grid-based ground plane fitting
                    print(f"Using grid-based ground plane fitting with {len(points_3d)} points...")
                    ground_model = grid_based_ground_plane_fit(points_3d, ground_model, visualize=visualize_ground, force_horizontal=force_horizontal)
                
                # Apply rotation offset if provided
                if rotation_offset is not None and len(rotation_offset) == 3:
                    ground_model = apply_rotation_to_plane(ground_model, rotation_offset)
                
                # Force ground plane to be more horizontal if requested
                if force_horizontal:
                    ground_model = force_horizontal_ground(
                        ground_model, 
                        max_angle=max_ground_angle,
                        completely_flat=perfectly_flat_ground
                    )
                
                # Save ground plane parameters for future use
                if image_path is not None and use_saved_ground:
                    save_ground_plane_params(ground_model, image_path, ground_params_dir)
        else:
            # Detect ground plane
            ground_mask, ground_model = detect_ground_plane(depth_map, optimize=optimize_ground)
            
            if use_smart_ground_fit:
                # Convert depth map to 3D points for smart ground fitting
                h, w = depth_map.shape
                h_indices, w_indices = np.mgrid[0:h, 0:w]
                valid_depth_mask = (depth_map > 0) & (~np.isnan(depth_map))
                x_indices = w_indices[valid_depth_mask].flatten()
                y_indices = h_indices[valid_depth_mask].flatten()
                depths = depth_map[valid_depth_mask].flatten()
                
                # Sample points to reduce computational load
                num_points = len(depths)
                sample_size = min(100000, num_points)
                if num_points > sample_size:
                    sample_indices = np.random.choice(num_points, sample_size, replace=False)
                    x_indices = x_indices[sample_indices]
                    y_indices = y_indices[sample_indices]
                    depths = depths[sample_indices]
                
                # Calculate focal length
                cx, cy = w / 2, h / 2
                focallength_px = max(w, h)
                
                # Convert to 3D points
                points_3d, _ = depth_to_3d(depth_map, focallength_px, w, h)
                
                # Apply grid-based ground plane fitting
                print(f"Using grid-based ground plane fitting with {len(points_3d)} points...")
                ground_model = grid_based_ground_plane_fit(points_3d, ground_model, visualize=visualize_ground, force_horizontal=force_horizontal)
            
            # Apply rotation offset if provided
            if rotation_offset is not None and len(rotation_offset) == 3:
                ground_model = apply_rotation_to_plane(ground_model, rotation_offset)
            
            # Force ground plane to be more horizontal if requested
            if force_horizontal:
                ground_model = force_horizontal_ground(
                    ground_model, 
                    max_angle=max_ground_angle,
                    completely_flat=perfectly_flat_ground
                )
            
            # Save ground plane parameters for future use
            if image_path is not None and use_saved_ground:
                save_ground_plane_params(ground_model, image_path, ground_params_dir)
                
        # Interpolate ground plane in shadow areas
        if ground_model is not None and ground_mask is not None:
            # Convert shadow mask to array of indices
            shadow_points = np.where(shadow_mask.flatten())[0]
            
            if len(shadow_points) > 0:
                # Get ground mask as flat array
                h, w = depth_map.shape
                ground_mask_flat = ground_mask.flatten()
                
                # Only consider shadows on ground
                shadow_ground_mask = np.zeros_like(ground_mask_flat)
                shadow_ground_mask[shadow_points] = True
                shadow_ground_mask = shadow_ground_mask & ground_mask_flat
                
                shadow_ground_points = np.where(shadow_ground_mask)[0]
                
                if len(shadow_ground_points) > 0:
                    print(f"Interpolating depth for {len(shadow_ground_points)} shadow ground points")
                    
                    # Get ground plane equation
                    normal = ground_model['normal']
                    d = ground_model['d']
                    
                    # Calculate indices for each point
                    y_indices = shadow_ground_points // w
                    x_indices = shadow_ground_points % w
                    
                    # Calculate depth (z) values for ground plane at these points
                    # Using plane equation: ax + by + cz + d = 0
                    # where [a,b,c] is the normal vector
                    # We'll first calculate the 3D world coordinates of each pixel with a dummy depth of 1.0
                    
                    cx, cy = w / 2, h / 2
                    focal_length_px = max(w, h)
                    
                    # Calculate coordinates with a dummy depth of 1.0
                    dummy_depth = np.ones_like(x_indices, dtype=float)
                    x = (x_indices - cx) * dummy_depth / focal_length_px
                    y = (y_indices - cy) * dummy_depth / focal_length_px
                    
                    # Calculate the proper depth using plane equation
                    # ax + by + cz + d = 0
                    # cz = -ax - by - d
                    # z = (-ax - by - d) / c
                    
                    # Handle different cases of the plane normal
                    if abs(normal[2]) > 1e-6:  # Normal has non-zero Z component
                        z = (-normal[0] * x - normal[1] * y - d) / normal[2]
                    elif abs(normal[0]) > 1e-6:  # X-aligned normal
                        # In this case, we solve for x instead: x = (-by - cz - d) / a
                        # We use a small range of z values and pick the most reasonable depth
                        z_range = np.linspace(0.1, 10.0, 100)  # Try depths from 0.1m to 10m
                        best_z = None
                        best_error = float('inf')
                        
                        for z_val in z_range:
                            # Calculate x for this z
                            x_val = (-normal[1] * y - normal[2] * z_val - d) / normal[0]
                            # Convert back to image coordinates
                            img_x = cx + x_val * z_val * focal_length_px
                            # Calculate error as distance from original image point
                            error = np.abs(img_x - x_indices)
                            if np.mean(error) < best_error:
                                best_error = np.mean(error)
                                best_z = z_val
                                
                        z = np.full_like(x, best_z)
                    else:  # Y-aligned normal (horizontal plane)
                        # For a horizontal plane (normal = [0, -1, 0]), the depth is independent of x, z
                        # The depth is determined by the plane offset: y = d
                        # Convert world y to camera depth
                        plane_y = -d / normal[1]  # World Y position of the plane
                        
                        # For each point, calculate the depth that would place it at this Y position
                        # We know: y = (y_indices - cy) * z / focal_length_px
                        # Solve for z: z = plane_y * focal_length_px / (y_indices - cy)
                        
                        # Avoid division by zero when y_index = cy
                        denominator = y_indices - cy
                        denominator[np.abs(denominator) < 1e-6] = 1e-6  # Small non-zero value
                        
                        z = plane_y * focal_length_px / denominator
                    
                    # Ensure positive depths
                    z = np.maximum(z, 0.1)
                    
                    # Update depth map with interpolated ground plane depths
                    processed_depth.flat[shadow_ground_points] = z
            
    # Fill remaining shadow areas with nearest valid depths
    valid_mask = ~shadow_mask | (processed_depth > 0)
    
    if not np.all(valid_mask):
        invalid_indices = np.where(~valid_mask)
        valid_indices = np.where(valid_mask)
        
        # Find nearest valid points for each invalid point
        tree = cKDTree(np.column_stack(valid_indices))
        _, nearest_indices = tree.query(np.column_stack(invalid_indices))
        
        # Replace invalid depths with nearest valid depths
        nearest_valid_row = valid_indices[0][nearest_indices]
        nearest_valid_col = valid_indices[1][nearest_indices]
        processed_depth[invalid_indices] = processed_depth[nearest_valid_row, nearest_valid_col]
        
    return processed_depth, ground_mask, ground_model

def visualize_ground_detection(depth_map, ground_mask, image=None):
    """
    Visualize the detected ground plane.
    
    Args:
        depth_map: Depth map
        ground_mask: Boolean mask of ground pixels
        image: Optional RGB image
    """
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.title("Original Depth")
    plt.imshow(depth_map, cmap='viridis')
    
    plt.subplot(132)
    plt.title("Ground Mask")
    plt.imshow(ground_mask, cmap='gray')
    
    plt.subplot(133)
    if image is not None:
        plt.title("Ground Overlay")
        plt.imshow(image)
        plt.imshow(ground_mask, alpha=0.5, cmap='cool')
    else:
        plt.title("Ground Depth")
        ground_depth = depth_map.copy()
        ground_depth[~ground_mask] = np.nan
        plt.imshow(ground_depth, cmap='viridis')
    
    plt.tight_layout()
    plt.show()

def create_floor_plan(depth_map, image, ground_model, focallength_px, height_threshold=0.5, 
                     output_path="floor_plan.png", visualize=True, dpi=300, 
                     simplified=False, color_by_height=False, max_height=2.5,
                     show_text=False):
    """
    Create a 2D floor plan by projecting objects above a certain height onto the ground plane.
    
    Args:
        depth_map: Depth map
        image: RGB image
        ground_model: Dictionary containing ground plane parameters
        focallength_px: Focal length in pixels
        height_threshold: Height threshold in meters above ground plane (objects above this will be included)
        output_path: Path to save the floor plan image
        visualize: Whether to visualize the floor plan
        dpi: DPI for the output image
        simplified: Whether to create a simplified floor plan with cleaner outlines
        color_by_height: Whether to color the floor plan by height
        max_height: Maximum height in meters for color mapping
        show_text: Whether to show text labels on the floor plan
        
    Returns:
        floor_plan: The floor plan image as a numpy array
    """
    if ground_model is None:
        print("No ground model available, cannot create floor plan")
        return None
    
    # Get image dimensions
    h, w = depth_map.shape
    
    # Calculate principal point (assuming center of image)
    cx, cy = w / 2, h / 2
    
    # Create meshgrid of pixel coordinates
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Convert pixel coordinates to camera coordinates
    Z = depth_map
    X = (x_coords - cx) * Z / focallength_px
    Y = (y_coords - cy) * Z / focallength_px
    
    # Get ground plane parameters
    normal = ground_model['normal']
    d = ground_model['d']
    
    # Create a binary mask for the floor plan
    floor_plan_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Calculate height above ground plane for each point
    # For each point (x,y,z), the distance to the plane is:
    # dist = (ax + by + cz + d) / sqrt(a^2 + b^2 + c^2)
    # Since normal is normalized, we can simplify to:
    # dist = ax + by + cz + d
    
    # Reshape 3D coordinates for vectorized calculation
    points_3d = np.stack((X, Y, Z), axis=-1)
    valid_mask = ~np.isnan(Z) & (Z > 0)
    
    # Calculate signed distance to ground plane (positive is above ground)
    # We want to flip the sign if normal points down (normal[1] > 0)
    sign = -1 if normal[1] > 0 else 1
    distances = sign * (np.sum(points_3d * normal, axis=-1) + d)
    
    # Create mask for points above the height threshold
    above_threshold = (distances > height_threshold) & valid_mask
    
    # Store height information for coloring
    height_values = distances.copy()
    
    # Create a top-down view (floor plan)
    # We'll create a 2D grid representing the floor plan
    # First, determine the bounds of the scene in world coordinates
    
    # Get min/max X and Z coordinates (top-down view uses X and Z)
    valid_points = points_3d[valid_mask]
    if len(valid_points) == 0:
        print("No valid points found, cannot create floor plan")
        return None
    
    min_x, max_x = np.min(valid_points[:, 0]), np.max(valid_points[:, 0])
    min_z, max_z = np.min(valid_points[:, 2]), np.max(valid_points[:, 2])
    
    # Add some padding
    padding = 0.5  # 0.5 meters padding
    min_x -= padding
    max_x += padding
    min_z -= padding
    max_z += padding
    
    # Create a high-resolution grid for the floor plan
    grid_resolution = 0.05  # 5cm grid cells
    grid_width = int((max_x - min_x) / grid_resolution) + 1
    grid_height = int((max_z - min_z) / grid_resolution) + 1
    
    # Create empty floor plan grid
    floor_plan_grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    floor_plan_grid.fill(255)  # White background
    
    # Create a height grid for coloring
    height_grid = np.zeros((grid_height, grid_width), dtype=np.float32)
    
    # Project points above threshold onto the floor plan
    points_above = points_3d[above_threshold]
    colors_above = np.array(image).reshape(-1, 3)[above_threshold.flatten()]
    heights_above = height_values[above_threshold]
    
    if len(points_above) > 0:
        # Convert 3D coordinates to grid indices
        grid_x = ((points_above[:, 0] - min_x) / grid_resolution).astype(int)
        grid_z = ((points_above[:, 2] - min_z) / grid_resolution).astype(int)
        
        # Ensure indices are within bounds
        valid_indices = (grid_x >= 0) & (grid_x < grid_width) & (grid_z >= 0) & (grid_z < grid_height)
        grid_x = grid_x[valid_indices]
        grid_z = grid_z[valid_indices]
        colors = colors_above[valid_indices]
        heights = heights_above[valid_indices]
        
        # Mark occupied cells in the grid
        for i in range(len(grid_x)):
            floor_plan_grid[grid_z[i], grid_x[i]] = colors[i]
            height_grid[grid_z[i], grid_x[i]] = heights[i]
    
    # Create a simplified floor plan if requested
    if simplified:
        # Create a binary occupancy grid
        occupied_mask = np.any(floor_plan_grid < 255, axis=2)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        cleaned_mask = cv2.morphologyEx(occupied_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours for the walls
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a new clean floor plan
        simplified_plan = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
        
        # Draw filled contours with different colors based on area
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10:  # Filter out tiny contours
                # Simplify the contour
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Fill the contour with a light color
                if color_by_height:
                    # Get the average height for this contour
                    mask = np.zeros((grid_height, grid_width), dtype=np.uint8)
                    cv2.drawContours(mask, [contour], 0, 1, -1)
                    heights_in_contour = height_grid[mask == 1]
                    if len(heights_in_contour) > 0:
                        avg_height = np.mean(heights_in_contour)
                        # Map height to color (blue to red gradient)
                        normalized_height = min(1.0, avg_height / max_height)
                        r = int(255 * normalized_height)
                        g = int(255 * (1 - abs(2 * normalized_height - 1)))
                        b = int(255 * (1 - normalized_height))
                        fill_color = (b, g, r)  # BGR format for OpenCV
                    else:
                        fill_color = (240, 240, 240)  # Light gray
                else:
                    fill_color = (240, 240, 240)  # Light gray
                
                cv2.drawContours(simplified_plan, [approx], 0, fill_color, -1)
                
                # Draw the contour outline with a thicker line
                cv2.drawContours(simplified_plan, [approx], 0, (0, 0, 0), 2)
        
        # Use the simplified plan
        floor_plan_with_outlines = simplified_plan
    else:
        # Add a border to the occupied cells to create outlines
        occupied_mask = np.any(floor_plan_grid < 255, axis=2)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(occupied_mask.astype(np.uint8), kernel, iterations=1)
        edge_mask = dilated & ~occupied_mask
        
        # Create a copy for the outlined version
        floor_plan_with_outlines = floor_plan_grid.copy()
        
        # Color by height if requested
        if color_by_height:
            # Create a height-colored version
            height_colored = np.ones_like(floor_plan_grid) * 255
            
            # Normalize heights to 0-1 range
            normalized_heights = np.clip(height_grid / max_height, 0, 1)
            
            # Create a colormap
            for i in range(grid_height):
                for j in range(grid_width):
                    if occupied_mask[i, j]:
                        h = normalized_heights[i, j]
                        # Map height to color (blue to red gradient)
                        r = int(255 * h)
                        g = int(255 * (1 - abs(2 * h - 1)))
                        b = int(255 * (1 - h))
                        height_colored[i, j] = [b, g, r]  # BGR format for OpenCV
            
            floor_plan_with_outlines = height_colored
        
        # Add black outlines
        floor_plan_with_outlines[edge_mask] = [0, 0, 0]
    
    # Add a scale bar
    scale_bar_length_meters = 1.0  # 1 meter
    scale_bar_pixels = int(scale_bar_length_meters / grid_resolution)
    scale_bar_height = 20
    scale_bar_margin = 50
    
    # Add scale bar at the bottom right
    scale_bar_start_x = grid_width - scale_bar_margin - scale_bar_pixels
    scale_bar_end_x = grid_width - scale_bar_margin
    scale_bar_y = grid_height - scale_bar_margin
    
    floor_plan_with_outlines[scale_bar_y:scale_bar_y+scale_bar_height, 
                            scale_bar_start_x:scale_bar_end_x] = [0, 0, 0]
    
    # Add text for scale bar only if show_text is True
    if show_text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(floor_plan_with_outlines, "1m", 
                    (scale_bar_start_x, scale_bar_y - 10), 
                    font, 0.7, (0, 0, 0), 2)
    
    # Add north arrow (simplified)
    arrow_center_x = scale_bar_start_x - 100
    arrow_center_y = scale_bar_y
    arrow_length = 40
    
    # Draw arrow
    cv2.arrowedLine(floor_plan_with_outlines, 
                   (arrow_center_x, arrow_center_y + arrow_length//2),
                   (arrow_center_x, arrow_center_y - arrow_length//2),
                   (0, 0, 0), 2, tipLength=0.3)
    
    # Add "N" label only if show_text is True
    if show_text:
        cv2.putText(floor_plan_with_outlines, "N", 
                    (arrow_center_x - 10, arrow_center_y - arrow_length//2 - 10), 
                    font, 0.7, (0, 0, 0), 2)
    
    # Add title and information only if show_text is True
    if show_text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        title = "Floor Plan (Objects above {:.2f}m)".format(height_threshold)
        cv2.putText(floor_plan_with_outlines, title, 
                    (50, 50), font, 1.0, (0, 0, 0), 2)
        
        # Add grid dimensions
        dimensions_text = "Dimensions: {:.1f}m x {:.1f}m".format(max_x - min_x, max_z - min_z)
        cv2.putText(floor_plan_with_outlines, dimensions_text, 
                    (50, 80), font, 0.7, (0, 0, 0), 2)
    
    # Add color legend if coloring by height
    if color_by_height:
        legend_width = 200
        legend_height = 20
        legend_x = 50
        legend_y = 100
        
        # Create gradient
        for i in range(legend_width):
            h = i / legend_width
            r = int(255 * h)
            g = int(255 * (1 - abs(2 * h - 1)))
            b = int(255 * (1 - h))
            color = [b, g, r]  # BGR format for OpenCV
            floor_plan_with_outlines[legend_y:legend_y+legend_height, legend_x+i] = color
        
        # Add border
        cv2.rectangle(floor_plan_with_outlines, 
                     (legend_x, legend_y), 
                     (legend_x+legend_width, legend_y+legend_height), 
                     (0, 0, 0), 1)
        
        # Add labels only if show_text is True
        if show_text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(floor_plan_with_outlines, "0m", 
                        (legend_x, legend_y+legend_height+20), 
                        font, 0.5, (0, 0, 0), 1)
            cv2.putText(floor_plan_with_outlines, "{:.1f}m".format(max_height), 
                        (legend_x+legend_width-30, legend_y+legend_height+20), 
                        font, 0.5, (0, 0, 0), 1)
            cv2.putText(floor_plan_with_outlines, "Height", 
                        (legend_x+legend_width//2-20, legend_y-10), 
                        font, 0.5, (0, 0, 0), 1)
    
    # Save the floor plan
    if output_path:
        plt.figure(figsize=(grid_width/dpi, grid_height/dpi), dpi=dpi)
        plt.imshow(floor_plan_with_outlines)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close()
        print(f"Floor plan saved to {output_path}")
    
    # Visualize if requested
    if visualize:
        plt.figure(figsize=(12, 10))
        plt.imshow(floor_plan_with_outlines)
        if show_text:
            plt.title("Floor Plan (Objects above {:.2f}m)".format(height_threshold))
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return floor_plan_with_outlines

def create_mesh_from_video3d_pointcloud(image_path, output_path=None, visualize=True, remove_shadows=True, 
                                        interpolate_ground=True, remove_strays=True, visualize_ground=False,
                                        add_ground_plane=True, use_saved_ground=True, rotation_offset=None,
                                        ground_params_dir=None, force_horizontal=False, max_ground_angle=5.0,
                                        perfectly_flat_ground=False, use_smart_ground_fit=False,
                                        normalize_ground_to_zero=False, grid_adjust_ground=False, 
                                        grid_size=20, ground_percentile=5):
    """
    Create a mesh directly from the exact same point cloud as video3D.py.
    This function first creates the point cloud exactly as in video3D.py,
    then converts it to a mesh with no additional processing.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the output mesh (optional)
        visualize: Whether to visualize the mesh
        remove_shadows: Whether to remove shadow artifacts
        interpolate_ground: Whether to interpolate ground plane in shadow areas
        remove_strays: Whether to remove stray points
        visualize_ground: Whether to visualize ground detection
        add_ground_plane: Whether to add a transparent ground plane to the visualization
        use_saved_ground: Whether to try loading saved ground plane parameters
        rotation_offset: Optional list of rotation angles in degrees [x_rot, y_rot, z_rot]
        ground_params_dir: Directory to save/load ground plane parameters
        force_horizontal: Whether to force the ground plane to be more horizontal
        max_ground_angle: Maximum allowed angle with horizontal plane in degrees
        perfectly_flat_ground: If True, make the plane perfectly horizontal
        use_smart_ground_fit: Whether to use the grid-based ground plane fitting
        normalize_ground_to_zero: Whether to normalize coordinates so ground plane is at y=0
        grid_adjust_ground: Whether to use grid-based approach to ensure points touch ground
        grid_size: Number of grid cells in each dimension for grid adjustment
        ground_percentile: Percentile of lowest points to consider for grid adjustment
    
    Returns:
        None
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
    focallength_px = prediction["focallength_px"]  # Focal length in pixels
    
    # Convert depth to numpy array
    depth_np = depth.detach().cpu().numpy()
    
    # Remove shadow artifacts if requested
    ground_mask = None
    ground_model = None
    if remove_shadows:
        depth_np, ground_mask, ground_model = remove_depth_shadows(
            depth_np, np.array(image), 
            interpolate_ground=interpolate_ground,
            image_path=image_path,
            use_saved_ground=use_saved_ground,
            rotation_offset=rotation_offset,
            ground_params_dir=ground_params_dir,
            optimize_ground=True,
            force_horizontal=force_horizontal,
            max_ground_angle=max_ground_angle,
            perfectly_flat_ground=perfectly_flat_ground,
            use_smart_ground_fit=use_smart_ground_fit,
            visualize_ground=visualize_ground
        )
        
        # Visualize ground detection if requested
        if visualize_ground and ground_mask is not None:
            visualize_ground_detection(depth.detach().cpu().numpy(), ground_mask, np.array(image))
    
    # Get image dimensions
    h, w = depth_np.shape
    
    # Calculate principal point (assuming center of image)
    cx, cy = w / 2, h / 2
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    
    # Use the new depth_to_3d function for consistent conversion 
    # This ensures Y is up in the world coordinate system
    points_3d, valid_mask = depth_to_3d(depth_np, focallength_px.item(), w, h)
    
    # Reshape the RGB image and get colors for valid points
    colors = np.array(image).reshape(-1, 3)[valid_mask.flatten()] / 255.0
    
    # Remove stray points if requested
    if remove_strays and len(points_3d) > 0:
        points_3d, colors = remove_stray_points(points_3d, colors)
    
    # Apply ground plane normalization if requested and ground model is available
    if normalize_ground_to_zero and ground_model is not None:
        print("Normalizing point cloud so ground plane is at y=0...")
        points_3d = normalize_point_cloud_to_ground(points_3d, ground_model)
        
        # After normalization, we need to update ground_model to reflect y=0 plane
        # The new plane equation is simply y = 0 (normal = [0,1,0], d = 0)
        ground_model = {
            'normal': np.array([0, 1, 0]),
            'origin': np.array([0, 0, 0]),
            'd': 0
        }
        
        # Apply grid-based adjustment if requested
        if grid_adjust_ground:
            print(f"Applying grid-based ground adjustment (grid size: {grid_size}, percentile: {ground_percentile})...")
            points_3d = grid_based_ground_adjustment(points_3d, grid_size=grid_size, percentile=ground_percentile)
    
    # Add points and colors to the point cloud
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create a mesh from the point cloud using Poisson surface reconstruction
    mesh = o3d.geometry.TriangleMesh()
    
    # Add ground plane mesh if requested and ground_model is available
    ground_mesh = None
    if add_ground_plane and ground_model is not None:
        # If we normalized the ground to y=0, create a simple horizontal ground plane
        if normalize_ground_to_zero:
            # Find extent of points
            x_min, x_max = np.min(points_3d[:, 0]), np.max(points_3d[:, 0])
            z_min, z_max = np.min(points_3d[:, 2]), np.max(points_3d[:, 2])
            
            # Add margin around points
            margin = max(x_max - x_min, z_max - z_min) * 0.2
            x_min -= margin
            x_max += margin
            z_min -= margin
            z_max += margin
            
            # Create a simple horizontal ground plane mesh at y=0
            ground_mesh = o3d.geometry.TriangleMesh()
            ground_mesh.vertices = o3d.utility.Vector3dVector([
                [x_min, 0, z_min],
                [x_max, 0, z_min],
                [x_max, 0, z_max],
                [x_min, 0, z_max]
            ])
            ground_mesh.triangles = o3d.utility.Vector3iVector(
                [[0, 1, 2], [0, 2, 3]]
            )
            
            # Set a semi-transparent gray color
            ground_color = np.array([0.8, 0.8, 0.8])
            ground_mesh.vertex_colors = o3d.utility.Vector3dVector(
                np.tile(ground_color, (4, 1))
            )
        else:
            # Create a large ground plane mesh based on the detected ground
            if ground_mask is not None:
                ground_mesh = create_ground_plane_mesh(
                    depth_np, ground_mask, ground_model, focallength_px.item(), 
                    scale_factor=5.0, rotation_offset=rotation_offset
                )
            else:
                # Create a simple ground plane mesh if no ground mask is available
                ground_size = 20.0  # meters
                ground_opacity = 0.5
                
                # Create vertices for the ground plane
                origin = ground_model['origin']
                normal = ground_model['normal']
                
                # Find two vectors perpendicular to the normal
                if abs(normal[0]) < abs(normal[1]) and abs(normal[0]) < abs(normal[2]):
                    v1 = np.array([0, -normal[2], normal[1]])
                elif abs(normal[1]) < abs(normal[2]):
                    v1 = np.array([-normal[2], 0, normal[0]])
                else:
                    v1 = np.array([-normal[1], normal[0], 0])
                
                v1 = v1 / np.linalg.norm(v1)
                v2 = np.cross(normal, v1)
                v2 = v2 / np.linalg.norm(v2)
                
                # Scale the vectors to the desired ground size
                v1 = v1 * ground_size
                v2 = v2 * ground_size
                
                # Create the 4 corners of the ground plane
                corners = [
                    origin - v1 - v2,
                    origin + v1 - v2,
                    origin + v1 + v2,
                    origin - v1 + v2
                ]
                
                # Create a mesh with 2 triangles
                ground_mesh = o3d.geometry.TriangleMesh()
                ground_mesh.vertices = o3d.utility.Vector3dVector(corners)
                ground_mesh.triangles = o3d.utility.Vector3iVector(
                    [[0, 1, 2], [0, 2, 3]]
                )
                
                # Set the color to gray with some transparency
                ground_color = np.array([0.8, 0.8, 0.8])
                ground_mesh.vertex_colors = o3d.utility.Vector3dVector(
                    np.tile(ground_color, (4, 1))
                )
    
    if output_path:
        # Save the point cloud as PLY
        point_cloud_path = output_path
        if not point_cloud_path.endswith('.ply'):
            point_cloud_path = point_cloud_path + '.ply'
        o3d.io.write_point_cloud(point_cloud_path, pcd)
        print(f"Point cloud saved to {point_cloud_path}")
        
        # If we have a mesh, save it too
        if mesh.vertices:
            mesh_path = output_path
            if not mesh_path.endswith('.obj'):
                mesh_path = mesh_path + '.obj'
            o3d.io.write_triangle_mesh(mesh_path, mesh)
            print(f"Mesh saved to {mesh_path}")
    
    if visualize:
        # Visualize the point cloud
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        # Add the point cloud to the visualizer
        vis.add_geometry(pcd)
        
        # Add ground plane if available
        if ground_mesh:
            vis.add_geometry(ground_mesh)
        
        # Set visualization options
        opt = vis.get_render_option()
        opt.point_size = 2.0
        
        # Set view control options - position camera to look at scene
        ctrl = vis.get_view_control()
        ctrl.set_up([0, -1, 0])  # Set Y axis as up direction in visualization
        
        # Show the visualization
        vis.run()
        vis.destroy_window()
    
    return

def create_textured_mesh(image_path, output_path=None, visualize=True, remove_shadows=True,
                         interpolate_ground=True, remove_strays=True, visualize_ground=False,
                         add_ground_plane=True, use_saved_ground=True, rotation_offset=None,
                         ground_params_dir=None, force_horizontal=False, max_ground_angle=5.0,
                         perfectly_flat_ground=False, use_smart_ground_fit=False,
                         normalize_ground_to_zero=False, grid_adjust_ground=False,
                         grid_size=20, ground_percentile=5):
    """
    Create a textured 3D mesh from an image using depth estimation.
    This uses surface reconstruction to create a smooth mesh with texture.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the output mesh (optional)
        visualize: Whether to visualize the mesh
        remove_shadows: Whether to remove shadow artifacts
        interpolate_ground: Whether to interpolate ground plane in shadow areas
        remove_strays: Whether to remove stray points
        visualize_ground: Whether to visualize ground detection
        add_ground_plane: Whether to add a transparent ground plane to the visualization
        use_saved_ground: Whether to try loading saved ground plane parameters
        rotation_offset: Optional list of rotation angles in degrees [x_rot, y_rot, z_rot]
        ground_params_dir: Directory to save/load ground plane parameters
        force_horizontal: Whether to force the ground plane to be more horizontal
        max_ground_angle: Maximum allowed angle with horizontal plane in degrees
        perfectly_flat_ground: If True, make the plane perfectly horizontal
        use_smart_ground_fit: Whether to use the grid-based ground plane fitting
        normalize_ground_to_zero: Whether to normalize coordinates so ground plane is at y=0
        grid_adjust_ground: Whether to use grid-based approach to ensure points touch ground
        grid_size: Number of grid cells in each dimension for grid adjustment
        ground_percentile: Percentile of lowest points to consider for grid adjustment
    
    Returns:
        None
    """
    # Implementation delegating to the direct method with poisson reconstruction
    create_mesh_from_video3d_pointcloud(
        image_path, output_path, visualize, remove_shadows, 
        interpolate_ground, remove_strays, visualize_ground,
        add_ground_plane, use_saved_ground, rotation_offset,
        ground_params_dir, force_horizontal, max_ground_angle, 
        perfectly_flat_ground, use_smart_ground_fit,
        normalize_ground_to_zero, grid_adjust_ground,
        grid_size, ground_percentile
    )

def create_simple_mesh(image_path, output_path=None, visualize=True, remove_shadows=True,
                       interpolate_ground=True, remove_strays=True, visualize_ground=False,
                       add_ground_plane=True, use_saved_ground=True, rotation_offset=None,
                       ground_params_dir=None, force_horizontal=False, max_ground_angle=5.0,
                       perfectly_flat_ground=False, use_smart_ground_fit=False,
                       normalize_ground_to_zero=False, grid_adjust_ground=False,
                       grid_size=20, ground_percentile=5):
    """
    Create a simple 3D mesh from an image using depth estimation.
    This uses basic triangulation of the point cloud without smoothing.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the output mesh (optional)
        visualize: Whether to visualize the mesh
        remove_shadows: Whether to remove shadow artifacts
        interpolate_ground: Whether to interpolate ground plane in shadow areas
        remove_strays: Whether to remove stray points
        visualize_ground: Whether to visualize ground detection
        add_ground_plane: Whether to add a transparent ground plane to the visualization
        use_saved_ground: Whether to try loading saved ground plane parameters
        rotation_offset: Optional list of rotation angles in degrees [x_rot, y_rot, z_rot]
        ground_params_dir: Directory to save/load ground plane parameters
        force_horizontal: Whether to force the ground plane to be more horizontal
        max_ground_angle: Maximum allowed angle with horizontal plane in degrees
        perfectly_flat_ground: If True, make the plane perfectly horizontal
        use_smart_ground_fit: Whether to use the grid-based ground plane fitting
        normalize_ground_to_zero: Whether to normalize coordinates so ground plane is at y=0
        grid_adjust_ground: Whether to use grid-based approach to ensure points touch ground
        grid_size: Number of grid cells in each dimension for grid adjustment
        ground_percentile: Percentile of lowest points to consider for grid adjustment
    
    Returns:
        None
    """
    # Implementation delegating to the direct method with simple mesh creation
    create_mesh_from_video3d_pointcloud(
        image_path, output_path, visualize, remove_shadows, 
        interpolate_ground, remove_strays, visualize_ground,
        add_ground_plane, use_saved_ground, rotation_offset,
        ground_params_dir, force_horizontal, max_ground_angle, 
        perfectly_flat_ground, use_smart_ground_fit,
        normalize_ground_to_zero, grid_adjust_ground,
        grid_size, ground_percentile
    )

# Add this helper function for computing 3D points from depth
def depth_to_3d(depth_in, focallength_px, width, height):
    """
    Convert a depth map to 3D points in world coordinates.
    
    Args:
        depth_in: The depth map
        focallength_px: The focal length of the camera, in pixels
        width: Width of the depth map
        height: Height of the depth map
        
    Returns:
        points_3d: Nx3 array of 3D points in world coordinate system where Y points up
        valid_mask: Boolean mask of valid depth values
    """
    # Convert depth map to numpy
    depth_np = np.asarray(depth_in)
    
    # Create grid of pixel coordinates
    y_indices, x_indices = np.indices((height, width))
    
    # Center of the image
    cx = width / 2
    cy = height / 2
    
    # Create depth values and filter invalid depths
    valid_mask = ~np.isnan(depth_np) & (depth_np > 0)
    z = depth_np[valid_mask].flatten()
    
    # Calculate the 3D coordinates
    x = (x_indices[valid_mask] - cx) * z / focallength_px
    # Invert Y so that it points up (positive Y is up in world coordinates)
    y = -1 * (y_indices[valid_mask] - cy) * z / focallength_px
    
    # Stack the coordinates into a 3D point cloud
    points_3d = np.column_stack((x, y, z))
    return points_3d, valid_mask

def point_plane_distances(normal, d, points):
    """
    Calculate signed distances from points to a plane.
    Positive distances indicate points above the plane, negative below.
    
    Args:
        normal: Normal vector of the plane [a, b, c]
        d: Distance parameter of the plane equation ax + by + cz + d = 0
        points: Nx3 array of 3D points
        
    Returns:
        distances: Array of signed distances from each point to the plane
    """
    # Ensure normal is normalized
    normal = np.asarray(normal)
    normal = normal / np.linalg.norm(normal)
    
    # Calculate signed distances using the plane equation: ax + by + cz + d = 0
    # For each point (x, y, z), the signed distance is: (ax + by + cz + d) / sqrt(a² + b² + c²)
    # Since the normal is normalized, we can simplify to: ax + by + cz + d
    return np.dot(points, normal) + d

def normalize_point_cloud_to_ground(points_3d, ground_model):
    """
    Normalizes the point cloud coordinates so the ground plane is at exactly y = 0.
    Resets any points that fall below y = 0 to y = 0.
    
    Args:
        points_3d: Nx3 array of 3D points
        ground_model: Dictionary containing ground plane parameters
        
    Returns:
        normalized_points: Nx3 array of normalized 3D points with ground at y = 0
    """
    # Ensure points_3d is a numpy array
    points_3d = np.asarray(points_3d)
    
    # Reconstruct the implicit plane equation: ax + by + cz + d = 0
    # where [a, b, c] is the normal vector
    normal = ground_model['normal']
    d = ground_model['d']
    
    # Calculate signed distances from all points to the plane
    # Positive values are above the plane, negative below
    distances = point_plane_distances(normal, d, points_3d)
    
    # To normalize, we need to transform the coordinates so the plane becomes y = 0
    # First, we need to find the transformation that aligns the plane normal with [0, 1, 0]
    # (the y-axis normal, since y points up in our world coordinate system)
    
    # Create a rotation matrix that would rotate the ground normal to [0, 1, 0]
    from_vec = normal
    to_vec = np.array([0, 1, 0])
    
    # If the normal is already close to [0, 1, 0], skip rotation
    if np.abs(np.dot(from_vec, to_vec)) > 0.99:
        # Just use a translation
        normalized_points = points_3d.copy()
    else:
        # Calculate rotation matrix using the formula for rotation between two vectors
        from_vec = from_vec / np.linalg.norm(from_vec)
        
        # Cross product to get rotation axis
        axis = np.cross(from_vec, to_vec)
        axis = axis / np.linalg.norm(axis)
        
        # Dot product to get angle
        angle = np.arccos(np.clip(np.dot(from_vec, to_vec), -1.0, 1.0))
        
        # Create the rotation matrix using Rodrigues' rotation formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        
        # Apply rotation to all points
        normalized_points = (R @ points_3d.T).T
        
        # Update the distances based on the rotated points
        # In the rotated space, the plane equation becomes y = constant
        # Calculate this constant based on the rotated plane
        rotated_normal = R @ normal
        const = -d / rotated_normal[1]  # Using y component
        
        # Calculate adjusted heights (y coordinates)
        normalized_points[:, 1] = normalized_points[:, 1] - const
    
    # Now, any points with y < 0 should be reset to y = 0
    below_ground_mask = normalized_points[:, 1] < 0
    normalized_points[below_ground_mask, 1] = 0.0
    
    print(f"Normalized point cloud: {len(points_3d)} points")
    print(f"Points adjusted to y=0: {np.sum(below_ground_mask)}")
    
    return normalized_points

def grid_based_ground_adjustment(points_3d, grid_size=20, percentile=5):
    """
    Implements a grid-based approach to ensure points properly touch the ground plane at y = 0.
    For each grid cell (in the XZ plane):
    1. Find the lowest 5% of points
    2. Measure how high these points are above y = 0
    3. Lower all points in that cell by that height
    4. Ensure no points fall below y = 0
    
    Args:
        points_3d: Nx3 array of 3D points with ground already normalized to y = 0
        grid_size: Number of grid cells in each dimension (X and Z)
        percentile: The percentile of lowest points to consider (e.g., 5 for lowest 5%)
        
    Returns:
        adjusted_points: Nx3 array of adjusted 3D points
    """
    points = np.asarray(points_3d)
    adjusted_points = points.copy()
    
    # Extract X, Y, Z coordinates
    x_coords = points[:, 0]
    y_coords = points[:, 1]  # Y is already normalized to have ground at y = 0
    z_coords = points[:, 2]
    
    # Determine bounds for X and Z to create the grid
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    
    # Create the grid
    x_edges = np.linspace(x_min, x_max, grid_size + 1)
    z_edges = np.linspace(z_min, z_max, grid_size + 1)
    
    # Initialize a counter for adjusted points
    total_adjusted = 0
    cells_with_points = 0
    cells_adjusted = 0
    
    # Visualize grid and adjustments (for debugging)
    if grid_size <= 30:  # Only visualize for reasonable grid sizes
        plt.figure(figsize=(12, 10))
        plt.scatter(x_coords, z_coords, c=y_coords, s=1, alpha=0.5, cmap='viridis')
        plt.colorbar(label='Height (y)')
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.title('Points colored by height before adjustment')
        
        # Draw grid lines
        for x in x_edges:
            plt.axvline(x=x, color='r', linestyle='-', alpha=0.3)
        for z in z_edges:
            plt.axhline(y=z, color='r', linestyle='-', alpha=0.3)
        
        plt.savefig('grid_before_adjustment.png')
        plt.close()
    
    # Process each grid cell
    for i in range(grid_size):
        for j in range(grid_size):
            # Define cell boundaries
            x_min_cell, x_max_cell = x_edges[i], x_edges[i+1]
            z_min_cell, z_max_cell = z_edges[j], z_edges[j+1]
            
            # Find points in this cell
            cell_mask = (x_coords >= x_min_cell) & (x_coords < x_max_cell) & \
                       (z_coords >= z_min_cell) & (z_coords < z_max_cell)
            
            cell_points_indices = np.where(cell_mask)[0]
            
            # Skip if cell has too few points (less than 10)
            if len(cell_points_indices) < 10:
                continue
            
            cells_with_points += 1
            
            # Get y-coordinates of points in this cell
            cell_y = y_coords[cell_points_indices]
            
            # Find the y-height at the given percentile (e.g., the 5th percentile)
            y_percentile = np.percentile(cell_y, percentile)
            
            # If the lowest points are significantly above y=0, adjust the cell
            # Only adjust if the gap is more than 0.01 units (to avoid numerical issues)
            if y_percentile > 0.01:
                # Calculate adjustment (shift everything down by y_percentile)
                cells_adjusted += 1
                
                # Apply the adjustment to all points in this cell
                adjusted_points[cell_points_indices, 1] -= y_percentile
                
                # Ensure no points go below y=0
                below_ground = adjusted_points[cell_points_indices, 1] < 0
                adjusted_points[cell_points_indices[below_ground], 1] = 0
                
                # Count adjusted points
                total_adjusted += len(cell_points_indices)
    
    print(f"Grid adjustment summary:")
    print(f"  - Grid size: {grid_size}x{grid_size}")
    print(f"  - Cells with sufficient points: {cells_with_points}/{grid_size*grid_size}")
    print(f"  - Cells requiring adjustment: {cells_adjusted}")
    print(f"  - Total points adjusted: {total_adjusted}")
    
    # Visualize after adjustment (for debugging)
    if grid_size <= 30:  # Only visualize for reasonable grid sizes
        plt.figure(figsize=(12, 10))
        plt.scatter(adjusted_points[:, 0], adjusted_points[:, 2], c=adjusted_points[:, 1], 
                   s=1, alpha=0.5, cmap='viridis')
        plt.colorbar(label='Height (y)')
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.title('Points colored by height after adjustment')
        
        # Draw grid lines
        for x in x_edges:
            plt.axvline(x=x, color='r', linestyle='-', alpha=0.3)
        for z in z_edges:
            plt.axhline(y=z, color='r', linestyle='-', alpha=0.3)
        
        plt.savefig('grid_after_adjustment.png')
        plt.close()
    
    return adjusted_points

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create a 3D mesh from a depth map")
    
    # Input/output arguments
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save the output mesh")
    
    # Processing options
    parser.add_argument("--method", type=str, default="direct", choices=["direct", "poisson", "simple"],
                        help="Mesh creation method (direct, poisson, simple)")
    parser.add_argument("--no_visualize", action="store_false", dest="visualize",
                        help="Don't visualize the mesh")
    parser.add_argument("--no_remove_shadows", action="store_false", dest="remove_shadows",
                        help="Don't remove shadow artifacts")
    parser.add_argument("--no_interpolate_ground", action="store_false", dest="interpolate_ground",
                        help="Don't interpolate ground plane in shadow areas")
    parser.add_argument("--no_remove_strays", action="store_false", dest="remove_strays",
                        help="Don't remove stray points")
    parser.add_argument("--visualize_ground", action="store_true",
                        help="Visualize ground detection")
    parser.add_argument("--no_add_ground_plane", action="store_false", dest="add_ground_plane",
                        help="Don't add a transparent ground plane to the visualization")
    parser.add_argument("--no_saved_ground", action="store_false", dest="use_saved_ground",
                        help="Don't use saved ground plane parameters")
    
    # Rotation options
    parser.add_argument("--rot_x", type=float, default=0.0,
                        help="Rotation angle around X axis (degrees)")
    parser.add_argument("--rot_y", type=float, default=0.0,
                        help="Rotation angle around Y axis (degrees)")
    parser.add_argument("--rot_z", type=float, default=0.0,
                        help="Rotation angle around Z axis (degrees)")
    
    # Ground plane parameters directory
    parser.add_argument("--ground_params_dir", type=str, default=None,
                        help="Directory to save/load ground plane parameters")
    
    # Floor plan options
    parser.add_argument("--create_floor_plan", action="store_true",
                        help="Create a floor plan from the depth map")
    parser.add_argument("--floor_plan_path", type=str, default="floor_plan.png",
                        help="Path to save the floor plan")
    parser.add_argument("--height_threshold", type=float, default=0.5,
                        help="Height threshold in meters above ground plane (objects above this will be included in floor plan)")
    parser.add_argument("--simplified_floor_plan", action="store_true",
                        help="Create a simplified floor plan with cleaner outlines")
    parser.add_argument("--color_by_height", action="store_true",
                        help="Color the floor plan by height")
    parser.add_argument("--max_height", type=float, default=2.5,
                        help="Maximum height in meters for color mapping")
    parser.add_argument("--show_text", action="store_true",
                        help="Show text labels on the floor plan")
    
    # Add argument for ground plane optimization
    parser.add_argument("--no_optimize_ground", action="store_false", dest="optimize_ground",
                        help="Don't optimize ground plane orientation")
    
    # Add arguments for forcing ground plane to be more horizontal
    parser.add_argument("--force_horizontal", action="store_true",
                        help="Force ground plane to be more horizontal")
    parser.add_argument("--max_ground_angle", type=float, default=5.0,
                        help="Maximum allowed angle with horizontal plane in degrees")
    parser.add_argument("--perfectly_flat", action="store_true", dest="perfectly_flat_ground",
                        help="Make ground plane perfectly horizontal")
    
    # Add argument for smart ground plane fitting
    parser.add_argument("--legacy_ground", action="store_false", dest="use_smart_ground_fit",
                        help="Use legacy ground plane detection instead of Z-binning approach")
    
    # Add arguments for ground normalization and grid-based adjustment
    parser.add_argument("--normalize_ground", action="store_true", dest="normalize_ground_to_zero",
                        help="Normalize coordinates so ground plane is exactly at y=0")
    parser.add_argument("--grid_adjust", action="store_true", dest="grid_adjust_ground",
                        help="Use grid-based approach to ensure points touch the ground properly")
    parser.add_argument("--grid_size", type=int, default=20,
                        help="Number of grid cells in each dimension for grid adjustment (default: 20)")
    parser.add_argument("--ground_percentile", type=int, default=5,
                        help="Percentile of lowest points to consider for ground adjustment (default: 5)")
    
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
    
    # Print message about ground plane approach
    if args.use_smart_ground_fit:
        print("Using improved Z-binning ground plane detection (default)")
    else:
        print("Using legacy ground plane detection (--legacy_ground flag enabled)")
    
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
            ground_params_dir=args.ground_params_dir,
            optimize_ground=args.optimize_ground,
            force_horizontal=args.force_horizontal,
            max_ground_angle=args.max_ground_angle,
            perfectly_flat_ground=args.perfectly_flat_ground,
            use_smart_ground_fit=args.use_smart_ground_fit,
            visualize_ground=args.visualize_ground
        )
    
    # Create floor plan if requested
    if args.create_floor_plan:
        if ground_model is None:
            print("Cannot create floor plan without ground plane detection")
            print("Detecting ground plane...")
            ground_mask, ground_model = detect_ground_plane(depth_np)
            
            # Apply rotation offset if provided
            if rotation_offset is not None:
                ground_model = apply_rotation_to_plane(ground_model, rotation_offset)
        
        create_floor_plan(
            depth_np, 
            np.array(image), 
            ground_model, 
            focallength_px.item(),
            height_threshold=args.height_threshold,
            output_path=args.floor_plan_path,
            visualize=args.visualize,
            simplified=args.simplified_floor_plan,
            color_by_height=args.color_by_height,
            max_height=args.max_height,
            show_text=args.show_text
        )
    
    # Continue with mesh creation if not only creating floor plan
    if args.method == "direct":
        create_mesh_from_video3d_pointcloud(
            args.image_path, args.output_path, args.visualize, 
            args.remove_shadows, args.interpolate_ground, args.remove_strays, 
            args.visualize_ground, args.add_ground_plane, args.use_saved_ground, rotation_offset,
            args.ground_params_dir, args.force_horizontal, args.max_ground_angle, args.perfectly_flat_ground,
            args.use_smart_ground_fit, args.normalize_ground_to_zero, args.grid_adjust_ground, 
            args.grid_size, args.ground_percentile
        )
    elif args.method == "poisson":
        create_textured_mesh(
            args.image_path, args.output_path, args.visualize, 
            args.remove_shadows, args.interpolate_ground, args.remove_strays, 
            args.visualize_ground, args.add_ground_plane, args.use_saved_ground, rotation_offset,
            args.ground_params_dir, args.force_horizontal, args.max_ground_angle, args.perfectly_flat_ground,
            args.use_smart_ground_fit, args.normalize_ground_to_zero, args.grid_adjust_ground, 
            args.grid_size, args.ground_percentile
        )
    else:
        create_simple_mesh(
            args.image_path, args.output_path, args.visualize, 
            args.remove_shadows, args.interpolate_ground, args.remove_strays, 
            args.visualize_ground, args.add_ground_plane, args.use_saved_ground, rotation_offset,
            args.ground_params_dir, args.force_horizontal, args.max_ground_angle, args.perfectly_flat_ground,
            args.use_smart_ground_fit, args.normalize_ground_to_zero, args.grid_adjust_ground, 
            args.grid_size, args.ground_percentile
        ) 