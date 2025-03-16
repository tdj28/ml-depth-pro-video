#!/usr/bin/env python3
"""
image_to_ground_normalized_pointcloud.py

Takes an image, detects the ground plane, normalizes the point cloud so the ground plane 
is at y=0, applies grid-based ground contact adjustment, and outputs a PLY point cloud file.

This is a streamlined version of functionality from mesh_from_depth.py.
"""

from PIL import Image
import depth_pro
import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt
import argparse
import os
import sys
from pathlib import Path
import json
import scipy.ndimage as ndimage
import cv2
from tqdm import tqdm
from scipy.spatial import KDTree
from sklearn.linear_model import RANSACRegressor

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
    points_3d[:, 1] = -1 * (y_img - cy) * z / focal_length_px  # Y (negative because Y is up in world space)
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
    
    # RANSAC iterations
    for i in range(max_iterations):
        # Sample 3 points to define a plane, with weighting
        sample_indices = np.random.choice(len(points_3d), 3, replace=False, p=weights)
        p1, p2, p3 = points_3d[sample_indices]
        
        # Calculate plane normal using cross product
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        
        # Skip if normal is too small (points nearly collinear)
        if np.linalg.norm(normal) < 1e-10:
            continue
        
        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)
        
        # Ensure normal points up (towards negative Y in camera space)
        if normal[1] > 0:
            normal = -normal
        
        # Calculate plane equation: ax + by + cz + d = 0
        # d = -(normal . point)
        d = -np.dot(normal, p1)
        
        # Calculate distances from all points to the plane
        dists = point_plane_distances(normal, d, points_3d)
        
        # Find inliers within threshold
        inlier_mask = np.abs(dists) < threshold
        num_inliers = np.sum(inlier_mask)
        
        # Update best plane if we found more inliers
        if num_inliers > most_inliers:
            most_inliers = num_inliers
            best_plane = (normal, d)
            best_inlier_mask = inlier_mask
    
    # If no good plane was found, return None
    if best_plane is None:
        print("Failed to find ground plane")
        return None, None
    
    normal, d = best_plane
    inlier_fraction = most_inliers / len(points_3d)
    print(f"  Found plane with {most_inliers} inliers ({inlier_fraction:.1%})")
    print(f"Ground plane normal: [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}]")
    
    # Calculate angle with horizontal plane
    horizontal_vector = np.array([0, -1, 0])  # Negative Y is "horizontal" in camera space
    cos_angle = np.abs(np.dot(normal, horizontal_vector))
    angle_degrees = np.arccos(cos_angle) * 180 / np.pi
    print(f"Angle with horizontal: {angle_degrees:.2f} degrees")
    
    # Create a ground model to return
    ground_model = {
        'normal': normal,
        'd': d,
        'origin': np.array([0, 0, 0])  # Will be updated if optimized
    }
    
    # Refine the plane fit using all inlier points
    if optimize and np.sum(best_inlier_mask) >= 10:  # Only optimize if we have enough inliers
        ground_model = optimize_ground_plane(points_3d, ground_model)
    
    # Create a ground mask for the full image
    ground_mask = np.zeros((h, w), dtype=bool)
    
    # Mark inlier pixels as ground
    for i, is_inlier in enumerate(best_inlier_mask):
        if is_inlier:
            y, x = point_to_pixel[i]
            ground_mask[y, x] = True
    
    # Dilate the ground mask to make it more continuous
    ground_mask = ndimage.binary_dilation(ground_mask, iterations=2)
    
    return ground_mask, ground_model

def optimize_ground_plane(points_3d, ground_model, min_points_above=0.95, visualize=False):
    """
    Optimize the ground plane parameters to ensure most points are above the plane,
    while keeping the plane as close as possible to the actual ground points.
    
    Args:
        points_3d: Nx3 array of 3D points
        ground_model: Initial ground plane model from RANSAC
        min_points_above: Minimum fraction of points that should be above the plane
        visualize: Whether to visualize the optimization process
        
    Returns:
        optimized_model: Dictionary containing optimized ground plane parameters
    """
    # Get plane parameters from the model
    initial_normal = ground_model['normal']
    initial_d = ground_model['d']
    
    # Convert normal to spherical coordinates for optimization
    # This ensures the normal stays normalized during optimization
    theta = np.arccos(initial_normal[1])  # Angle from Y axis
    phi = np.arctan2(initial_normal[2], initial_normal[0])  # Angle in XZ plane
    
    # Starting parameters: [theta, phi, d_offset]
    initial_params = [theta, phi, 0.0]  # Start with no offset to d
    
    # Select points for optimization (a smaller random sample for speed)
    max_points = 10000
    if len(points_3d) > max_points:
        indices = np.random.choice(len(points_3d), max_points, replace=False)
        points_3d = points_3d[indices]
    
    print(f"Optimizing ground plane with {len(points_3d)} points...")
    
    # Identify potential ground points (lower points in each vertical column)
    # This helps bias the optimization towards the true ground
    x_bins = 20
    z_bins = 20
    x_min, x_max = np.min(points_3d[:, 0]), np.max(points_3d[:, 0])
    z_min, z_max = np.min(points_3d[:, 2]), np.max(points_3d[:, 2])
    
    # Create bin edges
    x_edges = np.linspace(x_min, x_max, x_bins + 1)
    z_edges = np.linspace(z_min, z_max, z_bins + 1)
    
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
        
        # Calculate mean squared distance for points above the plane
        above_plane = all_distances > 0
        if np.sum(above_plane) > 0:
            mse_above = np.mean(all_distances[above_plane] ** 2)
        else:
            mse_above = 0
        
        # Calculate angle deviation from original plane (to avoid drastic changes)
        angle_deviation = np.arccos(np.clip(np.dot(normal, initial_normal), -1.0, 1.0))
        angle_penalty = 5.0 * angle_deviation  # Scale to make it comparable
        
        # Total cost: MSE above + angle penalty + points below penalty
        total_cost = mse_above + angle_penalty + penalty
        
        return total_cost
    
    # Find potential ground points in each bin
    ground_point_mask = np.zeros(len(points_3d), dtype=bool)
    
    for i in range(x_bins):
        for j in range(z_bins):
            # Get points in this bin
            bin_mask = ((points_3d[:, 0] >= x_edges[i]) & (points_3d[:, 0] < x_edges[i+1]) &
                        (points_3d[:, 2] >= z_edges[j]) & (points_3d[:, 2] < z_edges[j+1]))
            
            if np.sum(bin_mask) > 0:
                # Get the lowest 20% of points in this bin
                bin_points = points_3d[bin_mask]
                y_values = bin_points[:, 1]  # Y-coordinates
                lower_threshold = np.percentile(y_values, 20)  # Lower 20%
                
                # Mark these as potential ground points
                ground_points_in_bin = bin_mask & (points_3d[:, 1] <= lower_threshold)
                ground_point_mask = ground_point_mask | ground_points_in_bin
    
    print(f"Identified {np.sum(ground_point_mask)} potential ground points out of {len(points_3d)} total points")
    
    # Run the optimization with bounds to limit how much the plane can change
    from scipy.optimize import minimize
    bounds = [(theta - 0.3, theta + 0.3),  # Limit theta to ±0.3 radians (≈17°)
              (phi - 0.3, phi + 0.3),      # Limit phi to ±0.3 radians
              (-0.5, 0.5)]                 # Limit d offset to ±0.5 meters
    
    result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100})
    
    # Get the optimized plane parameters
    optimized_normal, optimized_d = params_to_plane(result.x)
    
    # Compare before and after
    initial_distances = point_plane_distances(initial_normal, initial_d, points_3d)
    optimized_distances = point_plane_distances(optimized_normal, optimized_d, points_3d)
    
    initial_below = np.sum(initial_distances < 0) / len(points_3d)
    optimized_below = np.sum(optimized_distances < 0) / len(points_3d)
    
    # Calculate angle change
    angle_change = np.arccos(np.clip(np.dot(optimized_normal, initial_normal), -1.0, 1.0)) * 180 / np.pi
    
    # Calculate angle with horizontal for both planes
    horizontal_vector = np.array([0, -1, 0])  # Negative Y is horizontal in camera space
    initial_angle = np.arccos(np.abs(np.dot(initial_normal, horizontal_vector))) * 180 / np.pi
    optimized_angle = np.arccos(np.abs(np.dot(optimized_normal, horizontal_vector))) * 180 / np.pi
    
    # Calculate d parameter change
    d_change = np.abs(optimized_d - initial_d)
    
    # Print optimization results
    print("Ground plane optimization results:")
    print(f"  Initial points below: {initial_below:.2%}")
    print(f"  Optimized points below: {optimized_below:.2%}")
    print(f"  Angle change: {angle_change:.2f} degrees")
    print(f"  Initial angle with horizontal: {initial_angle:.2f} degrees")
    print(f"  Optimized angle with horizontal: {optimized_angle:.2f} degrees")
    print(f"  Distance parameter change: {d_change:.4f} meters")
    
    # Create optimized ground model
    optimized_model = {
        'normal': optimized_normal,
        'd': optimized_d,
        'origin': np.array([0, 0, 0])  # Default origin
    }
    
    return optimized_model

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
    
    # If already near horizontal or below max_angle, and not forcing perfectly flat, return as is
    if current_angle < max_angle and not completely_flat:
        print(f"Ground plane already within {max_angle} degrees of horizontal (angle = {current_angle:.2f})")
        return ground_model
    
    # Create a new normal vector
    if completely_flat:
        # Make perfectly horizontal
        new_normal = np.array([0, -1, 0]) if normal[1] > 0 else np.array([0, 1, 0])
        print("Setting ground plane to perfectly horizontal")
    else:
        # Adjust to max_angle
        # First find the axis to rotate around (perpendicular to both current normal and horizontal)
        rotation_axis = np.cross(normal, horizontal_vector)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        
        # Angle to rotate by
        rotate_angle = np.radians(current_angle - max_angle)
        
        # Create rotation matrix using Rodriguez formula
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        R = np.eye(3) + np.sin(rotate_angle) * K + (1 - np.cos(rotate_angle)) * (K @ K)
        
        # Apply rotation to normal
        new_normal = R @ normal
        print(f"Adjusting ground plane to maximum {max_angle} degrees from horizontal")
    
    # Recalculate d parameter for the plane equation: ax + by + cz + d = 0
    # d = -(normal . origin)
    new_d = -np.dot(new_normal, origin)
    
    # Create new ground model
    adjusted_model = {
        'normal': new_normal,
        'd': new_d,
        'origin': origin
    }
    
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
        z_start = z_min + i * bin_size
        z_end = z_start + bin_size
        
        # Find points in this Z bin
        bin_indices = np.where((z_coords >= z_start) & (z_coords < z_end))[0]
        
        # If we have enough points in this bin
        if len(bin_indices) > 10:
            # Find the lowest 5% of points (lowest Y values)
            bin_y_values = y_coords[bin_indices]
            threshold = np.percentile(bin_y_values, 5)
            
            # Get the indices of the lowest points
            lowest_indices = bin_indices[bin_y_values <= threshold]
            
            # Take a sample of the lowest points to avoid having too many points in one bin
            if len(lowest_indices) > 5:
                lowest_indices = np.random.choice(lowest_indices, 5, replace=False)
                
            # Add these points to our ground trace
            for idx in lowest_indices:
                ground_trace_points.append(points[idx])
    
    # If we don't have enough ground trace points, return initial model or a default
    if len(ground_trace_points) < 3:
        print("Not enough ground trace points found. Using fallback approach.")
        if initial_ground_model is not None:
            return initial_ground_model
        else:
            # Fallback to a default horizontal ground plane at minimum Y
            min_y = np.min(y_coords)
            return {
                'normal': np.array([0, 1, 0]),
                'd': -min_y,
                'origin': np.array([0, min_y, 0])
            }
    
    # Convert ground trace points to numpy array
    ground_trace_points = np.array(ground_trace_points)
    
    print(f"Found {len(ground_trace_points)} ground trace points from Z bins")
    
    # Fit a plane to the ground trace points using RANSAC
    from sklearn.linear_model import RANSACRegressor
    
    # Check if force_horizontal is enabled
    if force_horizontal:
        # Create a perfectly horizontal plane at the lowest point
        min_y = np.min(ground_trace_points[:, 1])
        horizontal_model = {
            'normal': np.array([0, 1, 0]),
            'd': -min_y,
            'origin': np.array([0, min_y, 0])
        }
        print(f"Forced horizontal ground plane at y = {min_y:.4f}")
        return horizontal_model
    
    # Otherwise, use RANSAC to fit an arbitrary plane
    try:
        # If we have a plane from RANSAC, extract parameters
        # Use the ground trace points and build a model with X, Z as features and Y as target
        X = ground_trace_points[:, [0, 2]]  # X and Z coordinates
        y = ground_trace_points[:, 1]       # Y coordinates
        
        # Fit the model using RANSAC
        ransac = RANSACRegressor(min_samples=10, residual_threshold=0.05, max_trials=100)
        ransac.fit(X, y)
        
        # Extract plane parameters from the model
        a = ransac.estimator_.coef_[0]  # X coefficient
        c = ransac.estimator_.coef_[1]  # Z coefficient
        b = -1.0                        # Y coefficient (normalized to -1)
        d = ransac.estimator_.intercept_ # Constant term
        
        # Create normal vector and normalize it
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)
        
        # Calculate angle with horizontal
        horizontal_vector = np.array([0, 1, 0])  # Y points up
        cos_angle = np.abs(np.dot(normal, horizontal_vector))
        angle_degrees = np.arccos(cos_angle) * 180 / np.pi
        
        print(f"RANSAC fit with angle {angle_degrees:.2f}° from horizontal")
        
        # If angle is too steep, revert to a horizontal fit
        if angle_degrees > 45.0:
            print(f"Plane angle {angle_degrees:.2f}° exceeds threshold. Using fallback horizontal fit.")
            # Calculate the average Y-value of the ground trace for the horizontal plane
            avg_y = np.median(ground_trace_points[:, 1])
            
            # Create a horizontal ground model
            horizontal_model = {
                'normal': np.array([0, 1, 0]),
                'd': -avg_y,
                'origin': np.array([0, avg_y, 0])
            }
            ground_model = horizontal_model
        else:
            # Build the ground model
            # We need to adjust d because we normalized the normal vector
            normal_norm = np.sqrt(a**2 + b**2 + c**2)
            adjusted_d = d * b / normal_norm
            
            # Compute a point on the plane (0, d/b, 0) as the origin
            origin = np.array([0, adjusted_d / b, 0])
            
            ground_model = {
                'normal': normal,
                'd': adjusted_d,
                'origin': origin
            }
        
        # Validate the ground model by ensuring no points are below the plane
        # Calculate distances from all points to the plane
        dists = point_plane_distances(ground_model['normal'], ground_model['d'], points)
        
        # Count points below the plane (negative distances)
        points_below = np.sum(dists < 0)
        print(f"Found {points_below} points below the plane. Adjusting plane down.")
        
        # If there are points below the plane, adjust the d value
        if points_below > 0:
            # Get the lowest point
            min_dist = np.min(dists)
            
            # Adjust d to ensure no points are below
            adjusted_d = ground_model['d'] - min_dist - 0.01  # Add a small safety margin
            ground_model['d'] = adjusted_d
            
            # Recompute origin
            if np.abs(ground_model['normal'][1]) > 0.01:  # Avoid division by zero
                y_intercept = -adjusted_d / ground_model['normal'][1]
                ground_model['origin'] = np.array([0, y_intercept, 0])
            
            # Verify the adjustment worked
            dists_after = point_plane_distances(ground_model['normal'], ground_model['d'], points)
            points_below_after = np.sum(dists_after < 0)
            print(f"After adjustment: {points_below_after} points below the plane")
        
        return ground_model
        
    except Exception as e:
        print(f"Error in RANSAC ground fitting: {e}")
        # Fallback to a default horizontal ground plane
        if initial_ground_model is not None:
            return initial_ground_model
        else:
            # Create a horizontal ground plane at the minimum Y value
            min_y = np.min(ground_trace_points[:, 1])
            return {
                'normal': np.array([0, 1, 0]),
                'd': -min_y,
                'origin': np.array([0, min_y, 0])
            }

def get_ground_model_from_depth(depth_map, image_path=None, ground_params_dir=None, 
                              force_horizontal=False, use_smart_ground_fit=True, focallength_px=None):
    """
    Extract ground plane model from a depth map.
    This is a standalone implementation that doesn't import from mesh_from_depth.py.
    
    Args:
        depth_map: The depth map as numpy array
        image_path: Path to the original image (for saving ground params)
        ground_params_dir: Directory to save ground plane parameters
        force_horizontal: Whether to force the ground plane to be horizontal
        use_smart_ground_fit: Whether to use Z-binning approach
        focallength_px: Focal length in pixels, used for 3D conversion
        
    Returns:
        ground_model: Dictionary containing ground plane parameters
    """
    # Detect ground plane
    print("Detecting ground plane...")
    ground_mask, ground_model = detect_ground_plane(depth_map, optimize=True)
    
    # Force horizontal if requested
    if force_horizontal and ground_model is not None:
        print("Forcing ground plane to be more horizontal...")
        ground_model = force_horizontal_ground(ground_model)
    
    # Use improved Z-binning if requested
    if use_smart_ground_fit and ground_model is not None:
        print("Using smart grid-based ground plane fitting...")
        h, w = depth_map.shape
        
        # Use provided focal length or default if not provided
        if focallength_px is None:
            print("Warning: No focal length provided for Z-binning. Using default value.")
            focallength_px = 1000.0  # Default
        
        # Create 3D points for Z-binning approach
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        Z = depth_map.copy()  # Make a copy to avoid modifying the original
        
        # Filter out invalid depth values
        valid_mask = ~np.isnan(Z) & (Z > 0)
        
        # Calculate principal point (center of image)
        cx, cy = w / 2, h / 2
        
        # Convert to 3D coordinates
        X = np.zeros_like(Z)
        Y = np.zeros_like(Z)
        
        # Only calculate for valid depths
        X[valid_mask] = (x_coords[valid_mask] - cx) * Z[valid_mask] / focallength_px
        Y[valid_mask] = -1 * (y_coords[valid_mask] - cy) * Z[valid_mask] / focallength_px  # Negative because Y is up
        
        # Stack into 3D points
        points = np.stack((X[valid_mask], Y[valid_mask], Z[valid_mask]), axis=-1)
        
        # Apply Z-binning ground fit
        print(f"Z-binning using {len(points)} points with focal length {focallength_px}px")
        ground_model = grid_based_ground_plane_fit(points, initial_ground_model=ground_model, 
                                                 force_horizontal=force_horizontal)
    
    # Save ground plane parameters if requested
    if ground_model is not None and image_path is not None and ground_params_dir is not None:
        # Implement save ground plane parameters directly without importing
        save_ground_plane_params(ground_model, image_path, ground_params_dir)
    
    return ground_model

def save_ground_plane_params(ground_model, image_path, output_dir=None):
    """
    Save ground plane parameters to a JSON file.
    
    Args:
        ground_model: Dictionary containing ground plane parameters
        image_path: Path to the original image
        output_dir: Directory to save the parameters file (defaults to same as image)
    """
    if ground_model is None:
        print("No ground model to save")
        return
    
    # Create a serializable version of the ground model
    serializable_model = {
        'normal': ground_model['normal'].tolist(),
        'd': float(ground_model['d']),
        'origin': ground_model['origin'].tolist()
    }
    
    # Generate the output filename based on the image name
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = f"{image_basename}_ground_params.json"
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    
    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Full path to output file
    output_path = os.path.join(output_dir, output_filename)
    
    # Save the ground model parameters
    with open(output_path, 'w') as f:
        json.dump(serializable_model, f, indent=2)
    
    print(f"Ground plane parameters saved to {output_path}")

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

def grid_based_ground_adjustment(points_3d, grid_size=20, percentile=5, visualize=False):
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
        visualize: Whether to create visualization plots
        
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
    if visualize and grid_size <= 30:  # Only visualize for reasonable grid sizes
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
    if visualize and grid_size <= 30:  # Only visualize for reasonable grid sizes
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

def remove_stray_points(points, colors, min_neighbors=3, radius=0.2):
    """
    Remove stray points that don't have enough neighbors within a certain radius.
    
    Args:
        points: Nx3 array of 3D points
        colors: Nx3 array of point colors
        min_neighbors: Minimum number of neighbors required
        radius: Radius to search for neighbors
        
    Returns:
        filtered_points: Filtered points with stray points removed
        filtered_colors: Corresponding colors
    """
    if len(points) == 0:
        return points, colors
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Build KD tree for nearest neighbor search
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    # Count neighbors for each point
    keep_indices = []
    for i in range(len(points)):
        [k, idx, _] = kdtree.search_radius_vector_3d(pcd.points[i], radius)
        if k > min_neighbors:
            keep_indices.append(i)
    
    # Filter points and colors
    filtered_points = points[keep_indices]
    filtered_colors = colors[keep_indices]
    
    print(f"Removed {len(points) - len(filtered_points)} stray points out of {len(points)} total points")
    return filtered_points, filtered_colors

def create_normalized_pointcloud(image_path, output_path=None, grid_size=20, ground_percentile=5,
                                normalize_ground=True, grid_adjust=True, remove_strays=True,
                                force_horizontal=False, use_smart_ground_fit=True, 
                                ground_params_dir=None, visualize=False, downsample_factor=1):
    """
    Main function to create a normalized point cloud from an image.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the output point cloud (default: based on input name)
        grid_size: Number of grid cells in each dimension for grid adjustment
        ground_percentile: Percentile of lowest points to consider for ground adjustment
        normalize_ground: Whether to normalize coordinates so ground plane is at y=0
        grid_adjust: Whether to use grid-based ground contact adjustment
        remove_strays: Whether to remove stray points
        force_horizontal: Whether to force the ground plane to be horizontal
        use_smart_ground_fit: Whether to use Z-binning approach for ground fitting
        ground_params_dir: Directory to save/load ground plane parameters
        visualize: Whether to visualize the point cloud and create visualizations
        downsample_factor: Factor to downsample the point cloud (for faster processing)
        
    Returns:
        The path to the saved PLY file
    """
    # Generate default output path if not provided
    if output_path is None:
        base_path = Path(image_path).stem
        output_path = f"{base_path}_normalized.ply"
    
    # Load model and preprocessing transform
    print(f"Loading depth model...")
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()

    # Load and preprocess the image
    print(f"Processing image: {image_path}")
    image, _, f_px = depth_pro.load_rgb(image_path)
    image_tensor = transform(image)

    # Run inference
    print(f"Running depth inference...")
    prediction = model.infer(image_tensor, f_px=f_px)
    depth = prediction["depth"]  # Depth in [m]
    focallength_px = prediction["focallength_px"]  # Focal length in pixels
    
    # Store the original focal length
    original_focallength = focallength_px.item()
    
    # Convert depth to numpy array
    depth_np = depth.detach().cpu().numpy()
    
    # Get initial dimensions
    h, w = depth_np.shape
    
    # Downsample if requested
    if downsample_factor > 1:
        h_ds = h // downsample_factor
        w_ds = w // downsample_factor
        print(f"Downsampling from {w}x{h} to {w_ds}x{h_ds}...")
        depth_np = cv2.resize(depth_np, (w_ds, h_ds))
        image = cv2.resize(np.array(image), (w_ds, h_ds))
        
        # Critical: adjust focal length when downsampling
        focallength_px = original_focallength / downsample_factor
        print(f"Adjusting focal length from {original_focallength} to {focallength_px} due to downsampling")
        
        h, w = h_ds, w_ds
    else:
        image = np.array(image)
        focallength_px = original_focallength
    
    # Get ground model
    ground_model = get_ground_model_from_depth(
        depth_np, 
        image_path=image_path, 
        ground_params_dir=ground_params_dir,
        force_horizontal=force_horizontal,
        use_smart_ground_fit=use_smart_ground_fit,
        focallength_px=focallength_px
    )
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    
    # Use depth_to_3d function for consistent conversion 
    # This ensures Y is up in the world coordinate system
    print(f"Converting depth to 3D points (focal length: {focallength_px}px)...")
    points_3d, valid_mask = depth_to_3d(depth_np, focallength_px, w, h)
    
    # Reshape the RGB image and get colors for valid points
    colors = image.reshape(-1, 3)[valid_mask.flatten()] / 255.0
    
    # Remove stray points if requested
    if remove_strays and len(points_3d) > 0:
        print(f"Removing stray points...")
        points_3d, colors = remove_stray_points(points_3d, colors)
    
    # Apply ground plane normalization if requested and ground model is available
    if normalize_ground and ground_model is not None:
        print(f"Normalizing point cloud so ground plane is at y=0...")
        points_3d = normalize_point_cloud_to_ground(points_3d, ground_model)
        
        # After normalization, update ground_model to reflect y=0 plane
        ground_model = {
            'normal': np.array([0, 1, 0]),
            'origin': np.array([0, 0, 0]),
            'd': 0
        }
        
        # Apply grid-based adjustment if requested
        if grid_adjust:
            print(f"Applying grid-based ground adjustment...")
            points_3d = grid_based_ground_adjustment(
                points_3d, 
                grid_size=grid_size, 
                percentile=ground_percentile,
                visualize=visualize
            )
    
    # Add points and colors to the point cloud
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Save the point cloud as PLY
    print(f"Saving normalized point cloud to {output_path}...")
    o3d.io.write_point_cloud(output_path, pcd)
    
    # Visualize the point cloud if requested
    if visualize:
        print(f"Visualizing point cloud...")
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        # Add the point cloud to the visualizer
        vis.add_geometry(pcd)
        
        # Add a simple ground plane grid for reference
        if normalize_ground:
            # Create a grid on the ground plane for better orientation
            grid_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=1.0, origin=[0, 0, 0])
            vis.add_geometry(grid_mesh)
            
            # Create a ground plane mesh for reference
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
            vis.add_geometry(ground_mesh)
        
        # Set visualization options
        opt = vis.get_render_option()
        opt.point_size = 2.0
        
        # Set view control options - position camera to look at scene
        ctrl = vis.get_view_control()
        ctrl.set_up([0, -1, 0])  # Set Y axis as up direction
        
        # Show the visualization
        vis.run()
        vis.destroy_window()
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Create a normalized point cloud with ground plane at y=0")
    
    # Input/output arguments
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save the output point cloud (default: based on input name)")
    
    # Processing options
    parser.add_argument("--no_normalize_ground", action="store_false", dest="normalize_ground",
                        help="Don't normalize coordinates so ground plane is at y=0")
    parser.add_argument("--no_grid_adjust", action="store_false", dest="grid_adjust",
                        help="Don't use grid-based ground contact adjustment")
    parser.add_argument("--no_remove_strays", action="store_false", dest="remove_strays",
                        help="Don't remove stray points")
    parser.add_argument("--force_horizontal", action="store_true",
                        help="Force ground plane to be more horizontal")
    parser.add_argument("--legacy_ground", action="store_false", dest="use_smart_ground_fit",
                        help="Use legacy ground plane detection instead of Z-binning approach")
    parser.add_argument("--grid_size", type=int, default=20,
                        help="Number of grid cells in each dimension for grid adjustment (default: 20)")
    parser.add_argument("--ground_percentile", type=int, default=5,
                        help="Percentile of lowest points to consider for grid adjustment (default: 5)")
    parser.add_argument("--ground_params_dir", type=str, default=None,
                        help="Directory to save ground plane parameters")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the point cloud and create visualizations")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Downsample factor for faster processing (default: 1, no downsampling)")
    
    args = parser.parse_args()
    
    try:
        # Create normalized point cloud
        output_path = create_normalized_pointcloud(
            args.image_path,
            output_path=args.output_path,
            grid_size=args.grid_size,
            ground_percentile=args.ground_percentile,
            normalize_ground=args.normalize_ground,
            grid_adjust=args.grid_adjust,
            remove_strays=args.remove_strays,
            force_horizontal=args.force_horizontal,
            use_smart_ground_fit=args.use_smart_ground_fit,
            ground_params_dir=args.ground_params_dir,
            visualize=args.visualize,
            downsample_factor=args.downsample
        )
        
        print(f"Successfully created normalized point cloud: {output_path}")
        return 0
    except Exception as e:
        print(f"Error creating normalized point cloud: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 