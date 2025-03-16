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

def detect_ground_plane(depth_map, threshold=0.05, max_iterations=500, lower_region_height=0.4, visualize_steps=False):
    """
    Detect the ground plane in a depth map using optimized RANSAC in 3D space.
    
    Args:
        depth_map: Numpy array containing the depth map
        threshold: Threshold for RANSAC inlier detection
        max_iterations: Maximum number of RANSAC iterations
        lower_region_height: Proportion of the image height to consider as the lower region
        visualize_steps: Whether to visualize intermediate steps
        
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

def remove_depth_shadows(depth_map, image=None, threshold_factor=0.2, min_region_size=100, interpolate_ground=True, 
                     image_path=None, use_saved_ground=True, rotation_offset=None, ground_params_dir=None):
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
        
    Returns:
        processed_depth: Processed depth map with shadows removed
        ground_mask: Boolean mask of ground pixels (if ground interpolation is enabled)
        ground_model: The fitted ground plane model (if ground interpolation is enabled)
    """
    # Create a copy of the depth map
    processed_depth = depth_map.copy()
    
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
    
    # Detect ground plane if interpolation is enabled
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
            else:
                # If loading failed, detect ground plane
                ground_mask, ground_model = detect_ground_plane(depth_map)
                
                # Apply rotation offset if provided
                if rotation_offset is not None and len(rotation_offset) == 3:
                    ground_model = apply_rotation_to_plane(ground_model, rotation_offset)
                
                # Save ground plane parameters for future use
                if image_path is not None and use_saved_ground:
                    save_ground_plane_params(ground_model, image_path, ground_params_dir)
        else:
            # Detect ground plane
            ground_mask, ground_model = detect_ground_plane(depth_map)
            
            # Apply rotation offset if provided
            if rotation_offset is not None and len(rotation_offset) == 3:
                ground_model = apply_rotation_to_plane(ground_model, rotation_offset)
            
            # Save ground plane parameters for future use
            if image_path is not None and use_saved_ground:
                save_ground_plane_params(ground_model, image_path, ground_params_dir)
        
        # Create a mask for shadow areas
        shadow_mask = ~valid_mask
        
        # Interpolate ground in shadow areas
        if ground_model is not None:
            h, w = depth_map.shape
            y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            
            # Convert to 3D coordinates
            cx, cy = w / 2, h / 2
            focal_length_px = max(w, h)  # Estimate focal length
            
            # Create array for shadow points that need interpolation
            shadow_ground_mask = shadow_mask & ground_mask
            if np.sum(shadow_ground_mask) > 0:
                print(f"Interpolating depth for {np.sum(shadow_ground_mask)} shadow ground points")
                
                # For each shadow point on the ground, calculate its 3D position
                # and project it onto the ground plane
                shadow_y, shadow_x = np.where(shadow_ground_mask)
                
                for i in range(len(shadow_y)):
                    y_img = shadow_y[i]
                    x_img = shadow_x[i]
                    
                    # Estimate a reasonable depth at this position
                    # Use neighboring valid depths or median of ground depths
                    local_depths = []
                    for dy in range(-3, 4):
                        for dx in range(-3, 4):
                            ny, nx = y_img + dy, x_img + dx
                            if 0 <= ny < h and 0 <= nx < w and valid_mask[ny, nx]:
                                local_depths.append(depth_map[ny, nx])
                    
                    if len(local_depths) > 0:
                        # Use local median depth
                        estimated_depth = np.median(local_depths)
                    else:
                        # Use median depth of all ground points
                        ground_depths = depth_map[ground_mask & valid_mask]
                        if len(ground_depths) > 0:
                            estimated_depth = np.median(ground_depths)
                        else:
                            continue  # Skip if no reference depths available
                    
                    # Calculate 3D position
                    x_3d = (x_img - cx) * estimated_depth / focal_length_px
                    z_3d = estimated_depth
                    
                    # Use ground plane equation to find y_3d
                    normal = ground_model['normal']
                    d = ground_model['d']
                    
                    # If normal[1] is not close to zero, we can calculate y
                    if abs(normal[1]) > 1e-6:
                        y_3d = (-normal[0] * x_3d - normal[2] * z_3d - d) / normal[1]
                        
                        # Convert back to depth
                        adjusted_depth = z_3d
                        
                        # Assign the interpolated depth
                        processed_depth[y_img, x_img] = adjusted_depth
            
            # Update valid mask to include interpolated ground
            valid_mask = valid_mask | (shadow_ground_mask & (processed_depth > 0))
    
    # Apply the mask to the depth map (set non-valid regions to zero)
    processed_depth[~valid_mask] = 0
    
    # Optional: fill small holes
    processed_depth = ndimage.median_filter(processed_depth, size=3)
    
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

def create_mesh_from_video3d_pointcloud(image_path, output_path=None, visualize=True, remove_shadows=True, 
                                        interpolate_ground=True, remove_strays=True, visualize_ground=False,
                                        add_ground_plane=True, use_saved_ground=True, rotation_offset=None,
                                        ground_params_dir=None):
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
    
    Returns:
        o3d.geometry.TriangleMesh: The created mesh
    """
    # First, create the point cloud exactly as in video3D.py
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
            ground_params_dir=ground_params_dir
        )
        
        # Visualize ground detection if requested
        if visualize_ground and ground_mask is not None:
            visualize_ground_detection(depth.detach().cpu().numpy(), ground_mask, np.array(image))
    
    # Get image dimensions
    h, w = depth_np.shape
    
    # Calculate principal point (assuming center of image)
    cx, cy = w / 2, h / 2
    
    # Create meshgrid of pixel coordinates
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Convert pixel coordinates to camera coordinates
    Z = depth_np
    X = (x_coords - cx) * Z / focallength_px.item()
    Y = (y_coords - cy) * Z / focallength_px.item()
    
    # Stack coordinates and reshape to Nx3 array
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    
    # Get colors from original image
    colors = np.array(image).reshape(-1, 3) / 255.0
    
    # Remove points with zero depth (shadows that were removed)
    valid_indices = Z.reshape(-1) > 0
    points = points[valid_indices]
    colors = colors[valid_indices]
    
    # Remove points that are too far away
    far_mask = points[:, 2] < 100
    points = points[far_mask]
    colors = colors[far_mask]
    
    # Remove stray points if requested
    if remove_strays and len(points) > 0:
        points, colors = remove_stray_points(points, colors)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Save a copy of the point cloud for visualization
    pcd_for_viz = copy.deepcopy(pcd)
    
    # Now create a mesh from this point cloud
    print("Creating mesh from point cloud...")
    
    # Downsample the point cloud for faster mesh creation
    pcd_down = pcd.voxel_down_sample(voxel_size=0.05)
    
    # Estimate normals
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd_down.orient_normals_towards_camera_location()
    
    # Try different mesh creation methods
    
    # Method 1: Ball pivoting (preserves exact point positions)
    radii = [0.05, 0.1, 0.2, 0.4]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd_down, o3d.utility.DoubleVector(radii)
    )
    
    # Compute vertex normals for better visualization
    mesh.compute_vertex_normals()
    
    # Create ground plane mesh if requested
    ground_plane = None
    if add_ground_plane and ground_mask is not None and ground_model is not None:
        ground_plane = create_ground_plane_mesh(
            depth.detach().cpu().numpy(), 
            ground_mask, 
            ground_model, 
            focallength_px.item(),
            rotation_offset=rotation_offset
        )
    
    # Save mesh if output path is provided
    if output_path:
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"Mesh saved to {output_path}")
    
    # Visualize mesh and point cloud together
    if visualize:
        vis_objects = [mesh, pcd_for_viz]
        if ground_plane is not None and len(np.asarray(ground_plane.vertices)) > 0:
            vis_objects.append(ground_plane)
            
        o3d.visualization.draw_geometries(
            vis_objects,
            mesh_show_back_face=True,
            mesh_show_wireframe=False,
            point_show_normal=False
        )
    
    return mesh

def create_textured_mesh(image_path, output_path=None, visualize=True, remove_shadows=True,
                         interpolate_ground=True, remove_strays=True, visualize_ground=False,
                         add_ground_plane=True, use_saved_ground=True, rotation_offset=None,
                         ground_params_dir=None):
    """
    Convert a 2D image to a textured 3D mesh using Depth Pro.
    Uses Poisson surface reconstruction for a more detailed mesh.
    Uses exactly the same dimensions as the point cloud in video3D.py.
    
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
    
    Returns:
        o3d.geometry.TriangleMesh: The created mesh
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
    
    # Convert depth to numpy array - EXACTLY as in video3D.py
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
            ground_params_dir=ground_params_dir
        )
        
        # Visualize ground detection if requested
        if visualize_ground and ground_mask is not None:
            visualize_ground_detection(depth.detach().cpu().numpy(), ground_mask, np.array(image))
    
    # Get image dimensions
    h, w = depth_np.shape
    
    # Calculate principal point (assuming center of image)
    cx, cy = w / 2, h / 2
    
    # Create meshgrid of pixel coordinates - EXACTLY as in video3D.py
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Convert pixel coordinates to camera coordinates - EXACTLY as in video3D.py
    Z = depth_np
    X = (x_coords - cx) * Z / focallength_px.item()
    Y = (y_coords - cy) * Z / focallength_px.item()
    
    # Create point cloud directly - EXACTLY as in video3D.py
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    colors = np.array(image).reshape(-1, 3) / 255.0
    
    # Remove points with zero depth (shadows that were removed)
    valid_indices = Z.reshape(-1) > 0
    points = points[valid_indices]
    colors = colors[valid_indices]
    
    # Remove points that are too far away
    far_mask = points[:, 2] < 100
    points = points[far_mask]
    colors = colors[far_mask]
    
    # Remove stray points if requested
    if remove_strays and len(points) > 0:
        points, colors = remove_stray_points(points, colors)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Estimate normals for better mesh reconstruction
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_towards_camera_location()
    
    # Create mesh from point cloud
    print("Creating mesh from point cloud...")
    
    # Method 1: Poisson reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9, width=0, scale=1.1, linear_fit=False
    )
    
    # Remove low-density vertices
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # Transfer color from point cloud to mesh
    mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Default color
    
    # Compute vertex normals for better visualization
    mesh.compute_vertex_normals()
    
    # Create ground plane mesh if requested
    ground_plane = None
    if add_ground_plane and ground_mask is not None and ground_model is not None:
        ground_plane = create_ground_plane_mesh(
            depth.detach().cpu().numpy(), 
            ground_mask, 
            ground_model, 
            focallength_px.item(),
            rotation_offset=rotation_offset
        )
    
    # Save mesh if output path is provided
    if output_path:
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"Mesh saved to {output_path}")
    
    # Visualize mesh
    if visualize:
        vis_objects = [mesh]
        if ground_plane is not None and len(np.asarray(ground_plane.vertices)) > 0:
            vis_objects.append(ground_plane)
            
        o3d.visualization.draw_geometries(
            vis_objects,
            mesh_show_back_face=True,
            mesh_show_wireframe=False,
            point_show_normal=False
        )
    
    return mesh

def create_simple_mesh(image_path, output_path=None, visualize=True, remove_shadows=True,
                       interpolate_ground=True, remove_strays=True, visualize_ground=False,
                       add_ground_plane=True, use_saved_ground=True, rotation_offset=None,
                       ground_params_dir=None):
    """
    Create a simple height map mesh from a depth image.
    Uses exactly the same dimensions as the point cloud in video3D.py.
    
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
    
    # Convert to numpy - EXACTLY as in video3D.py
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
            ground_params_dir=ground_params_dir
        )
        
        # Visualize ground detection if requested
        if visualize_ground and ground_mask is not None:
            visualize_ground_detection(depth.detach().cpu().numpy(), ground_mask, np.array(image))
    
    # Create a height map mesh
    h, w = depth_np.shape
    
    # Downsample for faster processing if needed
    scale_factor = 1
    if max(h, w) > 1000:
        scale_factor = 4
    elif max(h, w) > 500:
        scale_factor = 2
        
    if scale_factor > 1:
        h_ds = h // scale_factor
        w_ds = w // scale_factor
        depth_np = cv2.resize(depth_np, (w_ds, h_ds))
        image = cv2.resize(np.array(image), (w_ds, h_ds))
        if ground_mask is not None:
            ground_mask = cv2.resize(ground_mask.astype(np.uint8), (w_ds, h_ds)) > 0
        h, w = h_ds, w_ds
    
    # Calculate principal point (assuming center of image)
    cx, cy = w / 2, h / 2
    
    # Create meshgrid of pixel coordinates - EXACTLY as in video3D.py
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Convert pixel coordinates to camera coordinates - EXACTLY as in video3D.py
    Z = depth_np
    X = (x_coords - cx) * Z / focallength_px.item()
    Y = (y_coords - cy) * Z / focallength_px.item()
    
    # Create vertices and colors
    vertices = []
    colors = []
    
    # Create a mask for valid points (non-shadow)
    valid_mask = Z > 0 if remove_shadows else np.ones_like(Z, dtype=bool)
    
    # Flatten for mesh creation
    for i in range(h):
        for j in range(w):
            if valid_mask[i, j]:
                vertices.append([X[i, j], Y[i, j], Z[i, j]])
                colors.append(image[i, j] / 255.0)
    
    # If we have no valid vertices, return an empty mesh
    if len(vertices) == 0:
        print("No valid vertices found after shadow removal.")
        mesh = o3d.geometry.TriangleMesh()
        return mesh
    
    # Convert to numpy arrays
    vertices = np.array(vertices)
    colors = np.array(colors)
    
    # Remove stray points if requested
    if remove_strays and len(vertices) > 0:
        vertices, colors = remove_stray_points(vertices, colors)
        
        # If all points were removed as strays, return empty mesh
        if len(vertices) == 0:
            print("No valid vertices left after stray point removal.")
            mesh = o3d.geometry.TriangleMesh()
            return mesh
    
    # Create a point cloud for triangulation
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_towards_camera_location()
    
    # Create mesh using Alpha shapes or Ball pivoting
    # This works better than the grid-based approach when we have removed points
    radii = [0.05, 0.1, 0.2, 0.4]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    
    # Compute normals
    mesh.compute_vertex_normals()
    
    # Create ground plane mesh if requested
    ground_plane = None
    if add_ground_plane and ground_mask is not None and ground_model is not None:
        ground_plane = create_ground_plane_mesh(
            depth.detach().cpu().numpy(), 
            ground_mask, 
            ground_model, 
            focallength_px.item(),
            rotation_offset=rotation_offset
        )
    
    # Save mesh if output path is provided
    if output_path:
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"Mesh saved to {output_path}")
    
    # Visualize mesh
    if visualize:
        vis_objects = [mesh]
        if ground_plane is not None and len(np.asarray(ground_plane.vertices)) > 0:
            vis_objects.append(ground_plane)
            
        o3d.visualization.draw_geometries(
            vis_objects,
            mesh_show_back_face=True,
            mesh_show_wireframe=False,
            point_show_normal=False
        )
    
    return mesh

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert 2D image to 3D mesh")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_path", type=str, help="Path to save mesh (optional)")
    parser.add_argument("--method", type=str, default="direct", 
                        choices=["direct", "poisson", "simple"], 
                        help="Mesh creation method: 'direct' (from point cloud), 'poisson', or 'simple'")
    parser.add_argument("--no_visualize", action="store_false", dest="visualize", 
                        help="Disable visualization")
    parser.add_argument("--keep_shadows", action="store_false", dest="remove_shadows",
                        help="Keep shadow artifacts (don't remove them)")
    parser.add_argument("--no_ground_interp", action="store_false", dest="interpolate_ground",
                        help="Don't interpolate ground in shadow areas")
    parser.add_argument("--keep_strays", action="store_false", dest="remove_strays",
                        help="Keep stray points (don't remove them)")
    parser.add_argument("--visualize_ground", action="store_true",
                        help="Visualize ground plane detection")
    parser.add_argument("--no_ground_plane", action="store_false", dest="add_ground_plane",
                        help="Don't add transparent ground plane to visualization")
    parser.add_argument("--shadow_threshold", type=float, default=0.2,
                        help="Threshold for shadow detection (0.1-0.5)")
    parser.add_argument("--no_saved_ground", action="store_false", dest="use_saved_ground",
                        help="Don't use saved ground plane parameters (always detect)")
    
    # Use separate arguments for each rotation axis
    parser.add_argument("--rot_x", type=float, default=0.0,
                        help="Rotation around X axis in degrees (can be negative)")
    parser.add_argument("--rot_y", type=float, default=0.0,
                        help="Rotation around Y axis in degrees (can be negative)")
    parser.add_argument("--rot_z", type=float, default=0.0,
                        help="Rotation around Z axis in degrees (can be negative)")
    
    parser.add_argument("--ground_params_dir", type=str,
                        help="Directory to save/load ground plane parameters (defaults to image directory)")
    
    args = parser.parse_args()
    
    # Create rotation offset from individual axis rotations
    rotation_offset = [args.rot_x, args.rot_y, args.rot_z]
    
    # Only use rotation if at least one axis has a non-zero rotation
    if rotation_offset == [0.0, 0.0, 0.0]:
        rotation_offset = None
    else:
        print(f"Using rotation offset: {rotation_offset[0]}, {rotation_offset[1]}, {rotation_offset[2]} degrees")
    
    if args.method == "direct":
        create_mesh_from_video3d_pointcloud(
            args.image_path, args.output_path, args.visualize, 
            args.remove_shadows, args.interpolate_ground, args.remove_strays, 
            args.visualize_ground, args.add_ground_plane, args.use_saved_ground, rotation_offset,
            args.ground_params_dir
        )
    elif args.method == "poisson":
        create_textured_mesh(
            args.image_path, args.output_path, args.visualize, 
            args.remove_shadows, args.interpolate_ground, args.remove_strays, 
            args.visualize_ground, args.add_ground_plane, args.use_saved_ground, rotation_offset,
            args.ground_params_dir
        )
    else:
        create_simple_mesh(
            args.image_path, args.output_path, args.visualize, 
            args.remove_shadows, args.interpolate_ground, args.remove_strays, 
            args.visualize_ground, args.add_ground_plane, args.use_saved_ground, rotation_offset,
            args.ground_params_dir
        ) 