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
import matplotlib

# Add this line to use a non-interactive backend for matplotlib
matplotlib.use('Agg')  

# Helper function to get the optimal torch device for the current system
def get_torch_device():
    """Get the optimal torch device for the system."""
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device

# Helper function to check if running on Apple Silicon
def is_apple_silicon():
    """Check if running on Apple Silicon."""
    import platform
    return platform.processor() == 'arm' and platform.system() == 'Darwin'

def render_point_cloud_to_image(pcd, output_path, width=1280, height=720, 
                             background_color=[1, 1, 1], view_preset="front",
                             multi_view=False):
    """
    Render a point cloud to a PNG image using Open3D visualization.
    
    Args:
        pcd: Open3D point cloud object
        output_path: Path to save the output PNG file
        width: Width of the output image in pixels (default: 1280)
        height: Height of the output image in pixels (default: 720)
        background_color: RGB background color as a list [r, g, b] with values in [0, 1]
        view_preset: Camera angle preset ("front", "top", "side", "isometric") (default: "front")
        multi_view: Whether to render multiple views and combine them (default: False)
        
    Returns:
        output_path: Path to the saved PNG file
    """
    # Ensure output path has .png extension
    if not output_path.lower().endswith('.png'):
        output_path = f"{os.path.splitext(output_path)[0]}.png"
    
    # Calculate bounding box to adjust view
    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    
    # Define view presets
    view_presets = {
        "front": {
            "front": [0, 0, -1],     # Look from negative Z
            "lookat": center,        # Look at center of object 
            "up": [0, 1, 0],         # Y is up (positive direction)
            "zoom": 0.4              # Zoom level (lower = closer)
        },
        "top": {
            "front": [0, -1, 0],     # Look from negative Y (top down)
            "lookat": center,
            "up": [0, 0, -1],        # Z is up from this view
            "zoom": 0.4              # Zoom level
        },
        "side": {
            "front": [1, 0, 0],      # Look from positive X
            "lookat": center,
            "up": [0, 1, 0],         # Y is up
            "zoom": 0.4              # Zoom level
        },
        "isometric": {
            "front": [1, -1, -1],    # 45 degree angle
            "lookat": center,
            "up": [0, 1, 0],         # Y is up
            "zoom": 0.35             # Zoom level (lower = closer)
        }
    }
    
    # If multi-view is requested, render all presets and combine them
    if multi_view:
        images = []
        preset_names = ["front", "top", "isometric", "side"]
        
        for preset_name in preset_names:
            # Create a visualization object
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=width, height=height, visible=False)
            
            # Add the point cloud to the visualization
            vis.add_geometry(pcd)
            
            # Set visualization options
            opt = vis.get_render_option()
            opt.background_color = np.asarray(background_color)
            opt.point_size = 7.0  # Larger points for better visibility
            opt.light_on = True
            
            # Set the camera viewpoint
            ctr = vis.get_view_control()
            preset = view_presets[preset_name]
            
            # Calculate proper front vector (normalize it)
            front = np.array(preset["front"])
            front = front / np.linalg.norm(front)
            
            ctr.set_front(front)
            ctr.set_lookat(preset["lookat"])
            ctr.set_up(preset["up"])
            ctr.set_zoom(preset["zoom"])
            
            # Update and render
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            
            # Capture the image
            image = vis.capture_screen_float_buffer(do_render=True)
            images.append(np.asarray(image))
            
            # Close the visualization
            vis.destroy_window()
        
        # Create a 2x2 grid of images
        grid_image = np.zeros((height*2, width*2, 3), dtype=np.float32)
        grid_image[0:height, 0:width] = images[0]  # Front (top-left)
        grid_image[0:height, width:width*2] = images[1]  # Top (top-right)
        grid_image[height:height*2, 0:width] = images[2]  # Isometric (bottom-left)
        grid_image[height:height*2, width:width*2] = images[3]  # Side (bottom-right)
        
        # Add labels
        label_offset = 30
        font_scale = min(width, height) / 1000.0
        font_thickness = max(2, int(font_scale * 3))
        label_color = (0, 0, 0)  # Black
        
        import cv2
        grid_image_cv = (grid_image * 255).astype(np.uint8)
        grid_image_cv = cv2.cvtColor(grid_image_cv, cv2.COLOR_RGB2BGR)
        
        # Add labels to the image
        cv2.putText(grid_image_cv, "Front View", (20, label_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, font_thickness)
        cv2.putText(grid_image_cv, "Top View", (width + 20, label_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, font_thickness)
        cv2.putText(grid_image_cv, "Isometric View", (20, height + label_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, font_thickness)
        cv2.putText(grid_image_cv, "Side View", (width + 20, height + label_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, font_thickness)
        
        # Convert back to RGB for saving
        grid_image_cv = cv2.cvtColor(grid_image_cv, cv2.COLOR_BGR2RGB)
        
        # Save the combined image
        plt.imsave(output_path, grid_image_cv)
        print(f"Multi-view point cloud visualization saved to {output_path}")
        return output_path
    
    # Single view rendering
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    
    # Add the point cloud to the visualization
    vis.add_geometry(pcd)
    
    # Set visualization options
    opt = vis.get_render_option()
    opt.background_color = np.asarray(background_color)
    opt.point_size = 7.0  # Larger points for better visibility
    opt.light_on = True
    
    # Set the camera viewpoint
    ctr = vis.get_view_control()
    
    # Use the specified preset
    if view_preset in view_presets:
        preset = view_presets[view_preset]
        
        # Calculate proper front vector (normalize it)
        front = np.array(preset["front"])
        front = front / np.linalg.norm(front)
        
        ctr.set_front(front)
        ctr.set_lookat(preset["lookat"])
        ctr.set_up(preset["up"])
        ctr.set_zoom(preset["zoom"])
    else:
        # Default to front view
        preset = view_presets["front"]
        ctr.set_front(preset["front"])
        ctr.set_lookat(preset["lookat"])
        ctr.set_up(preset["up"])
        ctr.set_zoom(preset["zoom"])
    
    # Update and render
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    
    # Capture the image
    image = vis.capture_screen_float_buffer(do_render=True)
    
    # Convert to numpy array and save
    image_np = np.asarray(image)
    plt.imsave(output_path, image_np)
    
    # Close the visualization
    vis.destroy_window()
    
    print(f"Point cloud visualization saved to {output_path}")
    return output_path

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
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output path - default to ground.json
    json_path = os.path.join(output_dir, "ground.json")
    
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
    # Set input directory to image directory if not specified
    if input_dir is None:
        input_dir = os.path.dirname(image_path)
    
    # Try to load from default ground.json first
    json_path = os.path.join(input_dir, "ground.json")
    
    # If the default file doesn't exist, try the legacy naming format
    if not os.path.exists(json_path):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        legacy_json_path = os.path.join(input_dir, f"{base_name}_ground_plane.json")
        if os.path.exists(legacy_json_path):
            json_path = legacy_json_path
    
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
    
    # Calculate the initial angle with horizontal plane
    horizontal = np.array([0, 1, 0])  # Y-up in world space
    init_cos_angle = np.dot(initial_normal, horizontal)
    
    # Ensure normal points "up" (positive Y)
    if init_cos_angle < 0:
        initial_normal = -initial_normal
        initial_d = -initial_d
        init_cos_angle = -init_cos_angle
    
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
        
        # Ensure normal maintains similar orientation to initial normal
        # This prevents flipping the normal vector which would cause people to disappear
        if np.dot(normal, initial_normal) < 0:
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
        horizontal = np.array([0, 1, 0])  # Y-up in world space
        cos_angle = np.abs(np.dot(normal, horizontal))
        angle_penalty = 2 * (1 - cos_angle)  # 0 when perfectly horizontal, 2 when vertical
        
        # Penalize large differences from initial model
        # This helps ensure we don't flip the plane orientation
        normal_diff = 1 - np.abs(np.dot(normal, initial_normal))
        diff_penalty = 50 * normal_diff
        
        # Combine all penalties
        total_objective = penalty + ground_penalty + variance_penalty + angle_penalty + diff_penalty
        
        return total_objective
    
    # Constraint: allow more angular change to find the correct orientation
    # But limit it to prevent completely flipping the plane
    max_angle_change = np.radians(20)  # Max 20 degrees change (was 30)
    
    # Bounds for parameters
    bounds = [
        (max(0, theta - max_angle_change), min(np.pi, theta + max_angle_change)),  # theta
        (phi - max_angle_change, phi + max_angle_change),      # phi
        (-0.3, 0.3)                                           # d_offset (in meters), reduced range
    ]
    
    # Run optimization
    result = minimize(
        objective, 
        initial_params, 
        method='L-BFGS-B', 
        bounds=bounds,
        options={'maxiter': 100}  # Reduced from 200 to avoid overoptimization
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
    horizontal = np.array([0, 1, 0])  # Y-up in world space
    initial_horizontal_angle = np.arccos(np.clip(np.dot(initial_normal, horizontal), -1.0, 1.0)) * 180 / np.pi
    optimized_horizontal_angle = np.arccos(np.clip(np.dot(optimized_normal, horizontal), -1.0, 1.0)) * 180 / np.pi
    
    print(f"Ground plane optimization results:")
    print(f"  Initial points below: {initial_below:.2%}")
    print(f"  Optimized points below: {optimized_below:.2%}")
    print(f"  Angle change: {angle:.2f} degrees")
    print(f"  Initial angle with horizontal: {initial_horizontal_angle:.2f} degrees")
    print(f"  Optimized angle with horizontal: {optimized_horizontal_angle:.2f} degrees")
    print(f"  Distance parameter change: {optimized_d - initial_d:.4f} meters")
    
    # Ensure we don't have too many points below plane after optimization
    # If optimization produced a worse result, revert to original
    if optimized_below > 0.3 or angle > 30:  # If more than 30% below or angle change is too large
        print(f"Optimization resulted in too many points below plane ({optimized_below:.1%}) or too large angle change ({angle:.1f}°).")
        print(f"Reverting to initial ground model.")
        return ground_model
    
    return optimized_model

def grid_based_ground_plane_fit(points_3d, initial_ground_model=None, grid_size=20, visualize=False):
    """
    Ground plane fitting using Z-binning approach:
    1. Divide the depth (Z-axis) into bins
    2. For each Z-bin, find the lowest points (in Y-axis) and use them as ground trace
    3. Fit a plane to these ground trace points, preferring a horizontal solution
    
    Args:
        points_3d: Nx3 array of 3D points
        initial_ground_model: Optional initial ground plane model
        grid_size: Number of bins for the Z direction
        visualize: Whether to visualize the fitting process
        
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
    
    # If we have an initial ground model, use it to filter points near the ground
    near_ground_points = None
    if initial_ground_model is not None:
        # Calculate distance from each point to the initial ground plane
        normal = initial_ground_model['normal']
        d = initial_ground_model['d']
        
        # Calculate signed distances
        distances = np.dot(points, normal) + d
        
        # Points within a reasonable distance of the ground plane (both above and below)
        # Using a larger threshold below the plane to ensure we catch the true ground
        # We use the scene scale to determine the threshold
        scene_scale = np.median(z_coords)
        threshold_above = 0.05 * scene_scale  # 5% of median depth above plane
        threshold_below = 0.10 * scene_scale  # 10% of median depth below plane
        
        # Select points that are near the ground plane
        near_ground_mask = (distances > -threshold_below) & (distances < threshold_above)
        near_ground_points = points[near_ground_mask]
        
        if len(near_ground_points) > 100:  # If we have enough near-ground points
            print(f"Using {len(near_ground_points)} points near initial ground plane")
            # Update points to use only the near-ground points
            points = near_ground_points
            # Update coordinates
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            z_coords = points[:, 2]
    
    # Create bin edges for fast digitize operation
    bin_edges = np.linspace(z_min, z_max, n_bins + 1)
    
    # Assign each point to a bin (much faster than iterating through bins)
    bin_indices = np.digitize(z_coords, bin_edges) - 1
    
    # Initialize arrays to store ground trace points
    ground_trace_points = []
    
    # Process each bin to find the lowest points (vectorized approach)
    for bin_idx in range(n_bins):
        # Get points in this bin
        bin_mask = (bin_indices == bin_idx)
        bin_count = np.sum(bin_mask)
        
        if bin_count > 10:  # Enough points for analysis
            # Get y-coordinates of points in this bin
            bin_y_values = y_coords[bin_mask]
            
            # Sort y-values and take the lowest 5%
            n_lowest = max(1, int(0.05 * bin_count))
            lowest_indices = np.argsort(bin_y_values)[:n_lowest]
            
            # Get the corresponding bin points
            bin_points = points[bin_mask][lowest_indices]
            
            # Calculate the average point
            avg_point = np.mean(bin_points, axis=0)
            ground_trace_points.append(avg_point)
    
    ground_trace_points = np.array(ground_trace_points)
    print(f"Found {len(ground_trace_points)} ground trace points from Z bins")
    
    if len(ground_trace_points) < 10:
        print("Warning: Too few ground trace points. Using fallback approach.")
        # Fall back to lowest 5% of all points by Y
        sorted_indices = np.argsort(y_coords)
        n_lowest = max(10, int(0.05 * len(points)))
        ground_trace_points = points[sorted_indices[:n_lowest]]
    
    # Now we have ground trace points, let's fit them
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
        
        # If we have an initial model, check if the new fit is reasonable
        if initial_ground_model is not None:
            # Calculate angle between initial and new normal
            initial_normal = initial_ground_model['normal']
            angle_with_initial = np.arccos(np.clip(np.dot(normal, initial_normal), -1.0, 1.0)) * 180 / np.pi
            
            # If the angle is too different, prefer the initial model
            if angle_with_initial > 20:
                print(f"New plane differs by {angle_with_initial:.2f}° from initial. Using initial plane.")
                normal = initial_normal
                d_plane = initial_ground_model['d']
        
        # If the angle is too steep (> 20 degrees), use a more horizontal approach
        if angle_degrees > 20:
            print(f"Plane angle {angle_degrees:.2f}° exceeds threshold. Using fallback horizontal fit.")
            normal = np.array([0, 1, 0])  # Y-up
            avg_y = np.median(ground_trace_points[:, 1])
            d_plane = -avg_y
    except Exception as e:
        print(f"RANSAC fitting error: {e}. Using horizontal plane.")
        normal = np.array([0, 1, 0])  # Y-up
        avg_y = np.median(ground_trace_points[:, 1])
        d_plane = -avg_y
    
    # Ensure normal points up (Y positive)
    if normal[1] < 0:
        normal = -normal
        d_plane = -d_plane
    
    # Ensure no points go below the plane (or only a very small number)
    # Calculate signed distances to the plane
    signed_distances = np.dot(points, normal) + d_plane
    
    # Check if there are points below the plane
    points_below = np.sum(signed_distances < 0)
    if points_below > 0:
        print(f"Found {points_below} points below the plane. Adjusting plane down.")
        
        # We want to move the plane down so that at most 0.1% of points are below it
        if points_below > 0.001 * len(points):
            # Find the percentile that gives us 0.1% of points below
            percentile_value = 0.1  # 0.1 percentile
            min_distance = np.percentile(signed_distances, percentile_value)
            
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
    
    return fitted_model

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
    # Negate x to preserve the correct left-right orientation from the original image
    # Without negation, left becomes right and vice versa in the point cloud
    x = -1 * (x_indices[valid_mask] - cx) * z / focallength_px
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
    
    # Calculate the minimum ground height after normalization
    # This helps ensure we're not burying points too deeply
    ground_y_values = normalized_points[np.abs(distances) < 0.1, 1]
    if len(ground_y_values) > 10:
        # Use a small percentile to avoid outliers but not bury everything
        ground_level = np.percentile(ground_y_values, 2)
        # Adjust all points up by this amount to ensure ground is at y=0
        # and most objects aren't buried
        normalized_points[:, 1] -= ground_level
    
    # Only adjust points that are very close to or below ground level
    # We use a threshold to identify clear ground points versus objects
    ground_threshold = 0.05  # 5cm threshold
    
    # First identify likely ground points (close to the ground plane)
    ground_mask = np.abs(distances) < ground_threshold
    
    # Calculate percentage of points identified as ground
    ground_percentage = np.sum(ground_mask) / len(points_3d)
    print(f"Identified {ground_percentage:.1%} of points as ground points")
    
    # For ground points below y=0, set them to exactly y=0
    below_ground_mask = (normalized_points[:, 1] < 0) & ground_mask
    normalized_points[below_ground_mask, 1] = 0.0
    
    # For non-ground points below ground level, only adjust if they're significantly below
    non_ground_below = (normalized_points[:, 1] < -0.1) & (~ground_mask)
    normalized_points[non_ground_below, 1] = -0.1  # Allow slight depth below ground
    
    print(f"Normalized point cloud: {len(points_3d)} points")
    print(f"Points adjusted to y=0: {np.sum(below_ground_mask)}")
    print(f"Non-ground points limited to -10cm: {np.sum(non_ground_below)}")
    
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
    
    # Create the grid edges
    x_edges = np.linspace(x_min, x_max, grid_size + 1)
    z_edges = np.linspace(z_min, z_max, grid_size + 1)
    
    # Initialize counters for reporting
    total_adjusted = 0
    cells_with_points = 0
    cells_adjusted = 0
    
    # Pre-digitize the points for faster grid cell assignment
    x_bin_indices = np.digitize(x_coords, x_edges) - 1
    z_bin_indices = np.digitize(z_coords, z_edges) - 1
    
    # Clamp indices to valid range (0 to grid_size-1)
    x_bin_indices = np.clip(x_bin_indices, 0, grid_size-1)
    z_bin_indices = np.clip(z_bin_indices, 0, grid_size-1)
    
    # Create a linear grid cell index for each point (allows vectorized operations)
    # Formula: cell_idx = x_idx * grid_size + z_idx
    cell_indices = x_bin_indices * grid_size + z_bin_indices
    
    # Get unique cell indices
    unique_cells = np.unique(cell_indices)
    
    # For reporting
    print(f"Grid adjustment: Processing {len(unique_cells)} non-empty cells")
    
    # Process each cell in parallel using numpy operations where possible
    for cell_idx in unique_cells:
        # Recover x and z indices
        x_idx = cell_idx // grid_size
        z_idx = cell_idx % grid_size
        
        # Get points in this cell
        cell_mask = (x_bin_indices == x_idx) & (z_bin_indices == z_idx)
        cell_point_indices = np.where(cell_mask)[0]
        
        # Skip if too few points
        if len(cell_point_indices) < 10:
            continue
        
        cells_with_points += 1
        
        # Get y-coordinates of points in this cell
        cell_y = y_coords[cell_point_indices]
        
        # Identify likely ground points in this cell (within 20cm of ground)
        lowest_points_mask = cell_y < 0.2
        
        # If we don't have enough low points, skip this cell
        if np.sum(lowest_points_mask) < 5:
            continue
            
        lowest_y = cell_y[lowest_points_mask]
        
        # Find the y-height at the given percentile among the lowest points
        if len(lowest_y) > 0:
            y_percentile = np.percentile(lowest_y, percentile)
        else:
            continue
        
        # If the lowest points are significantly above y=0, adjust the cell
        if y_percentile > 0.01:
            cells_adjusted += 1
            
            # Get the full indices of points in this cell
            cell_indices = cell_point_indices
            
            # Get y-values for all points in the cell
            cell_y_values = y_coords[cell_indices]
            
            # Calculate adjustment factors based on height 
            # Full adjustment for points near ground (< 10cm)
            # Graduated adjustment for points between 10cm and 1.5m
            # No adjustment for points above 1.5m
            
            # Start with an array of zeros for all points in the cell
            adjustment = np.zeros(len(cell_indices))
            
            # Apply full adjustment to points near ground level
            near_ground = cell_y_values < 0.1  # 10cm threshold
            adjustment[near_ground] = y_percentile
            
            # Apply graduated adjustment to points between 10cm and 1.5m
            mid_height = (cell_y_values >= 0.1) & (cell_y_values < 1.5)
            if np.any(mid_height):
                # Calculate adjustment factor (1.0 at 10cm, 0.0 at 1.5m)
                height_factor = 1.0 - (cell_y_values[mid_height] - 0.1) / 1.4
                adjustment[mid_height] = y_percentile * height_factor
            
            # Apply adjustments in one operation
            adjusted_points[cell_indices, 1] -= adjustment
            
            # Ensure no points go below y=0 (vectorized)
            below_ground = adjusted_points[cell_indices, 1] < 0
            adjusted_points[cell_indices[below_ground], 1] = 0.0
            
            # Count adjusted points
            total_adjusted += np.sum(adjustment > 0)
    
    print(f"Grid adjustment summary:")
    print(f"  - Grid size: {grid_size}x{grid_size}")
    print(f"  - Cells with sufficient points: {cells_with_points}/{grid_size*grid_size}")
    print(f"  - Cells requiring adjustment: {cells_adjusted}")
    print(f"  - Total points adjusted: {total_adjusted}")
    
    return adjusted_points

def create_normalized_pointcloud(image_path, output_path, 
                                rotation_offset=None, ground_params_dir=None, 
                                grid_size=20, ground_percentile=5,
                                downscale_factor=1.0, half_precision=False,
                                render_png=False, render_width=1280, render_height=720,
                                view_preset="front", multi_view=False, return_pointcloud=False):
    """
    Create a normalized point cloud from an image:
    1. Detect the ground plane
    2. Normalize points so the ground is at y=0
    3. Apply grid-based ground adjustment
    4. Save as PLY or render to PNG
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the output point cloud or PNG
        rotation_offset: Optional list of rotation angles in degrees [x_rot, y_rot, z_rot]
        ground_params_dir: Directory to save/load ground plane parameters
        grid_size: Number of grid cells in each dimension for grid adjustment (default: 20)
        ground_percentile: Percentile of lowest points to consider for grid adjustment (default: 5)
        downscale_factor: Downscale input image by this factor for faster processing (default: 1.0)
        half_precision: Use half precision (float16) for faster computation (default: False)
        render_png: Whether to render a PNG image instead of saving a PLY file (default: False)
        render_width: Width of the rendered PNG image in pixels (default: 1280)
        render_height: Height of the rendered PNG image in pixels (default: 720)
        view_preset: Camera angle preset for rendering (default: "front")
        multi_view: Whether to render multiple views of the point cloud in a grid (default: False)
        return_pointcloud: Whether to return the point cloud object (default: False)
    
    Returns:
        If return_pointcloud is True: tuple (pcd, colors) where pcd is the Open3D point cloud and colors are the point colors
        Otherwise: None - Saves the point cloud to output_path or renders it to a PNG
    """
    try:
        # Get the optimal device for the system
        device = get_torch_device()
        print(f"Using device: {device}")
        
        # Set precision - use half precision for faster computation on Apple Silicon
        precision = torch.float16 if (half_precision or device.type == 'mps') else torch.float32
        print(f"Using {precision} precision for computation")
        
        # Load model and preprocessing transform with device and precision specified
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
        
        # Set numpy to use multiple threads
        np.set_printoptions(precision=4, suppress=True)
        
        # Load the image
        image, _, f_px = depth_pro.load_rgb(image_path)
        
        # Downscale the image if requested (faster processing)
        orig_size = None
        if downscale_factor != 1.0 and downscale_factor > 0:
            orig_size = image.shape[:2]  # (height, width)
            new_height = int(orig_size[0] * downscale_factor)
            new_width = int(orig_size[1] * downscale_factor)
            print(f"Downscaling image from {orig_size[1]}x{orig_size[0]} to {new_width}x{new_height}")
            
            # Resize the image using cv2 for better quality
            image = cv2.resize(
                image, 
                (new_width, new_height), 
                interpolation=cv2.INTER_AREA if downscale_factor < 1.0 else cv2.INTER_LINEAR
            )
            
            # Adjust focal length based on the new size
            if f_px is not None:
                f_px = f_px * downscale_factor
        
        # Apply transform and run inference
        image_tensor = transform(image)

        # Run inference
        with torch.no_grad():
            prediction = model.infer(image_tensor, f_px=f_px)
            depth = prediction["depth"]  # Depth in [m]
            focallength_px = prediction["focallength_px"]  # Focal length in pixels
        
        # Convert depth to numpy array (make sure to move from GPU/MPS to CPU first)
        depth_np = depth.detach().cpu().numpy()
        
        # Get image dimensions
        h, w = depth_np.shape
        
        print(f"Depth map dimensions: {w}x{h}")
        print(f"Focal length: {focallength_px.item()} pixels")
        
        # Use the depth_to_3d function for consistent conversion 
        points_3d, valid_mask = depth_to_3d(depth_np, focallength_px.item(), w, h)
        
        print(f"Generated point cloud with {len(points_3d)} valid points")
        
        # Reshape the RGB image and get colors for valid points
        colors = np.array(image).reshape(-1, 3)[valid_mask.flatten()] / 255.0
        
        # Try to load saved ground parameters if available
        ground_model = None
        if ground_params_dir is not None:
            ground_model = load_ground_plane_params(image_path, ground_params_dir)
        else:
            # Default to looking in the same directory as the image
            ground_model = load_ground_plane_params(image_path)
        
        # If no saved ground parameters, detect the ground plane
        if ground_model is None:
            # Sample points for faster initial ground plane detection
            if len(points_3d) > 50000:
                sample_indices = np.random.choice(len(points_3d), 50000, replace=False)
                sample_points = points_3d[sample_indices]
            else:
                sample_points = points_3d
            
            # Initial Z-binning ground plane detection
            print("Using Z-binning ground plane detection...")
            initial_ground_model = grid_based_ground_plane_fit(sample_points)
            
            # Optimize the ground plane
            if initial_ground_model is not None:
                print(f"Optimizing ground plane with {len(sample_points)} points...")
                optimized_ground_model = optimize_ground_plane(sample_points, initial_ground_model)
            else:
                optimized_ground_model = None
            
            # Final ground plane detection with all points, using the optimized model as a guide
            print(f"Using grid-based ground plane fitting with {len(points_3d)} points...")
            ground_model = grid_based_ground_plane_fit(points_3d, initial_ground_model=optimized_ground_model)
            
            # Handle case where ground plane detection failed
            if ground_model is None:
                print("Warning: Failed to detect ground plane. Creating a default horizontal plane.")
                ground_model = {
                    'normal': np.array([0, 1, 0]),  # Y-up
                    'd': -np.min(points_3d[:, 1]) if len(points_3d) > 0 else -1.0,
                    'origin': np.array([0, np.min(points_3d[:, 1]) if len(points_3d) > 0 else 1.0, 0])
                }
        
        # Apply rotation offset if requested
        if rotation_offset is not None and ground_model is not None:
            ground_model = apply_rotation_to_plane(ground_model, rotation_offset)
        
        # Save ground model to default location (ground.json in the image directory) if not specified
        if ground_model is not None:
            if ground_params_dir is not None:
                save_ground_plane_params(ground_model, image_path, ground_params_dir)
            else:
                # Default to saving in the same directory as the image
                save_ground_plane_params(ground_model, image_path)
        
        # Normalize the point cloud (ground plane at y=0)
        print("Normalizing point cloud so ground plane is at y=0...")
        points_3d = normalize_point_cloud_to_ground(points_3d, ground_model)
        
        # Apply grid-based adjustment
        print(f"Applying grid-based ground adjustment (grid size: {grid_size}, percentile: {ground_percentile})...")
        points_3d = grid_based_ground_adjustment(points_3d, grid_size=grid_size, percentile=ground_percentile)
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # If requested to return the point cloud, do so
        if return_pointcloud:
            print("Returning point cloud without saving to disk")
            return pcd, colors
        
        # Either render to PNG or save as PLY
        if render_png:
            # Ensure output path ends with .png
            if not output_path.lower().endswith('.png'):
                output_path = f"{os.path.splitext(output_path)[0]}.png"
            
            # Render the point cloud to a PNG image
            render_point_cloud_to_image(
                pcd, 
                output_path, 
                width=render_width, 
                height=render_height,
                view_preset=view_preset,
                multi_view=multi_view
            )
        else:
            # Save the point cloud as PLY
            if not output_path.endswith('.ply'):
                output_path = output_path + '.ply'
            o3d.io.write_point_cloud(output_path, pcd)
            print(f"Point cloud saved to {output_path}")
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        
        if return_pointcloud:
            return None, None
    
    return None if return_pointcloud else output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create a normalized point cloud from an image")
    
    # Input/output arguments
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save the output point cloud or PNG")
    
    # Ground plane parameters directory
    parser.add_argument("--ground_params_dir", type=str, default=None,
                        help="Directory to save/load ground plane parameters (defaults to image directory and uses 'ground.json')")
    
    # Rotation offset arguments
    parser.add_argument("--rot_x", type=float, default=0.0,
                        help="Rotation offset around X axis in degrees")
    parser.add_argument("--rot_y", type=float, default=0.0,
                        help="Rotation offset around Y axis in degrees")
    parser.add_argument("--rot_z", type=float, default=0.0,
                        help="Rotation offset around Z axis in degrees")
    
    # Grid adjustment parameters
    parser.add_argument("--grid_size", type=int, default=20,
                        help="Number of grid cells in each dimension for grid adjustment (default: 20)")
    parser.add_argument("--ground_percentile", type=int, default=5,
                        help="Percentile of lowest points to consider for ground adjustment (default: 5)")
    
    # Performance optimization parameters
    parser.add_argument("--optimized", action="store_true",
                        help="Enable all performance optimizations for Apple Silicon")
    parser.add_argument("--num_threads", type=int, default=0,
                        help="Number of threads for parallel processing (0=auto)")
    parser.add_argument("--half_precision", action="store_true", 
                        help="Use half precision (float16) for faster computation")
    parser.add_argument("--downscale_factor", type=float, default=1.0,
                        help="Downscale input image by this factor for faster processing (e.g. 0.5 = half size)")
    
    # Rendering parameters
    parser.add_argument("--render_png", action="store_true",
                        help="Render a PNG image instead of saving a PLY file")
    parser.add_argument("--render_width", type=int, default=1280,
                        help="Width of the rendered PNG image in pixels (default: 1280)")
    parser.add_argument("--render_height", type=int, default=720,
                        help="Height of the rendered PNG image in pixels (default: 720)")
    parser.add_argument("--view_preset", type=str, default="front", 
                        choices=["front", "top", "side", "isometric"],
                        help="Camera angle preset for rendering (default: front)")
    parser.add_argument("--multi_view", action="store_true",
                        help="Render multiple views of the point cloud in a grid")
    
    args = parser.parse_args()
    
    # Set up multi-threading for numpy
    if args.num_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(args.num_threads)
        os.environ["MKL_NUM_THREADS"] = str(args.num_threads)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(args.num_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(args.num_threads)
    
    # Apply Apple Silicon specific optimizations if on M1/M2/M3
    if args.optimized or is_apple_silicon():
        # Show optimization status
        import platform
        print(f"Applying performance optimizations for Apple {platform.processor()}")
        
        # Set NumPy to use more threads
        if args.num_threads == 0:  # Auto
            # Use number of performance cores (typically 4-6 on M-series)
            # We don't want to use all cores as that might include efficiency cores
            try:
                import multiprocessing
                # Use 3/4 of available cores, which typically uses just the performance cores
                num_threads = max(2, int(multiprocessing.cpu_count() * 0.75))
                os.environ["OMP_NUM_THREADS"] = str(num_threads)
                os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
                os.environ["MKL_NUM_THREADS"] = str(num_threads)
                os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
                os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
                print(f"Using {num_threads} threads for numerical operations")
            except:
                print("Could not determine optimal thread count")
        
        # Set PyTorch to use MPS backend and optimize operations
        if torch.backends.mps.is_available():
            torch.set_num_threads(num_threads if args.num_threads == 0 else args.num_threads)
            torch.set_float32_matmul_precision('high')
            print("PyTorch MPS acceleration enabled")
    
    # Set default output path if none provided
    if args.output_path is None:
        base_name = os.path.splitext(os.path.basename(args.image_path))[0]
        args.output_path = f"{base_name}_normalized.ply"
    
    # Create rotation offset from individual axis rotations
    rotation_offset = [args.rot_x, args.rot_y, args.rot_z]
    
    # Only use rotation if at least one axis has a non-zero rotation
    if rotation_offset == [0.0, 0.0, 0.0]:
        rotation_offset = None
    else:
        print(f"Using rotation offset: {rotation_offset[0]}, {rotation_offset[1]}, {rotation_offset[2]} degrees")
    
    print(f"Grid-based ground adjustment: enabled (grid size: {args.grid_size}, percentile: {args.ground_percentile})")
    
    # Call the main function
    create_normalized_pointcloud(
        args.image_path,
        args.output_path,
        rotation_offset=rotation_offset,
        ground_params_dir=args.ground_params_dir,
        grid_size=args.grid_size,
        ground_percentile=args.ground_percentile,
        downscale_factor=args.downscale_factor,
        half_precision=args.half_precision,
        render_png=args.render_png,
        render_width=args.render_width,
        render_height=args.render_height,
        view_preset=args.view_preset,
        multi_view=args.multi_view
    )
    