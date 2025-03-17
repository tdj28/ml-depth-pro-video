import os
import numpy as np
import torch
import open3d as o3d
import cv2
import argparse
import copy
import json
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# Import functionality from the existing script
from img_to_normalized_pointcloud import (
    get_torch_device, 
    depth_to_3d, 
    is_apple_silicon,
    create_normalized_pointcloud,
    render_point_cloud_to_image
)

# Import our depth estimation model
import depth_pro

def get_normalized_pointcloud(image_path, grid_size=20, ground_percentile=5, 
                             downscale_factor=1.0, half_precision=False, 
                             rotation_offset=None, ground_params_dir=None):
    """
    Generate a normalized point cloud from an image without writing to disk.
    
    Args:
        image_path: Path to the input image
        grid_size: Number of grid cells for ground adjustment
        ground_percentile: Percentile of lowest points to consider for ground
        downscale_factor: Downscale image by this factor for processing
        half_precision: Use float16 for faster computation
        rotation_offset: Optional rotation offset [x_rot, y_rot, z_rot] in degrees
        ground_params_dir: Directory to save/load ground plane parameters
        
    Returns:
        pcd: Open3D point cloud object with normalized coordinates
    """
    try:
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
            focallength_px = prediction["focallength_px"]  # Focal length in pixels
        
        # Convert depth to numpy array
        depth_np = depth.detach().cpu().numpy()
        
        # Get image dimensions
        h, w = depth_np.shape
        
        print(f"Depth map dimensions: {w}x{h}")
        print(f"Focal length: {focallength_px.item()} pixels")
        
        # Convert depth to 3D points
        points_3d, valid_mask = depth_to_3d(depth_np, focallength_px.item(), w, h)
        
        print(f"Generated point cloud with {len(points_3d)} valid points")
        
        # Get colors for valid points
        colors = np.array(image).reshape(-1, 3)[valid_mask.flatten()] / 255.0
        
        # Process the point cloud using the existing functionality but return it instead of saving
        result = create_normalized_pointcloud(
            image_path=image_path,
            output_path=None,  # Don't save to disk
            rotation_offset=rotation_offset,
            ground_params_dir=ground_params_dir,
            grid_size=grid_size,
            ground_percentile=ground_percentile,
            downscale_factor=downscale_factor,
            half_precision=half_precision,
            render_png=False,
            return_pointcloud=True  # New parameter to return the point cloud
        )
        
        if result is None:
            # If create_normalized_pointcloud doesn't return the point cloud,
            # create our own Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_3d)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            return pcd, colors
        else:
            return result
        
    except Exception as e:
        print(f"Error generating point cloud: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def remove_stray_points(pcd, nb_points=20, radius=0.1):
    """
    Remove stray points that don't have enough neighbors within the specified radius.
    
    Args:
        pcd: Open3D point cloud
        nb_points: Minimum number of points required in the neighborhood
        radius: Radius for neighborhood search
        
    Returns:
        filtered_pcd: Point cloud with stray points removed
    """
    print(f"Removing stray points (min neighbors: {nb_points}, radius: {radius}m)...")
    
    # Create a copy of the input point cloud
    filtered_pcd = copy.deepcopy(pcd)
    
    # Convert to numpy arrays for faster processing
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # Build KD-tree for efficient neighbor search
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    # Initialize mask for points to keep
    keep_indices = []
    
    # Process in batches for better performance
    batch_size = 10000
    num_batches = (len(points) + batch_size - 1) // batch_size
    
    print(f"Processing {len(points)} points in {num_batches} batches...")
    
    num_kept = 0
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(points))
        
        batch_keep = []
        
        for j in range(start_idx, end_idx):
            # Find neighbors within radius
            [k, idx, _] = kdtree.search_radius_vector_3d(pcd.points[j], radius)
            
            # Keep point if it has enough neighbors
            if k >= nb_points:
                batch_keep.append(j)
        
        keep_indices.extend(batch_keep)
        num_kept += len(batch_keep)
        
        # Print progress
        if (i+1) % 10 == 0 or i == num_batches - 1:
            print(f"Processed batch {i+1}/{num_batches}, kept {num_kept}/{end_idx} points so far")
    
    # Create a new point cloud with only the kept points
    filtered_points = points[keep_indices]
    filtered_colors = colors[keep_indices]
    
    # Update the filtered point cloud
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    
    print(f"Stray point removal: kept {len(filtered_points)}/{len(points)} points ({len(filtered_points)/len(points)*100:.1f}%)")
    
    return filtered_pcd

def clean_shadows(pcd, shadow_height_threshold=0.1, max_shadow_angle=75, min_points_per_column=3):
    """
    Clean up shadows resulting from monocular depth perception.
    Identifies and removes shadow artifacts that appear as thin vertical structures.
    
    Args:
        pcd: Open3D point cloud
        shadow_height_threshold: Minimum height for potential shadow structures
        max_shadow_angle: Maximum angle with the vertical for shadow structures (degrees)
        min_points_per_column: Minimum points required to identify a shadow column
        
    Returns:
        cleaned_pcd: Point cloud with shadow artifacts removed
    """
    print("Cleaning up shadow artifacts...")
    
    # Create a copy of the input point cloud
    cleaned_pcd = copy.deepcopy(pcd)
    
    # Convert to numpy arrays for processing
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # Calculate XZ grid for columnar analysis
    # We'll divide the space into a grid on the XZ plane and analyze points in each cell
    
    # Determine grid size based on point cloud density
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    
    # Calculate an appropriate cell size (adapt based on point cloud density)
    point_density = len(points) / ((x_max - x_min) * (z_max - z_min))
    cell_size = max(0.05, 1.0 / np.sqrt(point_density / 10))  # Adjust threshold based on density
    
    print(f"Using cell size of {cell_size:.3f}m for shadow detection")
    
    # Create grid cells
    x_bins = np.arange(x_min, x_max + cell_size, cell_size)
    z_bins = np.arange(z_min, z_max + cell_size, cell_size)
    
    # Assign points to cells
    x_indices = np.digitize(points[:, 0], x_bins) - 1
    z_indices = np.digitize(points[:, 2], z_bins) - 1
    
    # Combined cell index for each point
    cell_indices = x_indices * len(z_bins) + z_indices
    
    # Find unique cells and count points per cell
    unique_cells, counts = np.unique(cell_indices, return_counts=True)
    
    # Initialize mask for points to keep
    shadow_mask = np.ones(len(points), dtype=bool)
    
    # Process cells with enough points
    shadow_point_count = 0
    
    for cell_idx in unique_cells:
        # Skip cells with too few points
        cell_mask = (cell_indices == cell_idx)
        cell_points = points[cell_mask]
        
        if len(cell_points) < min_points_per_column:
            continue
        
        # Analyze height distribution in this cell
        y_values = cell_points[:, 1]
        y_min, y_max = np.min(y_values), np.max(y_values)
        height = y_max - y_min
        
        # Check if this cell has a tall, thin structure (potential shadow)
        if height > shadow_height_threshold:
            # Sort points by height (Y coordinate)
            sorted_indices = np.argsort(y_values)
            sorted_cell_points = cell_points[sorted_indices]
            
            # Analyze the vertical distribution
            # Calculate angles between consecutive points
            if len(sorted_cell_points) >= 3:
                vectors = np.diff(sorted_cell_points, axis=0)
                # Calculate angles with vertical (Y axis)
                vertical_axis = np.array([0, 1, 0])
                angles = np.arccos(np.clip(vectors[:, 1] / np.linalg.norm(vectors, axis=1), -1.0, 1.0)) * 180 / np.pi
                
                # If most points form a near-vertical line, consider it a shadow
                if np.median(angles) < max_shadow_angle:
                    # Find indices of points in this cell that belong to the shadow
                    cell_indices_in_full_array = np.where(cell_mask)[0]
                    shadow_mask[cell_indices_in_full_array] = False
                    shadow_point_count += len(cell_indices_in_full_array)
    
    # Keep points that are not shadows
    clean_points = points[shadow_mask]
    clean_colors = colors[shadow_mask]
    
    # Update the cleaned point cloud
    cleaned_pcd.points = o3d.utility.Vector3dVector(clean_points)
    cleaned_pcd.colors = o3d.utility.Vector3dVector(clean_colors)
    
    print(f"Shadow cleanup: removed {shadow_point_count} points ({shadow_point_count/len(points)*100:.1f}% of total)")
    
    return cleaned_pcd

def create_mesh_from_pointcloud(pcd, voxel_size=0.05, depth=8, method="poisson"):
    """
    Create a mesh from a point cloud using the specified surface reconstruction method.
    
    Args:
        pcd: Open3D point cloud
        voxel_size: Voxel size for downsampling (smaller = more detail but slower)
        depth: Maximum depth of the octree for Poisson reconstruction
        method: Reconstruction method ("poisson", "ball_pivoting", or "simple")
        
    Returns:
        mesh: Reconstructed mesh
    """
    print(f"Creating mesh from point cloud using {method} method...")
    
    # Create a copy of the input point cloud
    pcd_for_mesh = copy.deepcopy(pcd)
    
    # Estimate normals if they don't exist
    if not pcd_for_mesh.has_normals() or method == "ball_pivoting":
        print("Estimating normals (this may take a while)...")
        # For each point, use 30 nearest neighbors to estimate normal
        pcd_for_mesh.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30)
        )
        
        # Orient normals consistently
        pcd_for_mesh.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 0]))
        print("Normal estimation complete")
    
    # Downsample the point cloud for faster processing
    print(f"Downsampling with voxel size {voxel_size}m...")
    downsampled_pcd = pcd_for_mesh.voxel_down_sample(voxel_size)
    
    # Re-estimate normals on the downsampled point cloud
    print("Re-estimating normals on downsampled point cloud...")
    downsampled_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30)
    )
    downsampled_pcd.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 0]))
    
    mesh = o3d.geometry.TriangleMesh()
    
    if method == "poisson":
        # Apply Poisson surface reconstruction
        print(f"Applying Poisson surface reconstruction with depth {depth}...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            downsampled_pcd, depth=depth
        )
        
        # Remove low-density vertices
        print("Filtering mesh to remove low-density areas...")
        vertices_to_remove = densities < np.quantile(densities, 0.1)  # Remove bottom 10% of densities
        mesh.remove_vertices_by_mask(vertices_to_remove)
    
    elif method == "ball_pivoting":
        # Apply Ball Pivoting algorithm for mesh reconstruction
        # Determine radius based on point cloud density - adjust multiplier as needed
        avg_dist = get_average_point_distance(downsampled_pcd)
        radii = [avg_dist*2, avg_dist*4, avg_dist*8, avg_dist*16]
        print(f"Applying Ball Pivoting Algorithm with radii: {radii}...")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            downsampled_pcd, o3d.utility.DoubleVector(radii)
        )
    
    elif method == "simple":
        # Simple direct triangulation method
        print("Creating simple triangulated mesh...")
        mesh = create_simple_triangulated_mesh(downsampled_pcd)
    
    else:
        raise ValueError(f"Unknown mesh creation method: {method}. Use 'poisson', 'ball_pivoting', or 'simple'.")
    
    # Perform mesh cleanup
    print("Performing mesh cleanup...")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    
    print(f"Mesh created with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    
    return mesh

def get_average_point_distance(pcd, k=20):
    """
    Calculate the average distance between a point and its k nearest neighbors.
    This helps determine appropriate radii for ball pivoting.
    
    Args:
        pcd: Open3D point cloud
        k: Number of nearest neighbors to consider
        
    Returns:
        avg_dist: Average distance to nearest neighbors
    """
    points = np.asarray(pcd.points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    total_dist = 0
    sample_size = min(1000, len(points))  # Use a sample for efficiency
    sample_indices = np.random.choice(len(points), sample_size, replace=False)
    
    for idx in sample_indices:
        _, indices, distances = kdtree.search_knn_vector_3d(points[idx], k+1)  # +1 because the point itself is included
        avg_neighbor_dist = np.mean(np.sqrt(distances[1:]))  # Exclude the point itself (distance 0)
        total_dist += avg_neighbor_dist
    
    return total_dist / sample_size

def create_simple_triangulated_mesh(pcd):
    """
    Create a simple triangulated mesh from a point cloud.
    This method creates a triangle mesh by directly connecting neighboring points.
    
    Args:
        pcd: Open3D point cloud
        
    Returns:
        mesh: Simple triangulated mesh
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # Create a KD-tree for nearest neighbor search
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    # Create an empty triangle mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    # Create triangles by connecting each point with its two closest neighbors
    triangles = []
    
    for i in range(len(points)):
        # Find 6 nearest neighbors
        _, indices, _ = kdtree.search_knn_vector_3d(points[i], 7)  # +1 because point itself is included
        neighbors = indices[1:]  # Exclude the point itself
        
        # Create triangles with neighboring points
        for j in range(len(neighbors)-1):
            # Don't create triangles with points that are too far apart
            if j < len(neighbors)-1:
                triangles.append([i, neighbors[j], neighbors[j+1]])
    
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    
    # Remove bad triangles
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    
    return mesh

def process_image_to_mesh(image_path, output_path=None, voxel_size=0.05, depth=8, 
                         remove_strays=True, clean_shadows_flag=True,
                         downscale_factor=1.0, grid_size=20, ground_percentile=5,
                         mesh_method="poisson"):
    """
    Process an image to create a clean mesh:
    1. Generate a normalized point cloud
    2. Remove stray points
    3. Clean up shadow artifacts
    4. Create a mesh using the specified reconstruction method
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the output mesh (default: derived from image_path)
        voxel_size: Voxel size for mesh creation (smaller = more detail but slower)
        depth: Maximum depth of the octree for Poisson reconstruction
        remove_strays: Whether to remove stray points
        clean_shadows_flag: Whether to clean shadow artifacts
        downscale_factor: Downscale image by this factor for faster processing
        grid_size: Number of grid cells for ground adjustment
        ground_percentile: Percentile of lowest points to consider for ground
        mesh_method: Method to use for mesh creation ("poisson", "ball_pivoting", or "simple")
        
    Returns:
        mesh: The final mesh
    """
    # Default output path if not specified
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"{base_name}_mesh.obj"
    
    # Generate normalized point cloud
    print(f"Generating normalized point cloud from {image_path}...")
    pcd, _ = get_normalized_pointcloud(
        image_path=image_path,
        grid_size=grid_size,
        ground_percentile=ground_percentile,
        downscale_factor=downscale_factor
    )
    
    if pcd is None:
        print("Failed to generate point cloud. Exiting.")
        return None
    
    # Remove stray points
    if remove_strays:
        pcd = remove_stray_points(pcd, nb_points=20, radius=0.1)
    
    # Clean up shadow artifacts
    if clean_shadows_flag:
        pcd = clean_shadows(pcd)
    
    # Create mesh
    mesh = create_mesh_from_pointcloud(pcd, voxel_size=voxel_size, depth=depth, method=mesh_method)
    
    # Save the mesh
    print(f"Saving mesh to {output_path}...")
    o3d.io.write_triangle_mesh(output_path, mesh)
    
    # Also save a preview image of the mesh
    preview_path = f"{os.path.splitext(output_path)[0]}_preview.png"
    visualize_mesh(mesh, preview_path)
    
    return mesh

def visualize_mesh(mesh, output_path, width=1280, height=720):
    """
    Visualize a mesh and save the visualization as an image.
    
    Args:
        mesh: Open3D mesh
        output_path: Path to save the visualization
        width: Width of the output image
        height: Height of the output image
        
    Returns:
        None
    """
    # Create a visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    
    # Add the mesh to the visualization
    vis.add_geometry(mesh)
    
    # Improve the visualization
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])  # White background
    opt.point_size = 1.0
    opt.light_on = True
    opt.mesh_show_wireframe = True
    opt.mesh_shade_option = o3d.visualization.MeshShadeOption.Color
    
    # Set the camera viewpoint
    ctr = vis.get_view_control()
    
    # Set to isometric view
    front = np.array([1, -1, -1])  # 45 degree angle
    front = front / np.linalg.norm(front)
    up = np.array([0, 1, 0])  # Y-up
    
    # Get mesh bounds
    mesh_bounds = mesh.get_axis_aligned_bounding_box()
    center = mesh_bounds.get_center()
    
    ctr.set_front(front)
    ctr.set_lookat(center)
    ctr.set_up(up)
    ctr.set_zoom(0.8)
    
    # Update and render
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    
    # Capture and save the image
    image = vis.capture_screen_float_buffer(do_render=True)
    image_array = np.asarray(image)
    # Clip values to valid 0-1 range to avoid ValueError in plt.imsave
    image_array_clipped = np.clip(image_array, 0, 1)
    plt.imsave(output_path, image_array_clipped)
    
    # Close the visualizer
    vis.destroy_window()
    print(f"Mesh visualization saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a clean mesh from an image")
    
    # Input/output arguments
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save the output mesh (default: based on input filename)")
    
    # Performance parameters
    parser.add_argument("--downscale_factor", type=float, default=1.0,
                        help="Downscale input image for faster processing (default: 1.0)")
    
    # Mesh creation parameters
    parser.add_argument("--voxel_size", type=float, default=0.05,
                        help="Voxel size for point cloud downsampling (default: 0.05)")
    parser.add_argument("--depth", type=int, default=8,
                        help="Depth for Poisson surface reconstruction (default: 8)")
    parser.add_argument("--mesh_method", type=str, choices=["poisson", "ball_pivoting", "simple"], default="poisson",
                        help="Method to use for mesh creation (default: poisson)")
    
    # Processing flags
    parser.add_argument("--remove_strays", action="store_true", default=True,
                        help="Remove stray points from the point cloud")
    parser.add_argument("--clean_shadows", action="store_true", default=True,
                        help="Clean shadow artifacts in the depth map")
    
    args = parser.parse_args()
    
    # Process the image and create a mesh
    process_image_to_mesh(
        image_path=args.image_path,
        output_path=args.output_path,
        voxel_size=args.voxel_size,
        depth=args.depth,
        remove_strays=args.remove_strays,
        clean_shadows_flag=args.clean_shadows,
        downscale_factor=args.downscale_factor,
        mesh_method=args.mesh_method
    ) 