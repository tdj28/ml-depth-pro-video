from PIL import Image
import depth_pro
import numpy as np
import torch
import open3d as o3d
import cv2
import copy
from scipy import ndimage

def remove_depth_shadows(depth_map, image=None, threshold_factor=0.2, min_region_size=100):
    """
    Remove shadow artifacts from depth maps by detecting abrupt depth changes.
    
    Args:
        depth_map: Numpy array containing the depth map
        image: Optional RGB image for edge detection assistance
        threshold_factor: Factor to determine depth discontinuity (0.1-0.5)
        min_region_size: Minimum size of regions to keep
        
    Returns:
        Processed depth map with shadows removed
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
    
    # Apply the mask to the depth map
    processed_depth[~valid_mask] = 0
    
    # Optional: fill small holes
    processed_depth = ndimage.median_filter(processed_depth, size=3)
    
    return processed_depth

def create_mesh_from_video3d_pointcloud(image_path, output_path=None, visualize=True, remove_shadows=True):
    """
    Create a mesh directly from the exact same point cloud as video3D.py.
    This function first creates the point cloud exactly as in video3D.py,
    then converts it to a mesh with no additional processing.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the output mesh (optional)
        visualize: Whether to visualize the mesh
        remove_shadows: Whether to remove shadow artifacts
    
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
    if remove_shadows:
        depth_np = remove_depth_shadows(depth_np, np.array(image))
    
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
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Remove points with zero depth (shadows that were removed)
    if remove_shadows:
        pcd = pcd.select_by_index(np.where(Z.reshape(-1) > 0)[0])
    
    # Optional: remove points that are too far away
    pcd = pcd.select_by_index(np.where(Z.reshape(-1) < 100)[0])
    
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
    
    # Save mesh if output path is provided
    if output_path:
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"Mesh saved to {output_path}")
    
    # Visualize mesh and point cloud together
    if visualize:
        o3d.visualization.draw_geometries([mesh, pcd_for_viz], mesh_show_back_face=True)
    
    return mesh

def create_textured_mesh(image_path, output_path=None, visualize=True, remove_shadows=True):
    """
    Convert a 2D image to a textured 3D mesh using Depth Pro.
    Uses Poisson surface reconstruction for a more detailed mesh.
    Uses exactly the same dimensions as the point cloud in video3D.py.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the output mesh (optional)
        visualize: Whether to visualize the mesh
        remove_shadows: Whether to remove shadow artifacts
    
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
    if remove_shadows:
        depth_np = remove_depth_shadows(depth_np, np.array(image))
    
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
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Remove points with zero depth (shadows that were removed)
    if remove_shadows:
        pcd = pcd.select_by_index(np.where(Z.reshape(-1) > 0)[0])
    
    # Optional: remove points that are too far away - EXACTLY as in video3D.py
    pcd = pcd.select_by_index(np.where(Z.reshape(-1) < 100)[0])
    
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
    
    # Save mesh if output path is provided
    if output_path:
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"Mesh saved to {output_path}")
    
    # Visualize mesh
    if visualize:
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    
    return mesh

def create_simple_mesh(image_path, output_path=None, visualize=True, remove_shadows=True):
    """
    Create a simple height map mesh from a depth image.
    Uses exactly the same dimensions as the point cloud in video3D.py.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the output mesh (optional)
        visualize: Whether to visualize the mesh
        remove_shadows: Whether to remove shadow artifacts
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
    if remove_shadows:
        depth_np = remove_depth_shadows(depth_np, np.array(image))
    
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
    
    # Create triangles (faces) - this is more complex with shadow removal
    # We need to create a mapping from (i,j) to vertex index
    if remove_shadows:
        # Create a mapping from (i,j) to vertex index
        vertex_map = -np.ones((h, w), dtype=int)
        vertex_idx = 0
        for i in range(h):
            for j in range(w):
                if valid_mask[i, j]:
                    vertex_map[i, j] = vertex_idx
                    vertex_idx += 1
        
        # Create triangles only for valid regions
        triangles = []
        for i in range(h-1):
            for j in range(w-1):
                # Get vertex indices for the four corners
                idx00 = vertex_map[i, j]
                idx01 = vertex_map[i, j+1]
                idx10 = vertex_map[i+1, j]
                idx11 = vertex_map[i+1, j+1]
                
                # Create triangles only if all vertices are valid
                if idx00 >= 0 and idx01 >= 0 and idx10 >= 0:
                    triangles.append([idx00, idx01, idx10])
                if idx01 >= 0 and idx11 >= 0 and idx10 >= 0:
                    triangles.append([idx01, idx11, idx10])
    else:
        # Original triangle creation for non-shadow removal case
        triangles = []
        for i in range(h-1):
            for j in range(w-1):
                # Calculate vertex indices
                idx = i * w + j
                
                # Create two triangles for each grid cell
                triangles.append([idx, idx + 1, idx + w])
                triangles.append([idx + 1, idx + w + 1, idx + w])
    
    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(colors))
    
    # Compute normals
    mesh.compute_vertex_normals()
    
    # Save mesh if output path is provided
    if output_path:
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"Mesh saved to {output_path}")
    
    # Visualize mesh
    if visualize:
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    
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
    parser.add_argument("--shadow_threshold", type=float, default=0.2,
                        help="Threshold for shadow detection (0.1-0.5)")
    
    args = parser.parse_args()
    
    if args.method == "direct":
        create_mesh_from_video3d_pointcloud(args.image_path, args.output_path, args.visualize, args.remove_shadows)
    elif args.method == "poisson":
        create_textured_mesh(args.image_path, args.output_path, args.visualize, args.remove_shadows)
    else:
        create_simple_mesh(args.image_path, args.output_path, args.visualize, args.remove_shadows) 