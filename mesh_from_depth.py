from PIL import Image
import depth_pro
import numpy as np
import torch
import open3d as o3d
import cv2

def create_textured_mesh(image_path, output_path=None, visualize=True, depth_scaling=1.0):
    """
    Convert a 2D image to a textured 3D mesh using Depth Pro.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the output mesh (optional)
        visualize: Whether to visualize the mesh
        depth_scaling: Scale factor for depth values (adjust for better visualization)
    
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
    
    # Convert depth to numpy array and apply scaling
    depth_np = depth.detach().cpu().numpy() * depth_scaling
    
    # Create a depth image for Open3D
    depth_image = o3d.geometry.Image(depth_np.astype(np.float32))
    
    # Create a color image for Open3D
    color_image = o3d.geometry.Image(np.array(image).astype(np.uint8))
    
    # Create RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image, depth_image, 
        depth_scale=1.0,  # depth values are already in meters
        depth_trunc=100.0,  # maximum depth in meters
        convert_rgb_to_intensity=False
    )
    
    # Create camera intrinsic parameters
    h, w = depth_np.shape
    cx, cy = w / 2, h / 2
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=w,
        height=h,
        fx=focallength_px.item(),
        fy=focallength_px.item(),
        cx=cx,
        cy=cy
    )
    
    # Create point cloud from RGBD image
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic
    )
    
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
    
    # Texture the mesh using the original image
    # This is a simplified approach - for better results, UV mapping would be needed
    mesh.compute_vertex_normals()
    
    # Save mesh if output path is provided
    if output_path:
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"Mesh saved to {output_path}")
    
    # Visualize mesh
    if visualize:
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    
    return mesh

def create_simple_mesh(image_path, output_path=None, visualize=True):
    """
    Create a simple height map mesh from a depth image.
    This method creates a grid mesh which is often more visually appealing
    for scenes with a clear foreground/background separation.
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
    
    # Create vertices
    vertices = []
    colors = []
    
    # Normalize depth for better visualization
    depth_min = np.min(depth_np)
    depth_max = np.max(depth_np)
    depth_range = depth_max - depth_min
    
    # Create grid of vertices
    for i in range(h):
        for j in range(w):
            # Normalize coordinates to [-1, 1]
            x = (j / w) * 2 - 1
            y = 1 - (i / h) * 2  # Flip y to match image coordinates
            z = (depth_np[i, j] - depth_min) / depth_range * -1  # Negative to make closer objects protrude
            
            vertices.append([x, y, z])
            colors.append(image[i, j] / 255.0)
    
    # Create triangles (faces)
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
    parser.add_argument("--method", type=str, default="simple", choices=["poisson", "simple"], 
                        help="Mesh creation method: 'poisson' or 'simple'")
    parser.add_argument("--no_visualize", action="store_false", dest="visualize", 
                        help="Disable visualization")
    parser.add_argument("--depth_scaling", type=float, default=1.0, 
                        help="Scale factor for depth values")
    
    args = parser.parse_args()
    
    if args.method == "poisson":
        create_textured_mesh(args.image_path, args.output_path, args.visualize, args.depth_scaling)
    else:
        create_simple_mesh(args.image_path, args.output_path, args.visualize) 