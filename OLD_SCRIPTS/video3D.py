from PIL import Image
import depth_pro
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

def image_to_point_cloud(image_path, output_path=None, visualize=True):
    """Convert a 2D image to a 3D point cloud using Depth Pro."""
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
    
    # Create meshgrid of pixel coordinates
    h, w = depth_np.shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Calculate principal point (assuming center of image)
    cx, cy = w / 2, h / 2
    
    # Convert pixel coordinates to camera coordinates
    # Z is the depth
    Z = depth_np
    # X = (x - cx) * Z / focallength_px
    X = (x - cx) * Z / focallength_px.item()
    # Y = (y - cy) * Z / focallength_px
    Y = (y - cy) * Z / focallength_px.item()
    
    # Stack coordinates and reshape to Nx3 array
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    
    # Get colors from original image
    colors = image.reshape(-1, 3) / 255.0
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Optional: remove points that are too far away
    pcd = pcd.select_by_index(np.where(Z.reshape(-1) < 100)[0])
    
    # Save point cloud if output path is provided
    if output_path:
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Point cloud saved to {output_path}")
    
    # Visualize point cloud
    if visualize:
        o3d.visualization.draw_geometries([pcd])
    
    return pcd

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert 2D image to 3D point cloud")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_path", type=str, help="Path to save point cloud (optional)")
    parser.add_argument("--no_visualize", action="store_false", dest="visualize", help="Disable visualization")
    
    args = parser.parse_args()
    
    image_to_point_cloud(args.image_path, args.output_path, args.visualize)
