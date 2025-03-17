import os
import numpy as np
import open3d as o3d
import cv2
import argparse
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from shapely.ops import polygonize, unary_union
from shapely import affinity
import skimage.measure
import skimage.morphology
import skimage.segmentation
import skimage.feature
import math
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import DBSCAN

def load_pointcloud(pointcloud_path):
    """
    Load a point cloud from file
    
    Args:
        pointcloud_path: Path to the point cloud file (.ply)
        
    Returns:
        pcd: Open3D point cloud
    """
    print(f"Loading point cloud from {pointcloud_path}...")
    try:
        pcd = o3d.io.read_point_cloud(pointcloud_path)
        num_points = len(np.asarray(pcd.points))
        print(f"Loaded point cloud with {num_points} points")
        return pcd
    except Exception as e:
        print(f"Error loading point cloud: {str(e)}")
        return None

def detect_floor_plane(pcd, distance_threshold=0.02, ransac_n=3, num_iterations=1000):
    """
    Detect the floor plane using RANSAC
    
    Args:
        pcd: Open3D point cloud
        distance_threshold: Maximum distance a point can be from the plane to be considered an inlier
        ransac_n: Number of points randomly sampled for each RANSAC iteration
        num_iterations: Number of RANSAC iterations
        
    Returns:
        floor_model: [a, b, c, d] parameters of the floor plane equation ax + by + cz + d = 0
        floor_inliers: Indices of points that belong to the floor
    """
    print("Detecting floor plane...")
    
    # Copy point cloud to prevent modifying the original
    pcd_copy = o3d.geometry.PointCloud(pcd)
    
    # Run RANSAC to find the plane
    plane_model, inliers = pcd_copy.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    
    # Extract the plane parameters [a, b, c, d]
    # where ax + by + cz + d = 0 is the plane equation
    a, b, c, d = plane_model
    
    # Make sure the normal vector points upward (assuming Y is up)
    if b < 0:
        a, b, c, d = -a, -b, -c, -d
    
    print(f"Detected floor plane with equation: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
    print(f"Found {len(inliers)} floor points out of {len(pcd_copy.points)} total points")
    
    # Check if this seems like a reasonable floor (should be mostly horizontal)
    normal_vector = np.array([a, b, c])
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    
    # Calculate angle with vertical axis
    vertical_axis = np.array([0, 1, 0])  # Y-axis
    angle_with_vertical = np.arccos(np.abs(np.dot(normal_vector, vertical_axis))) * 180 / np.pi
    
    if angle_with_vertical > 20:  # Allow some tilt, but not too much
        print(f"Warning: Detected plane doesn't seem to be horizontal (angle with vertical: {angle_with_vertical:.1f} degrees)")
        # We'll still use it, but it's worth a warning
    
    return plane_model, inliers

def normalize_points_to_floor(points, floor_model):
    """
    Normalize point cloud so the floor is at y=0
    
    Args:
        points: Nx3 numpy array of points
        floor_model: [a, b, c, d] parameters of the floor plane equation
        
    Returns:
        normalized_points: Nx3 numpy array of normalized points
    """
    a, b, c, d = floor_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    
    # Calculate signed distance from each point to the plane
    # Positive is above the plane, negative is below
    points_array = np.asarray(points)
    distances = (np.sum(points_array * normal, axis=1) + d) / np.linalg.norm(normal)
    
    # Create new points where floor points have y=0 and everything else is relative to the floor
    normalized_points = points_array.copy()
    
    # For each point, shift it along the normal vector by its distance to the plane
    normalized_points = normalized_points - np.outer(distances, normal)
    
    return normalized_points

def create_height_slices(points, colors, height_min=0.1, height_max=2.5, num_slices=5, height_threshold=None):
    """
    Create height slices of the point cloud
    
    Args:
        points: Nx3 numpy array of points
        colors: Nx3 numpy array of colors
        height_min: Minimum height to consider
        height_max: Maximum height to consider
        num_slices: Number of slices to create
        height_threshold: If not None, only consider points above this height
        
    Returns:
        slices: List of (points, colors, height) tuples for each slice
    """
    # If height_threshold is provided, create a single slice with everything above that height
    if height_threshold is not None:
        print(f"Creating a single slice with points above {height_threshold}m...")
        
        # Find points above the threshold height
        height_mask = (points[:, 1] >= height_threshold)
        slice_points = points[height_mask]
        slice_colors = colors[height_mask] if colors is not None else None
        
        # Use the average height or the threshold as the label
        avg_height = np.mean(slice_points[:, 1]) if len(slice_points) > 0 else height_threshold
        
        slices = [(slice_points, slice_colors, avg_height)]
        print(f"  Single slice: {len(slice_points)} points above {height_threshold:.2f}m")
        return slices
    
    # Original slicing logic
    print(f"Creating {num_slices} height slices between {height_min}m and {height_max}m...")
    
    slice_height = (height_max - height_min) / num_slices
    slices = []
    
    for i in range(num_slices):
        slice_start = height_min + i * slice_height
        slice_end = slice_start + slice_height
        
        # Find points in this height range
        height_mask = (points[:, 1] >= slice_start) & (points[:, 1] < slice_end)
        slice_points = points[height_mask]
        slice_colors = colors[height_mask] if colors is not None else None
        
        # Store the slice with its average height
        avg_height = (slice_start + slice_end) / 2
        slices.append((slice_points, slice_colors, avg_height))
        
        print(f"  Slice {i+1}: {len(slice_points)} points between {slice_start:.2f}m and {slice_end:.2f}m")
    
    return slices

def create_density_grid_from_points(points_2d, grid_resolution=0.05, padding=1.0, weights=None):
    """
    Create a density grid from 2D points
    
    Args:
        points_2d: Nx2 numpy array of 2D points
        grid_resolution: Grid cell size in meters
        padding: Padding around the grid in meters
        weights: Optional weights for each point
        
    Returns:
        grid: 2D numpy array with point density
        (min_x, min_z): Origin coordinates of the grid
        grid_resolution: Resolution of the grid
    """
    if len(points_2d) == 0:
        return None, None, grid_resolution
    
    # Determine grid dimensions
    min_x, max_x = np.min(points_2d[:, 0]) - padding, np.max(points_2d[:, 0]) + padding
    min_z, max_z = np.min(points_2d[:, 1]) - padding, np.max(points_2d[:, 1]) + padding
    
    # Calculate grid size
    grid_width = int(np.ceil((max_x - min_x) / grid_resolution))
    grid_height = int(np.ceil((max_z - min_z) / grid_resolution))
    
    # Create density grid
    grid = np.zeros((grid_height, grid_width), dtype=np.float32)
    
    # Map points to grid cells
    for i, point in enumerate(points_2d):
        x, z = point
        grid_x = int((x - min_x) / grid_resolution)
        grid_z = int((z - min_z) / grid_resolution)
        
        # Ensure within bounds
        if 0 <= grid_x < grid_width and 0 <= grid_z < grid_height:
            weight = weights[i] if weights is not None else 1.0
            grid[grid_z, grid_x] += weight
    
    return grid, (min_x, min_z), grid_resolution

def get_optimal_closing_kernel_size(grid):
    """
    Estimate an optimal kernel size for morphological closing based on the grid
    
    Args:
        grid: 2D binary grid
        
    Returns:
        kernel_size: Optimal kernel size
    """
    # Calculate average size of non-zero regions
    labeled_grid, num_features = skimage.measure.label(grid, return_num=True)
    if num_features == 0:
        return 3  # Default size
    
    region_sizes = []
    for i in range(1, num_features + 1):
        region_size = np.sum(labeled_grid == i)
        region_sizes.append(region_size)
    
    avg_region_size = np.mean(region_sizes)
    
    # Calculate kernel size as a function of region size
    kernel_size = max(3, min(11, int(np.sqrt(avg_region_size) / 5)))
    
    # Make sure it's odd
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    return kernel_size

def process_height_slice(points_slice, colors_slice, avg_height, grid_resolution=0.05, min_points=10, padding=0.5, height_threshold=None):
    """
    Process a height slice to create a 2D grid
    
    Args:
        points_slice: Nx3 numpy array of points in this height slice
        colors_slice: Nx3 numpy array of colors in this height slice
        avg_height: Average height of this slice
        grid_resolution: Grid cell size in meters
        min_points: Minimum number of points for a valid slice
        padding: Padding around the grid in meters
        height_threshold: If not None, we're in height threshold mode
        
    Returns:
        grid: Processed binary grid
        contours: List of detected contours
        grid_origin: (min_x, min_z) origin coordinates of the grid
    """
    if len(points_slice) < min_points:
        return None, [], None
    
    # Project to XZ plane (assuming Y is up)
    points_2d = points_slice[:, [0, 2]]  # Extract X and Z coordinates
    
    # Create density grid for this slice
    density_grid, grid_origin, _ = create_density_grid_from_points(
        points_2d,
        grid_resolution=grid_resolution,
        padding=padding  # Use the provided padding
    )
    
    if density_grid is None:
        return None, [], None
    
    # Create binary grid
    binary_grid = (density_grid > 0).astype(np.uint8)
    
    # Print some stats about the binary grid
    print(f"  Binary grid shape: {binary_grid.shape}, non-zero pixels: {np.sum(binary_grid)}")
    
    # In height threshold mode, use more aggressive morphological operations
    if height_threshold is not None:
        # Use a relatively large fixed kernel for closing in height threshold mode
        # This will connect nearby points better
        closing_size = 7
        kernel_closing = np.ones((closing_size, closing_size), np.uint8)
        
        # Apply multiple closing operations to better connect points
        processed_grid = binary_grid.copy()
        for _ in range(2):  # Apply closing twice
            processed_grid = cv2.morphologyEx(processed_grid, cv2.MORPH_CLOSE, kernel_closing)
    else:
        # Get optimal kernel size for normal mode
        closing_size = get_optimal_closing_kernel_size(binary_grid)
        kernel_closing = np.ones((closing_size, closing_size), np.uint8)
        
        # Closing to fill small gaps
        processed_grid = cv2.morphologyEx(binary_grid, cv2.MORPH_CLOSE, kernel_closing)
    
    # For both modes: Apply a small opening operation to remove noise
    kernel_opening = np.ones((3, 3), np.uint8)
    processed_grid = cv2.morphologyEx(processed_grid, cv2.MORPH_OPEN, kernel_opening)
    
    # Find contours
    contours, _ = cv2.findContours(processed_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"  Found {len(contours)} contours")
    
    return processed_grid, contours, grid_origin

def approximate_contour(contour, alpha=0.01, min_area=0.05, grid_resolution=0.05, height_threshold=None):
    """
    Approximate a contour with a simpler polygon
    
    Args:
        contour: OpenCV contour
        alpha: Douglas-Peucker epsilon as a fraction of contour perimeter
        min_area: Minimum area in square meters for a valid shape
        grid_resolution: Grid cell size in meters
        height_threshold: If not None, we're in height threshold mode
        
    Returns:
        simplified_polygon: Shapely polygon
    """
    # Calculate contour area in grid cells
    area = cv2.contourArea(contour)
    
    # Convert to square meters
    area_m2 = area * (grid_resolution ** 2)
    
    # In height threshold mode, be more permissive with small areas
    if height_threshold is not None:
        min_area = min_area / 4
    
    # Skip very small contours
    if area_m2 < min_area:
        return None
    
    # Find perimeter
    perimeter = cv2.arcLength(contour, True)
    
    # In height threshold mode, use less aggressive simplification
    if height_threshold is not None:
        alpha = alpha / 2
    
    # Adaptive epsilon - larger epsilon for larger contours
    epsilon = alpha * perimeter
    
    # Approximate contour with polygon (Douglas-Peucker algorithm)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Convert to shapely polygon
    polygon_points = []
    for point in approx:
        x, y = point[0]
        polygon_points.append((x, y))
    
    # Check if we have at least 3 points for a valid polygon
    if len(polygon_points) < 3:
        return None
    
    # Create shapely polygon
    shapely_polygon = Polygon(polygon_points)
    
    # Skip invalid polygons
    if not shapely_polygon.is_valid or shapely_polygon.area <= 0:
        return None
    
    # Try to simplify to rectangle if it's already close to one
    if 4 <= len(polygon_points) <= 6:
        # Calculate convex hull
        hull = ConvexHull(np.array(polygon_points))
        hull_points = np.array(polygon_points)[hull.vertices]
        
        # Find minimum-area bounding rectangle
        rect = cv2.minAreaRect(hull_points.astype(np.float32))
        box = cv2.boxPoints(rect)
        rect_poly = Polygon(box)
        
        # If the rectangle is close to the original shape, use it
        if abs(rect_poly.area - shapely_polygon.area) / shapely_polygon.area < 0.2:
            return rect_poly
    
    return shapely_polygon

def contour_to_polygon(contour, grid_origin, grid_resolution, min_area=0.05, simplify=True, height_threshold=None):
    """
    Convert an OpenCV contour to a Shapely polygon with world coordinates
    
    Args:
        contour: OpenCV contour
        grid_origin: (min_x, min_z) origin coordinates of the grid
        grid_resolution: Grid cell size in meters
        min_area: Minimum area in square meters for a valid shape
        simplify: Whether to try to simplify the polygon
        height_threshold: If not None, we're in height threshold mode
        
    Returns:
        shapely_polygon: Shapely polygon in world coordinates
    """
    if simplify:
        polygon = approximate_contour(contour, min_area=min_area, grid_resolution=grid_resolution, height_threshold=height_threshold)
        if polygon is None:
            return None
    else:
        # Just convert directly to shapely polygon
        points = []
        for point in contour:
            x, y = point[0]
            points.append((x, y))
        
        polygon = Polygon(points)
        
        # In height threshold mode, be more permissive
        effective_min_area = min_area / 4 if height_threshold is not None else min_area
        
        if not polygon.is_valid or polygon.area * (grid_resolution ** 2) < effective_min_area:
            return None
    
    # Convert to world coordinates
    min_x, min_z = grid_origin
    world_polygon = affinity.translate(
        affinity.scale(polygon, xfact=grid_resolution, yfact=grid_resolution),
        xoff=min_x, yoff=min_z
    )
    
    return world_polygon

def create_colored_height_map(slices, grid_resolution=0.05, min_area=0.05, height_threshold=None):
    """
    Create a colored height map from point cloud slices
    
    Args:
        slices: List of (points, colors, height) tuples for each slice
        grid_resolution: Grid cell size in meters
        min_area: Minimum area in square meters for a valid shape
        height_threshold: If not None, we're in height threshold mode
        
    Returns:
        polygons: List of (polygon, height, color) tuples
        combined_grid: Combined binary grid from all slices
        combined_origin: Origin coordinates of the combined grid
    """
    print("Creating colored height map...")
    
    all_polygons = []
    combined_grid = None
    combined_origin = None
    
    # Define a colormap for height visualization
    cmap = plt.cm.viridis  # Can use other colormaps: inferno, plasma, magma, etc.
    
    # Find total height range for normalization
    heights = [h for _, _, h in slices]
    min_height, max_height = min(heights), max(heights)
    height_range = max_height - min_height
    
    # In height threshold mode, use less padding
    padding = 0.2 if height_threshold is not None else 0.5
    
    # Process each slice
    for i, (points, colors, height) in enumerate(slices):
        if len(points) < 10:  # Skip slices with too few points
            continue
            
        # Process this height slice
        grid, contours, origin = process_height_slice(
            points, colors, height,
            grid_resolution=grid_resolution,
            padding=padding,
            height_threshold=height_threshold
        )
        
        if grid is None or len(contours) == 0:
            continue
        
        # Keep track of the combined grid (for visualization)
        if combined_grid is None:
            combined_grid = grid.copy()
            combined_origin = origin
        elif combined_grid.shape == grid.shape:
            # Combine grids using logical OR
            combined_grid = np.logical_or(combined_grid, grid).astype(np.uint8)
        
        # Normalize height for coloring
        norm_height = (height - min_height) / height_range if height_range > 0 else 0.5
        
        # Get color from colormap
        color = cmap(norm_height)[:3]  # RGB components
        
        # Process contours for this slice
        for contour in contours:
            polygon = contour_to_polygon(contour, origin, grid_resolution, min_area=min_area, height_threshold=height_threshold)
            if polygon is not None:
                all_polygons.append((polygon, height, color))
    
    return all_polygons, combined_grid, combined_origin

def plot_floorplan(polygons, output_path, background_grid=None, grid_origin=None, grid_resolution=None,
                  floor_bounds=None, show_grid=True, color_by_height=True, dpi=300, height_threshold=None):
    """
    Plot floor plan with colored height regions
    
    Args:
        polygons: List of (polygon, height, color) tuples
        output_path: Path to save the output image
        background_grid: Optional binary grid for background visualization
        grid_origin: Origin coordinates of the grid
        grid_resolution: Grid cell size in meters
        floor_bounds: Bounding box of the floor [minx, miny, maxx, maxy]
        show_grid: Whether to show grid lines
        color_by_height: Whether to color regions by height
        dpi: DPI for output image
        height_threshold: If not None, the height threshold used
    """
    print(f"Generating floor plan visualization at {output_path}...")
    
    # Create figure
    plt.figure(figsize=(16, 12))
    
    # Plot background grid if provided
    if background_grid is not None and grid_origin is not None and grid_resolution is not None:
        # Create a custom colormap from white to light gray
        cmap = LinearSegmentedColormap.from_list('bg_cmap', ['#ffffff', '#f5f5f5'])
        
        plt.imshow(background_grid, origin='lower', 
                  extent=[grid_origin[0], grid_origin[0] + background_grid.shape[1] * grid_resolution,
                         grid_origin[1], grid_origin[1] + background_grid.shape[0] * grid_resolution],
                  cmap=cmap, alpha=0.3)
    
    # Plot floor bounds if provided
    if floor_bounds is not None:
        minx, miny, maxx, maxy = floor_bounds
        floor_poly = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
        x, y = floor_poly.exterior.xy
        plt.fill(x, y, alpha=0.1, fc='#e6e6e6', ec='#666666', linewidth=1.0, linestyle='--')
    
    # Group polygons by height and sort by height (lowest first)
    sorted_polygons = sorted(polygons, key=lambda x: x[1])
    
    # Plot polygons
    for i, (polygon, height, color) in enumerate(sorted_polygons):
        x, y = polygon.exterior.xy
        
        if color_by_height:
            fc = color
            alpha = 0.7
            ec = 'black'
            linewidth = 0.5
        else:
            # Use a single color scheme if not coloring by height
            fc = '#a8d5ff'  # Light blue
            alpha = 0.7
            ec = 'black'
            linewidth = 0.5
        
        plt.fill(x, y, alpha=alpha, fc=fc, ec=ec, linewidth=linewidth)
    
    # Set axis properties
    plt.axis('equal')
    
    # Set axis limits based on floor bounds if provided
    if floor_bounds is not None:
        minx, miny, maxx, maxy = floor_bounds
        plt.xlim(minx, maxx)
        plt.ylim(miny, maxy)
    
    if show_grid:
        plt.grid(True, linestyle='--', alpha=0.4)
    
    # Set title based on whether we're using a height threshold
    if height_threshold is not None:
        plt.title(f'2D Floor Plan - Objects Above {height_threshold:.2f}m')
    else:
        plt.title('2D Floor Plan with Height Map')
    
    plt.xlabel('X (meters)')
    plt.ylabel('Z (meters)')
    
    # Add a colorbar to show height scale if coloring by height
    if color_by_height and polygons:
        # Find the height range
        heights = [h for _, h, _ in polygons]
        min_height, max_height = min(heights), max(heights)
        
        # Create a colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(min_height, max_height))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label('Height (meters)')
    
    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Floor plan visualization saved to {output_path}")

def save_floorplan_data(polygons, output_path):
    """
    Save floor plan data to a text file
    
    Args:
        polygons: List of (polygon, height, color) tuples
        output_path: Path to save the output data
    """
    print(f"Saving floor plan data to {output_path}...")
    
    with open(output_path, 'w') as f:
        f.write("# Floor Plan Data\n")
        f.write("# Units: meters\n\n")
        
        f.write("# Shapes by height\n")
        f.write("# Format: height, num_points, x1, z1, x2, z2, ...\n")
        
        for i, (polygon, height, _) in enumerate(sorted(polygons, key=lambda x: x[1])):
            coords = list(polygon.exterior.coords)
            f.write(f"{height:.3f}, {len(coords)-1}")  # -1 because last point is the same as first
            
            for coord in coords[:-1]:  # Exclude the repeated last point
                f.write(f", {coord[0]:.3f}, {coord[1]:.3f}")
            
            f.write("\n")
    
    print(f"Floor plan data saved to {output_path}")

def create_simple_point_visualization(points, output_path, height_threshold=None, dpi=300):
    """
    Create a very simple visualization of all points directly projected to the XZ plane
    
    Args:
        points: Nx3 numpy array of points
        output_path: Path to save the visualization
        height_threshold: If not None, only show points above this threshold
        dpi: DPI for output image
    """
    print("Creating simple point visualization...")
    
    # Filter points by height if threshold is provided
    if height_threshold is not None:
        height_mask = points[:, 1] >= height_threshold
        filtered_points = points[height_mask]
        title = f"Points Above {height_threshold:.2f}m"
    else:
        filtered_points = points
        title = "All Points"
    
    print(f"Visualizing {len(filtered_points)} points")
    
    # Create figure
    plt.figure(figsize=(16, 12))
    
    # Extract X and Z coordinates (assuming Y is height)
    x = filtered_points[:, 0]
    z = filtered_points[:, 2]
    
    # Create a scatter plot
    plt.scatter(x, z, s=1, alpha=0.5, c=filtered_points[:, 1], cmap='viridis')
    
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.title(title)
    plt.xlabel('X (meters)')
    plt.ylabel('Z (meters)')
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Height (meters)')
    
    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Simple visualization saved to {output_path}")

def create_direct_floorplan(points, colors, output_path, height_threshold=None, 
                           grid_resolution=0.05, dpi=300, color_by_height=True,
                           max_height=2.5, simplified=True, fast_mode=True):
    """
    Create a floor plan directly by projecting points onto a 2D grid
    Similar to the approach that worked before but for point cloud data
    
    Args:
        points: Nx3 numpy array of points
        colors: Nx3 numpy array of colors
        output_path: Path to save the output image
        height_threshold: If not None, only show points above this height
        grid_resolution: Resolution of the grid in meters
        dpi: DPI for output image
        color_by_height: Whether to color by height
        max_height: Maximum height for color normalization
        simplified: Whether to apply morphological operations for cleanup
        fast_mode: If True, use faster processing with downsampling
    """
    print("Creating direct floor plan visualization...")
    
    # Filter points by height if threshold is provided
    if height_threshold is not None:
        height_mask = points[:, 1] >= height_threshold
        filtered_points = points[height_mask]
        filtered_colors = colors[height_mask] if colors is not None else None
        title = f"Floor Plan - Objects Above {height_threshold:.2f}m"
    else:
        filtered_points = points
        filtered_colors = colors
        title = "Floor Plan - All Objects"
    
    # In fast mode, downsample the points to speed up processing
    if fast_mode and len(filtered_points) > 50000:
        # Determine the sampling rate based on the number of points
        sample_rate = max(0.1, min(0.5, 50000 / len(filtered_points)))
        print(f"Fast mode: Downsampling points by factor {sample_rate:.2f}")
        
        # Random downsampling
        indices = np.random.choice(
            len(filtered_points), 
            size=int(len(filtered_points) * sample_rate), 
            replace=False
        )
        filtered_points = filtered_points[indices]
        if filtered_colors is not None:
            filtered_colors = filtered_colors[indices]
    
    print(f"Using {len(filtered_points)} points for floor plan")
    
    if len(filtered_points) == 0:
        print("No points found above threshold, cannot create floor plan")
        return
    
    # Determine bounds of the scene in XZ plane (top down view)
    min_x, max_x = np.min(filtered_points[:, 0]), np.max(filtered_points[:, 0])
    min_z, max_z = np.min(filtered_points[:, 2]), np.max(filtered_points[:, 2])
    
    # Add padding
    padding = 0.5  # meters
    min_x -= padding
    max_x += padding
    min_z -= padding
    max_z += padding
    
    # In fast mode, use a coarser grid resolution
    if fast_mode:
        # Use a coarser grid to speed up processing
        grid_resolution = grid_resolution * 2
        print(f"Fast mode: Using coarser grid resolution: {grid_resolution}m")
    
    # Create grid dimensions
    grid_width = int((max_x - min_x) / grid_resolution) + 1
    grid_height = int((max_z - min_z) / grid_resolution) + 1
    
    print(f"Grid dimensions: {grid_width}x{grid_height} cells at {grid_resolution}m resolution")
    
    # Create empty grid
    grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255  # White background
    
    # Create height grid for coloring
    height_grid = np.zeros((grid_height, grid_width), dtype=np.float32)
    
    # Project points onto grid - use numpy operations instead of loop for speed
    grid_x = ((filtered_points[:, 0] - min_x) / grid_resolution).astype(int)
    grid_z = ((filtered_points[:, 2] - min_z) / grid_resolution).astype(int)
    
    # Make sure points are within grid bounds
    valid_indices = (grid_x >= 0) & (grid_x < grid_width) & (grid_z >= 0) & (grid_z < grid_height)
    grid_x = grid_x[valid_indices]
    grid_z = grid_z[valid_indices]
    heights = filtered_points[valid_indices, 1]  # Y is height
    
    # Create a sparse grid representation for faster processing
    occupied_points = np.zeros((grid_height, grid_width), dtype=np.uint8)
    
    # Mark occupied cells
    for i in range(len(grid_x)):
        # Mark as occupied
        occupied_points[grid_z[i], grid_x[i]] = 1
        
        # Track maximum height at each cell for coloring
        if height_grid[grid_z[i], grid_x[i]] < heights[i]:
            height_grid[grid_z[i], grid_x[i]] = heights[i]
            
            # Use color from input if available, otherwise use height for color
            if filtered_colors is not None:
                grid[grid_z[i], grid_x[i]] = (filtered_colors[valid_indices][i] * 255).astype(np.uint8)
            else:
                # Normalize height for coloring
                h = min(1.0, heights[i] / max_height)
                r = int(255 * h)
                g = int(255 * (1 - abs(2 * h - 1)))
                b = int(255 * (1 - h))
                grid[grid_z[i], grid_x[i]] = [b, g, r]  # BGR format
    
    # Create a binary occupancy grid
    occupied = occupied_points.astype(np.uint8)
    
    if simplified:
        # Apply morphological operations to clean up
        kernel_close = np.ones((3, 3), np.uint8)  # Smaller kernel for speed
        
        # Close to fill small gaps
        cleaned = cv2.morphologyEx(occupied, cv2.MORPH_CLOSE, kernel_close)
        
        # In fast mode, skip the opening operation
        if not fast_mode:
            kernel_open = np.ones((3, 3), np.uint8)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)
        
        # Find contours for better visualization
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"Found {len(contours)} contours")
        
        # Create new clean grid
        clean_grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
        
        # Draw filled contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5:  # Filter small noise (lower threshold for speed)
                # Optional: simplify contour
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Get average height inside contour for coloring
                mask = np.zeros((grid_height, grid_width), dtype=np.uint8)
                cv2.drawContours(mask, [contour], 0, 1, -1)
                heights_in_contour = height_grid[mask == 1]
                
                if len(heights_in_contour) > 0 and color_by_height:
                    avg_height = np.mean(heights_in_contour)
                    # Normalize height to 0-1
                    h = min(1.0, avg_height / max_height)
                    r = int(255 * h)
                    g = int(255 * (1 - abs(2 * h - 1)))
                    b = int(255 * (1 - h))
                    fill_color = (b, g, r)  # BGR
                else:
                    fill_color = (180, 180, 180)  # Gray
                
                # Fill and draw outline
                cv2.drawContours(clean_grid, [approx], 0, fill_color, -1)
                cv2.drawContours(clean_grid, [approx], 0, (0, 0, 0), 1)  # Thinner outline for speed
        
        # Use the cleaned grid
        final_grid = clean_grid
    else:
        # Simple visualization without contour processing
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(occupied, kernel, iterations=1)
        edges = dilated & ~occupied
        
        # Apply edges to original grid
        grid[edges] = [0, 0, 0]  # Black outline
        final_grid = grid
    
    # Add scale bar (1 meter)
    scale_bar_length = int(1.0 / grid_resolution)
    scale_bar_height = 10 if fast_mode else 20
    scale_bar_margin = 30 if fast_mode else 50
    
    # Position scale bar at bottom right
    scale_bar_start_x = min(grid_width - scale_bar_margin - scale_bar_length, grid_width - 10)
    scale_bar_y = min(grid_height - scale_bar_margin, grid_height - 10)
    
    if scale_bar_start_x > 0 and scale_bar_y > 0 and scale_bar_y < grid_height and scale_bar_start_x + scale_bar_length < grid_width:
        # Draw scale bar
        final_grid[scale_bar_y:scale_bar_y+scale_bar_height, 
                  scale_bar_start_x:scale_bar_start_x+scale_bar_length] = [0, 0, 0]
        
        # Add scale text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(final_grid, "1m", (scale_bar_start_x, scale_bar_y - 5), 
                    font, 0.5, (0, 0, 0), 1)
    
    # Add title
    cv2.putText(final_grid, title, (10, 20), font, 0.7, (0, 0, 0), 1)
    
    # Save the floor plan
    plt.figure(figsize=(grid_width/100, grid_height/100), dpi=dpi)
    plt.imshow(cv2.cvtColor(final_grid, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close()
    
    print(f"Direct floor plan saved to {output_path}")
    return final_grid

def pointcloud_to_floorplan(pointcloud_path, output_dir=None, 
                           grid_resolution=0.05, min_area=0.1,
                           height_min=0.1, height_max=2.5, num_slices=5,
                           height_threshold=None,
                           visualize=True, color_by_height=True,
                           show_grid=True, dpi=300, fast_mode=True):
    """
    Convert a point cloud to a 2D floor plan with height information
    
    Args:
        pointcloud_path: Path to the input point cloud
        output_dir: Directory to save output files (default: same as input)
        grid_resolution: Resolution of the grid in meters
        min_area: Minimum area in square meters for a valid shape
        height_min: Minimum height to consider
        height_max: Maximum height to consider
        num_slices: Number of height slices to create
        height_threshold: If not None, only consider points above this height
        visualize: Whether to create visualization
        color_by_height: Whether to color regions by height
        show_grid: Whether to show grid lines in visualization
        dpi: DPI for output image
        fast_mode: Whether to use fast processing mode (default: True)
        
    Returns:
        True if successful, False otherwise
    """
    # Set default output directory
    if output_dir is None:
        output_dir = os.path.dirname(pointcloud_path)
        # If dirname returned empty string (when file is in current directory), use '.'
        if output_dir == '':
            output_dir = '.'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate output paths
    base_name = os.path.splitext(os.path.basename(pointcloud_path))[0]
    floorplan_image_path = os.path.join(output_dir, f"{base_name}_floorplan.png")
    floorplan_data_path = os.path.join(output_dir, f"{base_name}_floorplan.txt")
    simple_viz_path = os.path.join(output_dir, f"{base_name}_simple_viz.png")
    direct_floorplan_path = os.path.join(output_dir, f"{base_name}_direct_floorplan.png")
    
    # Load point cloud
    pcd = load_pointcloud(pointcloud_path)
    if pcd is None:
        print("Failed to load point cloud. Exiting.")
        return False
    
    # Extract points and colors
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    try:
        # Detect floor plane
        floor_model, floor_inliers = detect_floor_plane(pcd)
        
        # Normalize points so floor is at y=0
        normalized_points = normalize_points_to_floor(points, floor_model)
        
        # Create a direct floor plan visualization using the simpler approach
        create_direct_floorplan(
            normalized_points, 
            colors, 
            direct_floorplan_path,
            height_threshold=height_threshold,
            grid_resolution=grid_resolution,
            dpi=dpi,
            color_by_height=color_by_height,
            max_height=height_max,
            fast_mode=fast_mode
        )
        print(f"Created direct floor plan at {direct_floorplan_path}")
        
        # Continue with the regular processing...
        
        # Create a simple direct visualization of the points
        create_simple_point_visualization(normalized_points, simple_viz_path, height_threshold, dpi)
        print(f"Created simple visualization at {simple_viz_path}")
        
        # Continue with the normal process
        # Determine floor bounds (for visualization)
        floor_points = normalized_points[floor_inliers]
        floor_min_x, floor_min_z = np.min(floor_points[:, [0, 2]], axis=0)
        floor_max_x, floor_max_z = np.max(floor_points[:, [0, 2]], axis=0)
        floor_bounds = [floor_min_x, floor_min_z, floor_max_x, floor_max_z]
        
        # Create height slices
        slices = create_height_slices(
            normalized_points, colors,
            height_min=height_min,
            height_max=height_max,
            num_slices=num_slices,
            height_threshold=height_threshold
        )
        
        # If using height threshold, adjust floor bounds to focus on objects above threshold
        if height_threshold is not None and len(slices) > 0 and len(slices[0][0]) > 0:
            threshold_points = slices[0][0]  # Get points from the first (and only) slice
            # Calculate tighter bounds based on threshold points
            threshold_min_x, threshold_min_z = np.min(threshold_points[:, [0, 2]], axis=0)
            threshold_max_x, threshold_max_z = np.max(threshold_points[:, [0, 2]], axis=0)
            
            # Add some padding (10% of the range)
            padding_x = 0.1 * (threshold_max_x - threshold_min_x)
            padding_z = 0.1 * (threshold_max_z - threshold_min_z)
            
            # Use these tighter bounds instead
            floor_bounds = [
                threshold_min_x - padding_x,
                threshold_min_z - padding_z,
                threshold_max_x + padding_x,
                threshold_max_z + padding_z
            ]
            print(f"Using tighter bounds for height threshold: {floor_bounds}")
        
        # Create colored height map
        # When using height threshold, use a larger grid resolution and smaller min_area
        if height_threshold is not None:
            # Use coarser grid resolution for height threshold mode (to better connect points)
            threshold_grid_resolution = grid_resolution * 2
            threshold_min_area = min_area / 4  # More permissive min_area
            print(f"Using coarser grid resolution: {threshold_grid_resolution} and smaller min_area: {threshold_min_area}")
            
            polygons, combined_grid, combined_origin = create_colored_height_map(
                slices,
                grid_resolution=threshold_grid_resolution,
                min_area=threshold_min_area,
                height_threshold=height_threshold
            )
        else:
            polygons, combined_grid, combined_origin = create_colored_height_map(
                slices,
                grid_resolution=grid_resolution,
                min_area=min_area,
                height_threshold=height_threshold
            )
        
        # Create visualization
        if visualize:
            plot_floorplan(
                polygons,
                floorplan_image_path,
                background_grid=combined_grid,
                grid_origin=combined_origin,
                grid_resolution=grid_resolution,
                floor_bounds=floor_bounds,
                show_grid=show_grid,
                color_by_height=color_by_height,
                dpi=dpi,
                height_threshold=height_threshold
            )
        
        # Save data
        save_floorplan_data(polygons, floorplan_data_path)
        
        print("Floor plan generation complete!")
        return True
        
    except Exception as e:
        print(f"Error creating floor plan: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a point cloud to a 2D floor plan")
    
    # Input/output arguments
    parser.add_argument("--pointcloud_path", type=str, required=True,
                       help="Path to the input point cloud (.ply file)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save output files (default: same as input)")
    
    # Processing parameters
    parser.add_argument("--grid_resolution", type=float, default=0.05,
                       help="Resolution of the grid in meters (default: 0.05)")
    parser.add_argument("--min_area", type=float, default=0.1,
                       help="Minimum area in square meters for valid shapes (default: 0.1)")
    parser.add_argument("--height_min", type=float, default=0.1,
                       help="Minimum height to consider in meters (default: 0.1)")
    parser.add_argument("--height_max", type=float, default=2.5,
                       help="Maximum height to consider in meters (default: 2.5)")
    parser.add_argument("--num_slices", type=int, default=5,
                       help="Number of height slices to create (default: 5)")
    parser.add_argument("--height_threshold", type=float, default=None,
                       help="If specified, only show objects above this height (in meters)")
    
    # Visualization options
    parser.add_argument("--no_visualization", action="store_true",
                       help="Skip generating a visualization image")
    parser.add_argument("--no_color", action="store_true",
                       help="Don't color regions by height")
    parser.add_argument("--no_grid", action="store_true",
                       help="Don't show grid lines in visualization")
    parser.add_argument("--dpi", type=int, default=300,
                       help="DPI for the output image (default: 300)")
    parser.add_argument("--no_fast_mode", action="store_true",
                       help="Disable fast processing mode (more accurate but slower)")
    
    args = parser.parse_args()
    
    # Convert point cloud to floor plan
    success = pointcloud_to_floorplan(
        pointcloud_path=args.pointcloud_path,
        output_dir=args.output_dir,
        grid_resolution=args.grid_resolution,
        min_area=args.min_area,
        height_min=args.height_min,
        height_max=args.height_max,
        num_slices=args.num_slices,
        height_threshold=args.height_threshold,
        visualize=not args.no_visualization,
        color_by_height=not args.no_color,
        show_grid=not args.no_grid,
        dpi=args.dpi,
        fast_mode=not args.no_fast_mode
    )
    
    if success:
        print("Conversion successful!")
        base_name = os.path.splitext(os.path.basename(args.pointcloud_path))[0]
        
        if args.output_dir:
            print(f"Floor plan saved to: {os.path.join(args.output_dir, base_name+'_floorplan.png')}")
            print(f"Floor plan data saved to: {os.path.join(args.output_dir, base_name+'_floorplan.txt')}")
        else:
            output_dir = os.path.dirname(args.pointcloud_path)
            print(f"Floor plan saved to: {os.path.join(output_dir, base_name+'_floorplan.png')}")
            print(f"Floor plan data saved to: {os.path.join(output_dir, base_name+'_floorplan.txt')}")
    else:
        print("Conversion failed.") 