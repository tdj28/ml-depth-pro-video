import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import cv2
from sklearn.cluster import DBSCAN
from scipy import optimize
from shapely.geometry import Polygon, Point

def fit_circle(points):
    """
    Fit a circle to a set of 2D points using least squares
    
    Args:
        points: Nx2 array of points
        
    Returns:
        center_x, center_y, radius
    """
    def calc_R(xc, yc):
        """Calculate distances from center to all points"""
        return np.sqrt((points[:, 0] - xc) ** 2 + (points[:, 1] - yc) ** 2)

    def f_2(c):
        """Calculate sum of squared errors"""
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    # Initial guess: mean of points
    x_mean = np.mean(points[:, 0])
    y_mean = np.mean(points[:, 1])
    
    center_estimate = x_mean, y_mean
    center_2, _ = optimize.leastsq(f_2, center_estimate)
    
    xc, yc = center_2
    Ri = calc_R(xc, yc)
    radius = Ri.mean()
    
    return xc, yc, radius

def is_better_fit_as_circle(points, rectangle_area, circle_area, circle_fit_error, circularity_threshold=0.85):
    """
    Determine if a set of points is better fit as a circle than a rectangle
    
    Args:
        points: Nx2 array of points
        rectangle_area: Area of fitted rectangle
        circle_area: Area of fitted circle
        circle_fit_error: Error of circle fit
        circularity_threshold: Threshold for circularity (0-1)
        
    Returns:
        True if better fit as circle, False otherwise
    """
    # Calculate convex hull for shape analysis
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
        
        # Calculate circularity as ratio of hull area to circle area
        # Perfect circle has ratio close to 1
        circularity = hull_area / circle_area
        
        # Adjust for the fact that hull area can be slightly smaller than circle area
        circularity = min(circularity, 1.0/circularity)
        
        # Check if shape is more circular than rectangular
        return (circularity > circularity_threshold and 
                circle_fit_error < 0.15 and  # Low fit error
                abs(circle_area - rectangle_area) / max(circle_area, rectangle_area) < 0.3)  # Areas are similar
    except:
        # If hull calculation fails, use a simpler heuristic
        return circle_fit_error < 0.1 and abs(circle_area - rectangle_area) / max(circle_area, rectangle_area) < 0.2

def detect_and_split_l_shapes(rectangles, points_2d, min_overlap_ratio=0.8):
    """
    Detect and split L-shaped structures into multiple rectangles
    
    Args:
        rectangles: List of rectangle parameters (center_x, center_y, width, height, angle)
        points_2d: Original 2D points used for clustering
        min_overlap_ratio: Minimum ratio of points that must be covered by new rectangles
        
    Returns:
        new_rectangles: Updated list of rectangles with L-shapes split
    """
    new_rectangles = []
    
    # Process each existing rectangle
    for rect_idx, rect in enumerate(rectangles):
        center_x, center_y, width, height, angle = rect
        
        # Only consider larger rectangles for L-shape detection
        if width * height < 10.0:  # Skip rectangles smaller than 10 square meters
            new_rectangles.append(rect)
            continue
            
        # Create the rectangle vertices
        rect_box = cv2.boxPoints((
            (center_x, center_y),
            (width, height),
            angle
        ))
        rect_poly = Polygon(rect_box)
        
        # Find points inside this rectangle
        cluster_points = []
        for point in points_2d:
            if rect_poly.contains(Point(point)):
                cluster_points.append(point)
        
        cluster_points = np.array(cluster_points)
        
        if len(cluster_points) < 50:  # Need enough points for reliable analysis
            new_rectangles.append(rect)
            continue
        
        # Rotate points to align with rectangle axes
        # Calculate rotation angle to align with axes
        rotation_angle = -angle * np.pi / 180.0
        rotation_matrix = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)]
        ])
        
        # Translate to origin, rotate, then translate back
        centered_points = cluster_points - np.array([center_x, center_y])
        rotated_points = np.dot(centered_points, rotation_matrix.T)
        
        # Calculate the aligned rectangle dimensions
        aligned_width, aligned_height = width, height
        
        # Create a grid to analyze point density within the rectangle
        grid_size = 0.2  # 20cm grid cells
        grid_width = int(aligned_width / grid_size) + 1
        grid_height = int(aligned_height / grid_size) + 1
        
        if grid_width <= 2 or grid_height <= 2:  # Rectangle too small for meaningful grid
            new_rectangles.append(rect)
            continue
        
        # Create density grid
        density_grid = np.zeros((grid_height, grid_width), dtype=np.int32)
        
        # Map points to grid cells
        for point in rotated_points:
            x, y = point
            # Convert to grid coordinates (origin at center of rectangle)
            grid_x = int((x + aligned_width/2) / grid_size)
            grid_y = int((y + aligned_height/2) / grid_size)
            
            # Ensure within bounds
            if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                density_grid[grid_y, grid_x] += 1
        
        # Convert to binary occupied/empty grid
        binary_grid = (density_grid > 0).astype(np.uint8)
        
        # Dilate to connect nearby points
        kernel = np.ones((2, 2), np.uint8)
        dilated_grid = cv2.dilate(binary_grid, kernel, iterations=1)
        
        # Identify empty regions in the grid
        empty_grid = 1 - dilated_grid
        
        # Only consider substantial empty regions
        min_empty_size = 6  # Minimum size of empty region to consider
        
        # Label connected empty regions
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(empty_grid, connectivity=4)
        
        # Find significant empty regions (excluding background which is label 0)
        significant_empty_regions = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_empty_size:
                significant_empty_regions.append(i)
        
        # No significant empty regions - not an L-shape
        if len(significant_empty_regions) == 0:
            new_rectangles.append(rect)
            continue
        
        # Create a mask of significant empty regions
        empty_mask = np.zeros_like(empty_grid)
        for region_id in significant_empty_regions:
            empty_mask[labels == region_id] = 1
        
        # If empty region is more than 25% of rectangle area, it's potentially an L-shape
        empty_ratio = np.sum(empty_mask) / (grid_width * grid_height)
        if empty_ratio < 0.2 or empty_ratio > 0.6:  # Not in the range expected for L-shapes
            new_rectangles.append(rect)
            continue
            
        print(f"Detected L-shape in rectangle {rect_idx} (empty ratio: {empty_ratio:.2f})")
        
        # Find the largest empty region
        largest_empty_region = 0
        largest_area = 0
        for region_id in significant_empty_regions:
            area = np.sum(labels == region_id)
            if area > largest_area:
                largest_area = area
                largest_empty_region = region_id
        
        # Create a mask for occupied regions
        occupied_mask = 1 - empty_mask
        
        # Find connected occupied regions
        num_occupied, occupied_labels = cv2.connectedComponents(occupied_mask, connectivity=4)
        
        # Skip if we don't have clearly separate occupied regions
        if num_occupied <= 2:  # Background counts as one
            new_rectangles.append(rect)
            continue
            
        # Process each occupied region to create a rectangle
        sub_rectangles = []
        
        for i in range(1, num_occupied):
            # Create a mask for this region
            region_mask = (occupied_labels == i).astype(np.uint8)
            
            # Only consider regions with sufficient area
            region_area = np.sum(region_mask)
            if region_area < 6:  # Skip very small regions
                continue
                
            # Find points in this region
            region_points = []
            for y in range(grid_height):
                for x in range(grid_width):
                    if region_mask[y, x] > 0:
                        # Convert from grid to rotated coordinates
                        rx = (x * grid_size) - aligned_width/2
                        ry = (y * grid_size) - aligned_height/2
                        region_points.append([rx, ry])
            
            if len(region_points) < 4:  # Need at least 4 points for a rectangle
                continue
                
            # Fit rectangle to these points
            region_points = np.array(region_points)
            region_rect = cv2.minAreaRect(region_points.astype(np.float32))
            (sub_cx, sub_cy), (sub_w, sub_h), sub_angle = region_rect
            
            # Convert back to original coordinate system
            # First rotate the center point back
            rotation_matrix_inv = rotation_matrix.T  # Inverse rotation
            rotated_center = np.dot(np.array([sub_cx, sub_cy]), rotation_matrix_inv)
            orig_cx = rotated_center[0] + center_x
            orig_cy = rotated_center[1] + center_y
            
            # Adjust angle for original orientation
            orig_angle = (sub_angle + angle) % 180
            
            # Add sub-rectangle if it's a reasonable size
            if sub_w * sub_h > 1.0:  # At least 1 square meter
                sub_rectangles.append((orig_cx, orig_cy, sub_w, sub_h, orig_angle))
        
        # If we successfully created sub-rectangles, add them to the result
        # Otherwise, keep the original rectangle
        if len(sub_rectangles) >= 2:
            # Calculate total area of sub-rectangles
            sub_area = sum(w * h for _, _, w, h, _ in sub_rectangles)
            orig_area = width * height
            
            # Only replace if the sub-rectangles have a reasonable total area
            # compared to the original (to avoid over-splitting)
            if 0.4 < sub_area / orig_area < 1.3:
                new_rectangles.extend(sub_rectangles)
                print(f"Split rectangle {rect_idx} into {len(sub_rectangles)} sub-rectangles")
            else:
                new_rectangles.append(rect)
        else:
            new_rectangles.append(rect)
    
    return new_rectangles

def split_large_rectangle(rect, points_2d):
    """
    Forcibly split a large rectangle into smaller ones
    
    Args:
        rect: Rectangle parameters (center_x, center_y, width, height, angle)
        points_2d: Points to consider for the split
        
    Returns:
        List of new rectangles
    """
    center_x, center_y, width, height, angle = rect
    
    # Convert angle to radians
    angle_rad = angle * np.pi / 180.0
    
    # Large rectangle - try to split it in half along the longer side
    if width > height:
        # Split along width
        split_offset = width / 4  # Try to split at 1/4 and 3/4 points
        
        # Create points for the two halves
        left_center_x = center_x - split_offset * np.cos(angle_rad)
        left_center_y = center_y - split_offset * np.sin(angle_rad)
        right_center_x = center_x + split_offset * np.cos(angle_rad)
        right_center_y = center_y + split_offset * np.sin(angle_rad)
        
        # Create rectangles for each half
        left_rect = (left_center_x, left_center_y, width/2, height, angle)
        right_rect = (right_center_x, right_center_y, width/2, height, angle)
        
        return [left_rect, right_rect]
    else:
        # Split along height
        split_offset = height / 4  # Try to split at 1/4 and 3/4 points
        
        # Create points for the two halves
        bottom_center_x = center_x - split_offset * np.sin(angle_rad)
        bottom_center_y = center_y + split_offset * np.cos(angle_rad)
        top_center_x = center_x + split_offset * np.sin(angle_rad)
        top_center_y = center_y - split_offset * np.cos(angle_rad)
        
        # Create rectangles for each half
        bottom_rect = (bottom_center_x, bottom_center_y, width, height/2, angle)
        top_rect = (top_center_x, top_center_y, width, height/2, angle)
        
        return [bottom_rect, top_rect]

def fit_shapes_to_clusters(points_2d, eps=0.2, min_samples=5, circularity_threshold=0.85):
    """
    Find clusters of points and fit rectangles or circles to them
    
    Args:
        points_2d: Nx2 array of 2D points
        eps: DBSCAN epsilon parameter (max distance between points in a cluster)
        min_samples: DBSCAN min_samples parameter
        circularity_threshold: Threshold for circularity (0-1)
        
    Returns:
        rectangles: List of rectangle parameters (center_x, center_y, width, height, angle)
        circles: List of circle parameters (center_x, center_y, radius)
    """
    # Cluster points using DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_2d)
    labels = clustering.labels_
    
    # Count number of clusters (excluding noise with label -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Found {n_clusters} clusters")
    
    rectangles = []
    circles = []
    
    # Process each cluster
    for cluster_id in range(n_clusters):
        # Get points in this cluster
        cluster_points = points_2d[labels == cluster_id]
        
        if len(cluster_points) < 5:  # Skip very small clusters
            continue
            
        # Try rectangle fit (minimum area rectangle)
        rect = cv2.minAreaRect(cluster_points.astype(np.float32))
        box = cv2.boxPoints(rect)
        
        # Calculate rectangle area
        rect_width = np.linalg.norm(box[0] - box[1])
        rect_height = np.linalg.norm(box[1] - box[2])
        rect_area = rect_width * rect_height
        
        # Try circle fit
        try:
            xc, yc, radius = fit_circle(cluster_points)
            circle_area = np.pi * radius**2
            
            # Calculate circle fit error (mean squared error)
            distances = np.sqrt((cluster_points[:, 0] - xc)**2 + (cluster_points[:, 1] - yc)**2)
            circle_fit_error = np.mean((distances - radius)**2) / radius**2
            
            # Decide if this is better as a circle or rectangle
            if is_better_fit_as_circle(cluster_points, rect_area, circle_area, circle_fit_error, circularity_threshold):
                circles.append((xc, yc, radius))
            else:
                # Handle the specific case of very large rectangles with a lot of points
                (center_x, center_y), (width, height), angle = rect
                
                # If this is a very large rectangle, force split it into multiple smaller ones
                if width * height > 100 and len(cluster_points) > 1000:
                    print(f"Splitting large rectangle with area {width * height:.1f}m²")
                    sub_rects = split_large_rectangle((center_x, center_y, width, height, angle), cluster_points)
                    rectangles.extend(sub_rects)
                else:
                    rectangles.append((center_x, center_y, width, height, angle))
        except:
            # If circle fitting fails, default to rectangle
            (center_x, center_y), (width, height), angle = rect
            
            # If this is a very large rectangle, force split it into multiple smaller ones
            if width * height > 100 and len(cluster_points) > 1000:
                print(f"Splitting large rectangle with area {width * height:.1f}m²")
                sub_rects = split_large_rectangle((center_x, center_y, width, height, angle), cluster_points)
                rectangles.extend(sub_rects)
            else:
                rectangles.append((center_x, center_y, width, height, angle))
    
    # Now detect and split L-shaped structures
    rectangles = detect_and_split_l_shapes(rectangles, points_2d)
    
    return rectangles, circles

def export_shape_data(rectangles, circles, output_path):
    """
    Export shape data to a simple text file
    
    Args:
        rectangles: List of rectangle parameters (center_x, center_y, width, height, angle)
        circles: List of circle parameters (center_x, center_y, radius)
        output_path: Path to save the output file
    """
    with open(output_path, 'w') as f:
        f.write("# Floor Plan Shape Data\n")
        f.write("# Units: meters\n\n")
        
        f.write(f"Total Shapes: {len(rectangles) + len(circles)}\n")
        f.write(f"Rectangles: {len(rectangles)}\n")
        f.write(f"Circles: {len(circles)}\n\n")
        
        # Calculate total area
        total_rect_area = sum(rect[2] * rect[3] for rect in rectangles)
        total_circle_area = sum(np.pi * circ[2]**2 for circ in circles)
        total_area = total_rect_area + total_circle_area
        f.write(f"Total Area: {total_area:.2f} square meters\n\n")
        
        # Write rectangle data
        f.write("# Rectangles\n")
        f.write("# Format: ID, center_x, center_y, width, height, angle_degrees, area_m2\n")
        for i, rect in enumerate(rectangles):
            center_x, center_y, width, height, angle = rect
            area = width * height
            f.write(f"{i+1}, {center_x:.3f}, {center_y:.3f}, {width:.3f}, {height:.3f}, {angle:.1f}, {area:.3f}\n")
        
        f.write("\n# Circles\n")
        f.write("# Format: ID, center_x, center_y, radius, area_m2\n")
        for i, circle in enumerate(circles):
            center_x, center_y, radius = circle
            area = np.pi * radius**2
            circle_id = len(rectangles) + i + 1
            f.write(f"{circle_id}, {center_x:.3f}, {center_y:.3f}, {radius:.3f}, {area:.3f}\n")
    
    print(f"Shape data exported to: {output_path}")

def simple_pointcloud_visualization(pointcloud_path, output_path=None, height_threshold=None, 
                                  point_size=2, min_points=5, dpi=300, max_points=50000,
                                  tight_crop=True, fit_shapes=False, cluster_eps=0.2, 
                                  min_cluster_size=5, circularity_threshold=0.85):
    """
    Create an extremely simple top-down view of a point cloud
    
    Args:
        pointcloud_path: Path to the input point cloud
        output_path: Path to save the output image
        height_threshold: Only show points above this height (meters)
        point_size: Size of points in visualization
        min_points: Minimum number of points required to create visualization
        dpi: DPI for output image
        max_points: Maximum number of points to visualize (downsamples if more)
        tight_crop: Whether to crop view tightly around visible points
        fit_shapes: Whether to fit rectangles and circles to point clusters
        cluster_eps: DBSCAN epsilon parameter (max distance between points in a cluster)
        min_cluster_size: Minimum points for a valid cluster
        circularity_threshold: Threshold for circularity (0-1)
    
    Returns:
        True if successful, False otherwise
    """
    # Set default output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(pointcloud_path))[0]
        output_dir = os.path.dirname(pointcloud_path)
        if output_dir == '':
            output_dir = '.'
        output_path = os.path.join(output_dir, f"{base_name}_simple_view.png")
    
    # Load point cloud
    print(f"Loading point cloud from {pointcloud_path}...")
    try:
        pcd = o3d.io.read_point_cloud(pointcloud_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        print(f"Loaded {len(points)} points")
    except Exception as e:
        print(f"Error loading point cloud: {str(e)}")
        return False
    
    # Basic floor detection (find lowest significant cluster of points)
    # Sort points by height (y-coordinate)
    sorted_indices = np.argsort(points[:, 1])
    sorted_points = points[sorted_indices]
    
    # Assume first 20% of points belong to the floor (or points within 10cm of minimum)
    floor_threshold = min(
        sorted_points[int(len(sorted_points) * 0.2), 1],
        sorted_points[0, 1] + 0.1
    )
    
    print(f"Estimated floor height: {floor_threshold:.3f}m")
    
    # Normalize heights so floor is at y=0
    normalized_points = points.copy()
    normalized_points[:, 1] -= floor_threshold
    
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
    
    if len(vis_points) < min_points:
        print(f"Not enough points to create visualization (only {len(vis_points)})")
        return False
    
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
    
    # Draw fitted shapes
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
    
    # Save a separate image with just the shapes if requested
    if fit_shapes:
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a simple top-down view of a point cloud")
    
    parser.add_argument("--pointcloud_path", type=str, required=True,
                       help="Path to the input point cloud (.ply file)")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Path to save the output image")
    parser.add_argument("--height_threshold", type=float, default=None,
                       help="Only show points above this height (meters)")
    parser.add_argument("--point_size", type=float, default=2,
                       help="Size of points in visualization")
    parser.add_argument("--dpi", type=int, default=300,
                       help="DPI for output image")
    parser.add_argument("--max_points", type=int, default=50000,
                       help="Maximum points to plot (downsamples if more)")
    parser.add_argument("--no_tight_crop", action="store_true",
                       help="Don't crop view tightly around visible points")
    parser.add_argument("--fit_shapes", action="store_true",
                       help="Fit rectangles and circles to point clusters")
    parser.add_argument("--cluster_eps", type=float, default=0.2,
                       help="DBSCAN epsilon parameter (max distance between points in a cluster)")
    parser.add_argument("--min_cluster_size", type=int, default=5,
                       help="Minimum points for a valid cluster")
    parser.add_argument("--circularity_threshold", type=float, default=0.85,
                       help="Threshold for circularity (0-1)")
    
    args = parser.parse_args()
    
    success = simple_pointcloud_visualization(
        pointcloud_path=args.pointcloud_path,
        output_path=args.output_path,
        height_threshold=args.height_threshold,
        point_size=args.point_size,
        dpi=args.dpi,
        max_points=args.max_points,
        tight_crop=not args.no_tight_crop,
        fit_shapes=args.fit_shapes,
        cluster_eps=args.cluster_eps,
        min_cluster_size=args.min_cluster_size,
        circularity_threshold=args.circularity_threshold
    )
    
    if success:
        print("Visualization created successfully!")
    else:
        print("Failed to create visualization.") 