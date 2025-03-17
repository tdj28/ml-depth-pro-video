# Image to 3D Mesh Pipeline

This project provides tools to convert a single image into a clean 3D mesh through monocular depth estimation, point cloud processing, and mesh generation.

## Features

- **Normalized Point Cloud Generation:** Creates 3D point clouds with ground plane aligned to y=0
- **Stray Point Removal:** Eliminates disconnected points that don't have enough neighbors
- **Shadow Artifact Cleanup:** Identifies and removes shadow artifacts caused by occlusion
- **Mesh Generation:** Converts the processed point cloud into a clean, textured 3D mesh
- **Visualization:** Generates preview images of the created meshes

## Requirements

Make sure you have all the required dependencies installed:

```bash
pip install -e .  # Install depth_pro from the local repo
pip install -r 3d_effects_requirements.txt
```

## Usage

### Basic Usage

The simplest way to use this tool is with the demo script:

```bash
python demo_pointcloud_to_mesh.py --image_path path/to/your/image.jpg
```

This will:
1. Generate a normalized point cloud from the image
2. Remove stray points and shadow artifacts
3. Create a mesh using Poisson surface reconstruction
4. Save the mesh as `[image_name]_mesh.obj`
5. Generate a preview image as `[image_name]_mesh_preview.png`

### Quality Settings

You can adjust processing quality with these flags:

```bash
# Fast processing (lower quality, quicker results)
python demo_pointcloud_to_mesh.py --image_path path/to/your/image.jpg --fast

# High quality processing (better quality, slower)
python demo_pointcloud_to_mesh.py --image_path path/to/your/image.jpg --high_quality
```

### Advanced Usage

For more control, you can use the main script directly:

```bash
python pointcloud_to_mesh.py --image_path path/to/your/image.jpg [options]
```

#### Options

- `--output_path`: Path to save the output mesh (default: `[image_name]_mesh.obj`)
- `--voxel_size`: Voxel size for mesh creation (smaller = more detail but slower)
- `--depth`: Maximum depth of the octree for Poisson reconstruction (8-9 recommended)
- `--skip_stray_removal`: Skip the stray point removal step
- `--skip_shadow_cleanup`: Skip the shadow artifact cleanup step
- `--downscale_factor`: Downscale input image by this factor for faster processing (e.g., 0.5 = half size)
- `--grid_size`: Number of grid cells for ground adjustment
- `--ground_percentile`: Percentile of lowest points to consider for ground adjustment

## How It Works

### 1. Normalized Point Cloud Generation

The pipeline starts by estimating depth from the input image using a pre-trained monocular depth estimation model. The depth map is converted to a 3D point cloud, normalized so the ground plane is at y=0.

### 2. Stray Point Removal

Using nearest neighbor analysis, the pipeline identifies and removes points that don't have enough neighbors within a specified radius. This helps eliminate floating artifacts that aren't well-connected to the main scene.

### 3. Shadow Artifact Cleanup

One limitation of monocular depth estimation is the inability to see behind objects, which creates "shadow" artifacts - thin vertical structures that extend from foreground objects. This step detects and removes these shadows by analyzing the point distribution in vertical columns.

### 4. Mesh Generation

The cleaned point cloud is converted to a mesh using Poisson surface reconstruction. This process:
- Estimates surface normals for each point
- Downsamples the point cloud to improve processing speed
- Applies Poisson reconstruction to create a water-tight mesh
- Filters out low-density areas that may represent noise
- Cleans up the mesh by removing degenerate triangles, duplicates, etc.

## Examples

Below are example use cases:

1. Creating a mesh from a landscape photo:
```bash
python demo_pointcloud_to_mesh.py --image_path landscape.jpg --high_quality
```

2. Quickly processing a room interior:
```bash
python demo_pointcloud_to_mesh.py --image_path room.jpg --fast
```

3. Fine-tuning parameters for a specific image:
```bash
python pointcloud_to_mesh.py --image_path photo.jpg --voxel_size 0.03 --depth 9 --downscale_factor 0.8
```

## Limitations

- Works best with well-lit images that have clear depth cues
- Performance depends on image quality and complexity
- The mesh may not capture very fine details
- Occlusions can still cause some artifacts in the final mesh

## Acknowledgments

This project builds upon the img_to_normalized_pointcloud.py script and uses the depth_pro framework for monocular depth estimation. 