# 3D Effects from 2D Images

This collection of scripts allows you to convert 2D images into various 3D representations using the Depth Pro model for monocular depth estimation.

## Prerequisites

Before using these scripts, make sure you have installed the required dependencies:

```bash
# Install depth_pro according to the main README
pip install -e .

# Install additional dependencies
pip install numpy opencv-python open3d tqdm
```

## Available Scripts

### 1. Point Cloud Generation (`video3D.py`)

Convert a 2D image into a 3D point cloud.

```bash
python video3D.py --image_path path/to/your/image.jpg --output_path output/pointcloud.ply
```

Options:
- `--image_path`: Path to the input image (required)
- `--output_path`: Path to save the point cloud (optional)
- `--no_visualize`: Disable visualization

### 2. 3D Mesh Generation (`mesh_from_depth.py`)

Convert a 2D image into a 3D mesh using two different methods.

```bash
python mesh_from_depth.py --image_path path/to/your/image.jpg --output_path output/mesh.ply --method simple
```

Options:
- `--image_path`: Path to the input image (required)
- `--output_path`: Path to save the mesh (optional)
- `--method`: Mesh creation method: 'poisson' or 'simple' (default: 'simple')
- `--no_visualize`: Disable visualization
- `--depth_scaling`: Scale factor for depth values (default: 1.0)

### 3. 3D Visual Effects (`depth_video_effect.py`)

Create 3D visual effects from a single image.

#### Parallax Effect (3D Video)

```bash
python depth_video_effect.py --image_path path/to/your/image.jpg --output_path output/video.mp4 --effect parallax --motion circle
```

Options:
- `--image_path`: Path to the input image (required)
- `--output_path`: Path to save the output video (required)
- `--effect`: Type of effect ('parallax' or 'anaglyph')
- `--duration`: Duration of video in seconds (default: 5.0)
- `--fps`: Frames per second (default: 30)
- `--amplitude`: Amplitude of camera motion (default: 0.05)
- `--motion`: Type of camera motion ('circle', 'zoom', or 'swing')
- `--scale`: Scale factor for output resolution (default: 1.0)

#### 3D Anaglyph (Red-Cyan 3D Image)

```bash
python depth_video_effect.py --image_path path/to/your/image.jpg --output_path output/anaglyph.jpg --effect anaglyph
```

Options:
- `--separation`: Separation between left and right views (default: 0.05)

## Examples

### Creating a Point Cloud

```bash
python video3D.py --image_path data/example.jpg --output_path output/example_pointcloud.ply
```

### Creating a 3D Mesh

```bash
python mesh_from_depth.py --image_path data/example.jpg --output_path output/example_mesh.ply --method simple
```

### Creating a Parallax Effect Video

```bash
python depth_video_effect.py --image_path data/example.jpg --output_path output/example_parallax.mp4 --effect parallax --motion circle --amplitude 0.07
```

### Creating a 3D Anaglyph Image

```bash
python depth_video_effect.py --image_path data/example.jpg --output_path output/example_anaglyph.jpg --effect anaglyph --separation 0.03
```

## Tips for Best Results

1. **Image Selection**: Choose images with clear foreground and background separation for the best 3D effect.

2. **Depth Scaling**: Adjust the depth scaling parameter to enhance or reduce the 3D effect.

3. **Motion Parameters**: For parallax videos, experiment with different motion types and amplitudes to find the most pleasing effect.

4. **Anaglyph Viewing**: Use red-cyan 3D glasses to view anaglyph images.

5. **Point Cloud Filtering**: For better point cloud visualization, you may want to filter out points that are too far away or apply smoothing.

## How It Works

These scripts use the Depth Pro model to estimate the depth of each pixel in the input image. The depth map is then used to:

1. **Point Cloud**: Project each pixel into 3D space based on its depth.
2. **Mesh**: Create a 3D mesh by connecting neighboring pixels and using the depth as the Z-coordinate.
3. **Parallax Effect**: Simulate camera movement by shifting pixels based on their depth.
4. **Anaglyph**: Create separate left and right views with different perspectives based on depth.

The Depth Pro model is particularly good at capturing fine details and producing sharp depth boundaries, which results in high-quality 3D conversions. 