# Point Cloud Floor Plan Pipeline

This is an optimized pipeline that combines three separate processes into a single efficient workflow:

1. **Image to Normalized Point Cloud**: Converts images to 3D point clouds and normalizes them to the ground plane
2. **Point Cloud Cleaning**: Removes stray points and shadow artifacts from the point cloud
3. **Floor Plan Generation**: Creates top-down floor plan visualizations with shape detection

## Features

- **Memory Efficient**: Keeps point clouds in memory between processing steps
- **Ground Plane Reuse**: Calculates ground plane parameters once and reuses them for all frames
- **Parallel Processing**: Supports multi-process parallel execution for faster batch processing
- **Output Options**: Generates multiple visualization types for each frame
- **Hardware Acceleration**: Optimizations for Apple Silicon and CUDA devices

## Usage

```bash
python pointcloud_pipeline.py --frames_dir ./TEMP/FRAMES --output_dir ./outputs
```

### Key Arguments

- `--frames_dir`: Directory containing input image frames
- `--output_dir`: Directory to save outputs (defaults to frames_dir)
- `--start_frame` / `--end_frame`: Process a specific range of frames
- `--height_threshold`: Only show points above this height (meters) (default: 1.3)
- `--point_size`: Size of points in visualization (default: 10)
- `--downscale_factor`: Shrink input images for faster processing (default: 1.0)
- `--half_precision`: Use float16 for faster computation
- `--num_workers`: Number of parallel workers (default: 0 for sequential processing)
- `--fit_shapes`: Detect and draw rectangular and circular shapes (on by default)
- `--visualize_3d`: Generate additional 3D point cloud visualizations
- `--simple_output`: Create simple visualization without shapes or labels
- `--output_main_only`: Only output the main clean_simple_view.png file (no additional files)

### Examples

Process all frames in a directory:
```bash
python pointcloud_pipeline.py --frames_dir ./TEMP/FRAMES --output_dir ./outputs
```

Process specific frames with 4 parallel workers:
```bash
python pointcloud_pipeline.py --frames_dir ./TEMP/FRAMES --output_dir ./outputs --start_frame 100 --end_frame 200 --num_workers 4
```

Speed up processing with reduced quality:
```bash
python pointcloud_pipeline.py --frames_dir ./TEMP/FRAMES --output_dir ./outputs --downscale_factor 0.75 --half_precision
```

Generate only the main visualization without shapes or additional files:
```bash
python pointcloud_pipeline.py --frames_dir ./TEMP/FRAMES --output_dir ./outputs --simple_output --output_main_only
```

## Output Files

For each input frame (e.g., `output_0243.png`), the pipeline generates:

- `output_0243_clean_simple_view.png`: Main floor plan visualization
- `output_0243_clean_simple_view_shapes.png`: Visualization showing just the fitted shapes
- `output_0243_clean_simple_view_floor_plan.png`: Simplified gray floor plan
- `output_0243_clean_simple_view_shapes.txt`: Text file with shape data

## Performance

- Processing time depends on image resolution, hardware, and number of workers
- Using `--half_precision` and `--downscale_factor` can significantly improve speed
- For Apple Silicon Macs, MPS acceleration is automatically enabled
- Parallel processing with `--num_workers` is recommended for batch processing

## Requirements

The pipeline uses the same dependencies as the original scripts:
- PyTorch
- Open3D
- NumPy
- OpenCV
- scikit-learn
- matplotlib
- depth_pro (for depth estimation) 