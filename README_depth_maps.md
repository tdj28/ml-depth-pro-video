# Depth Map Generator

This script uses ML-DEPTH-PRO to generate colored depth maps from input images. It's designed to batch process all images in a directory and save the resulting depth maps to an output directory.

## Features

- **Batch Processing**: Process all images in a directory with a single command
- **Colored Visualization**: Generate visually appealing colorized depth maps
- **Hardware Acceleration**: Optimized for CUDA and Apple Silicon (M1/M2/M3) devices
- **Performance Options**: Supports downscaling and half precision for faster processing

## Usage

```bash
python generate_depth_maps.py --input_dir ./TEMP/FRAMES --output_dir ./TMP/DEPTH
```

### Key Arguments

- `--input_dir`: Directory containing input images (default: `./TEMP/FRAMES`)
- `--output_dir`: Directory to save output depth maps (default: `./TMP/DEPTH`)
- `--pattern`: Glob pattern to match input images (default: `*.png`)
- `--downscale_factor`: Shrink input images for faster processing (default: `1.0`)
- `--half_precision`: Use float16 for faster computation (recommended for GPUs)
- `--raw`: Save raw depth maps instead of colored visualizations
- `--colormap`: Choose colormap for visualization (`turbo`, `viridis`, `plasma`, `inferno`, etc.)

### Examples

Process all PNG files in the default directories:
```bash
python generate_depth_maps.py
```

Process specific files with a custom pattern:
```bash
python generate_depth_maps.py --pattern "frame_*.jpg"
```

Faster processing with reduced quality:
```bash
python generate_depth_maps.py --downscale_factor 0.75 --half_precision
```

Generate raw (grayscale) depth maps instead of colored visualizations:
```bash
python generate_depth_maps.py --raw
```

Try different colormaps for visualization:
```bash
python generate_depth_maps.py --colormap plasma
```

## Output

For each input image (e.g., `output_0243.png`), the script generates a corresponding depth map:
- `output_0243_depth.png`: Colorized depth map where brighter/warmer colors represent closer objects

## Performance

- Processing time depends on image resolution, hardware, and options
- Using `--half_precision` is recommended for GPU acceleration
- The `--downscale_factor` option can significantly improve speed (e.g., 0.5 = 4x faster)
- For Apple Silicon Macs, MPS acceleration is automatically enabled

## Requirements

- PyTorch
- OpenCV
- NumPy
- Matplotlib
- depth_pro (for depth estimation) 