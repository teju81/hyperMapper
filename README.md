# HyperMapper: Video to 3D Gaussian Splatting Pipeline

Convert videos into high-quality 3D Gaussian Splatting models with this end-to-end pipeline. HyperMapper combines Structure from Motion (SfM), Multi-View Stereo (MVS), and Gaussian Splatting to create stunning 3D reconstructions from video input.

![Demo Reconstruction](assets/demo.gif)
*Example reconstruction from a video input*

## Features

- 🎥 **Video Frame Extraction**: Intelligent frame extraction with configurable FPS
- 📸 **Structure from Motion**: Robust camera pose estimation and sparse reconstruction using COLMAP
- ☁️ **Point Cloud Processing**: Advanced outlier removal and filtering
- 🎯 **Gaussian Splatting**: High-quality 3D reconstruction with real-time rendering
- ⚙️ **Configurable Pipeline**: Easy-to-modify YAML configuration
- 🚀 **GPU Acceleration**: CUDA support for faster processing

## Results

### Sample Reconstructions
![Sample 1](assets/sample1.gif) | ![Sample 2](assets/sample2.gif)
:-------------------------:|:-------------------------:
*Urban Scene Reconstruction* | *Object Reconstruction*

## Installation

### Prerequisites
- Python 3.9+
- CUDA capable GPU (tested on NVIDIA GPUs)
- COLMAP
- PyTorch

### Setup

1. Clone the repository:
```bash
git clone https://github.com/krankit00/hyperMapper.git
cd hyperMapper
```

2. Create and activate a conda environment:
```bash
conda create -n hypermapper python=3.9
conda activate hypermapper
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install COLMAP (if not already installed):
```bash
# Ubuntu/Debian
sudo apt-get install colmap

# Arch Linux
sudo pacman -S colmap

# macOS
brew install colmap
```

## Usage

### Basic Usage

1. Place your video file in the project directory
2. Update the configuration in `config.yaml` if needed
3. Run the pipeline:
```bash
python run.py
```

### Custom Configuration

The pipeline can be customized through the `config.yaml` file:

```yaml
video:
  input_path: "your_video.mp4"
  fps: 2  # Adjust based on video content

pointcloud:
  sor:
    nb_neighbors: 5000
    std_ratio: 0.1
```

Run with custom config:
```bash
python run.py --config your_config.yaml
```

## Pipeline Stages

1. **Video Processing**
   - Frame extraction at specified FPS
   - Automatic frame selection
   ![Frame Extraction](assets/frames.gif)

2. **Structure from Motion**
   - Camera pose estimation
   - Sparse reconstruction
   ![SfM](assets/sfm.gif)

3. **Point Cloud Processing**
   - Statistical outlier removal
   - Point cloud filtering
   ![Point Cloud](assets/pointcloud.gif)

4. **Gaussian Splatting**
   - High-quality 3D reconstruction
   - Real-time rendering
   ![Gaussian Splatting](assets/gsplat.gif)

## Configuration Options

### Video Processing
- Input path
- Frame extraction rate
- Frame quality settings

### Point Cloud Processing
- Outlier removal parameters
- Visualization settings
- Processing flags

### Gaussian Splatting
- Model parameters
- Training settings
- Rendering options

See [Configuration Guide](docs/configuration.md) for detailed options.

## Contributing [TBD]

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- COLMAP for Structure from Motion
- Gaussian Splatting implementation based on [gsplat](https://github.com/graphdeco-inria/gaussian-splatting)
- Point cloud processing utilizing Open3D

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{hyperMapper2024,
  author = {Kumar Ankit},
  title = {HyperMapper: Video to 3D Gaussian Splatting Pipeline},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/krankit00/hyperMapper}
}
```

## Contact

Ankit Kumar - [kr.ankit00@gmail.com](mailto:kr.ankit00@gmail.com)

Project Link: [https://github.com/krankit00/hyperMapper](https://github.com/krankit00/hyperMapper) 