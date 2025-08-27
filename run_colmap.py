
import pathlib
from sfm_project.pycolmap_sfm.sfm_colmap import run_sfm 
from sfm_project.pycolmap_sfm.utils import extract_frames, process_pointcloud
import argparse
import os
import yaml
import sys

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def sfm_pipeline(config):
    """Run the SfM pipeline with the given configuration"""
    video_input = config['input']['video_input']
    extract = config['input']['extract'] and False

    # Extract frames from video
    if video_input:
        video_input_path = config['input']['video_input_path']
        extracted_image_dir = config['input']['extracted_image_dir']
        input_image_dir = extracted_image_dir
        fps = config['input']['fps']
        num_frames = config['input']['num_frames']
        if extract:
            extract_frames(video_input_path, extracted_image_dir, fps, num_frames)
    else:
        input_image_dir = pathlib.Path(config['colmap_directories']['image_dir'])

    sfm_dir = pathlib.Path(config['colmap_directories']['sfm_dir'])

    # Run Structure from Motion
    maps = run_sfm(image_dir=input_image_dir, output_path=sfm_dir, is_dense=config['flags']['is_dense'])
    
    # Process point cloud
    process_pointcloud(
        sfm_dir, 
        view_pcd=config['flags']['view_pointcloud'],
        save_pcd=config['flags']['save_pointcloud'],
        filter_pcd=config['flags']['filter_pointcloud'],
        is_dense=config['flags']['is_dense']
    )

def main():
    parser = argparse.ArgumentParser(description="Run COLMAP Structure-from-Motion pipeline using PyCOLMAP.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Run SfM pipeline
    sfm_pipeline(config)

if __name__ == "__main__":
    main()
