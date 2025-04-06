import pathlib
from sfm_project.pycolmap_sfm.sfm_colmap import run_sfm 
from sfm_project.pycolmap_sfm.utils import extract_frames, process_pointcloud
import argparse
import os

import sys
sys.path.append('gsplat_light')
from internal.entrypoints.gspl import cli as gspl_pipeline
sys.path.append('gsplat_light/utils')
from utils.sd_feature_extraction import main as extract_sd_features
from utils.sd_feature_extraction import parse_args as sd_parse_args

custom_args = [
    "fit",
    "--config", "gsplat_light/configs/spot_less_splats/gsplat-cluster.yaml",
    "--data.parser.split_mode", "reconstruction",
    "--data.path", "output/mvs",
    "-n", "spotless_watchtower",
    "--trainer.max_epochs", "50",
    "--viewer",
]

def sfm_pipeline(video_path, image_dir, output_path, fps):
    


    extract_frames(video_path, image_dir, fps)
    maps = run_sfm(image_dir=image_dir, output_path=output_path, is_dense=False)
    process_pointcloud(output_path, view_pcd=False, save_pcd=True, filter_pcd=True, is_dense=False)


def main():
    parser = argparse.ArgumentParser(description="Run COLMAP Structure-from-Motion pipeline using PyCOLMAP.")
    
    parser.add_argument("--video_path", type=str, help="Path to the input video file.", default="watchtower.mp4")
    parser.add_argument("--image_dir", type=str, help="Directory to save extracted frames.", default="output/extracted_frames")
    parser.add_argument("--output_path", type=str, help="Path to store SfM results.", default="output")
    parser.add_argument("--fps", type=int, help="Frames per second for extraction.", default=1)
    args = parser.parse_args()

    image_dir = pathlib.Path(args.image_dir)
    output_path = pathlib.Path(args.output_path)

    if not os.path.exists(args.output_path):
        sfm_pipeline(args.video_path, image_dir, output_path, fps=args.fps)
    
    sd_args = sd_parse_args()
    sd_args.image_dir = "output/mvs/images"
    sd_args.output = "output/mvs/SD"

    if not os.path.exists(sd_args.output):
        extract_sd_features(args=sd_args)
    gspl_pipeline(args=custom_args)

if __name__ == "__main__":
    main()
