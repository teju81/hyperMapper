import pathlib
from sfm_project.pycolmap_sfm.sfm_colmap import run_sfm 
from sfm_project.pycolmap_sfm.utils import extract_frames, process_pointcloud
import argparse
# from gsplat_light.internal.entrypoints.gspl import cli as gspl_pipeline


def sfm_pipeline():
    parser = argparse.ArgumentParser(description="Run COLMAP Structure-from-Motion pipeline using PyCOLMAP.")
    
    parser.add_argument("--video_path", type=str, help="Path to the input video file.", default="watchtower.mp4")
    parser.add_argument("--image_dir", type=str, help="Directory to save extracted frames.", default="extracted_frames")
    parser.add_argument("--output_path", type=str, help="Path to store SfM results.", default="sfm_output")
    parser.add_argument("--fps", type=int, help="Frames per second for extraction.", default=1)

    args = parser.parse_args()

    image_dir = pathlib.Path(args.image_dir)
    output_path = pathlib.Path(args.output_path)
    extract_frames(args.video_path, image_dir, fps=args.fps)
    maps = run_sfm(image_dir=image_dir, output_path=output_path, is_dense=True)
    process_pointcloud(output_path, view_pcd=True, save_pcd=True, filter_pcd=True, is_dense=True)


def main():
    sfm_pipeline()
    # gspl_pipeline()

if __name__ == "__main__":
    main()
