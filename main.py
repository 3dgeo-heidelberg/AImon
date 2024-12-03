from helpers import utils
from processing_module import pc_projection, change_projection
import argparse

"""
Main script for executing point cloud projection and change visualization.

This script initializes the required classes using a JSON configuration file
and executes the projection of point clouds and change detection visualization.

Modules Used:
    - pc_projection: For handling point cloud projection.
    - change_projection: For projecting change events in image coordinates.

Example:
    python main.py "config/Trier_2d_projection_config.json"
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Project config file containing information for the projection of the point cloud and change events.", type=str)
    args = parser.parse_args()
    config = utils.read_json_file(args.config)
    
    pc_prj = pc_projection.PCloudProjection(
        project=config["pc_projection"]["project"],
        pc_path=config["pc_projection"]["pc_path"],
        make_range_image=config["pc_projection"]["make_range_image"],
        make_color_image=config["pc_projection"]["make_color_image"],
        top_view=config["pc_projection"]["top_view"],
        save_rot_pc=config["pc_projection"]["save_rot_pc"],
        outfolder=config["pc_projection"]["outfolder"],
        resolution_cm = config["pc_projection"]["resolution_cm"],
        camera_position = config["pc_projection"]["camera_position"],
        rgb_light_intensity = config["pc_projection"]["rgb_light_intensity"],
        range_light_intensity = config["pc_projection"]["range_light_intensity"],
        sigma = config["pc_projection"]["sigma"]
    )
    pc_prj.project_pc()
    change_prj = change_projection.ProjectChange(
        project = config["pc_projection"]["project"],
        bg_img_path = config["change_projection"]["bg_img_path"],
        path_change_events = config["change_projection"]["path_change_events"]
        )
    change_prj.project_change()