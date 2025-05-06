import os, sys, io
import numpy as np
from pathlib import Path
import argparse
import glob

from aimon.voxel_wise_change_detection_01 import compute_bitemporal_vapc
from aimon.change_analysis_02 import ChangeAnalysisM3C2
from aimon.pc_projection_03 import PCloudProjection
from aimon.change_projection_04 import ProjectChange

from aimon.helpers.utilities import setup_configuration, get_min_sec, Loader
from aimon.helpers.cluster import cluster
from aimon.helpers.change_events import process_m3c2_file_into_change_events
# from bi_vapc_01 import compute_bitemporal_vapc

import vapc
import datetime
import time


def fn_parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments, including:
            - config_folder (str): Configuration of the processing pipeline.
            - filenames (list of str): List of filenames.
    """

    parser = argparse.ArgumentParser(description="Use processing pipeline.")

    # In case you debug
    if sys.gettrace() is not None:
        # In case it's Windows (Ronny)
        if os.name == 'nt':
            args_filenames=[""]
            args_config_file = ""
        # In case it's Ubuntu (Will)
        else:
            args_filenames = ["/home/william/Documents/DATA/Obergurgl/obergurgl_pls/ScanPos007 - SINGLESCANS - 210622_121528.laz", "/home/william/Documents/DATA/Obergurgl/obergurgl_pls/ScanPos007 - SINGLESCANS - 210622_151528.laz"]
            args_config_file = "/home/william/Documents/DATA/Obergurgl/aimon_configs/Obergurgl_dev.json"

        parser.add_argument('-c', '--config_file', default=args_config_file, help='Configuration of the processing pipeline.')
        parser.add_argument('-f', '--filenames', default=args_filenames, nargs='+', help='List of filenames')
    # If not in debug mode, parse command-line arguments
    else:
        parser.add_argument('-c', '--config_file', help="Configuration of the processing pipeline.")
        parser.add_argument('-f', '--filenames', nargs='+', help='List of filenames')

    return parser.parse_args()

def main() -> None:
    """
    Main function to execute the full workflow.
    """

    loader = Loader("Computing... ", "Finished", 0.35).start()
    args = fn_parse_args()
    # Iterate over all pairs of input files and all configuration files
    start = datetime.datetime.now()
    timestamp = start.strftime("%Y_%m_%d_%H-%M-%S")
    for i, t1_file in enumerate(args.filenames[:-1]):
        t2_file = args.filenames[i+1]
        config_file = args.config_file
        (
        configuration,
        t1_vapc_out_file,
        t2_vapc_out_file,
        m3c2_out_file,
        m3c2_clustered,
        change_event_folder,
        change_event_file,
        delta_t,
        project_name,
        projected_image_folder,
        projected_events_folder
        ) = setup_configuration(config_file, t1_file, t2_file,timestamp)
        
        if configuration["project_setting"]["silent_mode"]:
            vapc.enable_trace(False)
            vapc.enable_timeit(False)
            import logging
            logging.disable(logging.CRITICAL)

        #BI-VAPC - Change detetction module
        compute_bitemporal_vapc(
            t1_file,
            t2_file,
            t1_vapc_out_file,
            t2_vapc_out_file,
            configuration
            )

        #M3C2 - Change analysis module
        ChangeAnalysisM3C2.do_two_sided_bitemporal_m3c2(
            t1_vapc_out_file,
            t2_vapc_out_file,
            m3c2_out_file,
            configuration
            )
        
        # #remove output files t1_vapc_out_file t2_vapc_out_file m3c2_out_file
        # os.remove(t1_vapc_out_file)
        # os.remove(t2_vapc_out_file)
        # os.remove(m3c2_out_file)

        # ChangeAnalysisM3C2.add_original_points_to_m3c2(m3c2_out_file.replace(".laz", "_bk.laz"),
        #                             m3c2_out_file,
        #                             t1_vapc_out_file.replace(".laz", "_bk.laz"),
        #                             t2_vapc_out_file.replace(".laz", "_bk.laz"),
        #                             configuration["m3c2_settings"]["subsampling"]["voxel_size"])


        #Cluster Changes
        cluster(m3c2_out_file, 
                m3c2_clustered,
                configuration
                )
        

        
        #Process clustered M3C2 file into change events
        process_m3c2_file_into_change_events(m3c2_clustered)


    # Project the RBG point cloud to image
    pc_prj = PCloudProjection(configuration, project_name, projected_image_folder)
    pc_prj.project_pc()
    
    # Project the 3D change events point cloud to pixel and UTM 32N coordinates
    epsg = int(configuration['pc_projection']['epsg'])
    change_prj = ProjectChange(change_event_file, project_name, projected_image_folder, projected_events_folder, epsg)
    change_prj.project_change()
    loader.stop()


if __name__ == "__main__":
    main()