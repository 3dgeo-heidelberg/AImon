import os, sys, io
import numpy as np
from pathlib import Path
import argparse
import glob
from helpers.utilities import setup_configuration, Loader
from helpers.change_analysis_m3c2 import ChangeAnalysisM3C2
from helpers.cluster import cluster
from helpers.change_events import convert_cluster_to_change_events,merge_change_events
from bi_vapc_01 import compute_bitemporal_vapc
from pc_projection_03 import PCloudProjection
from change_projection_04 import ProjectChange
import vapc
import datetime


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments, including:
            - config_folder (str): Configuration of the processing pipeline.
            - filenames (list of str): List of filenames.
    """

    parser = argparse.ArgumentParser(description="Use processing pipeline.")
    parser.add_argument("config_folder", help="Configuration of the processing pipeline.")
    parser.add_argument('filenames', nargs='+', help='List of filenames')
    return parser.parse_args()

def main() -> None:
    """
    Main function to execute the full workflow.
    """

    loader = Loader("Computing... ", "Finished", 0.35).start()

    # In case you debug
    if sys.gettrace() is not None:
        # In case it's Windows (Ronny)
        if os.name == 'nt':
            args_filenames=['', '']
            args_config_folder = ""
        # In case it's Ubuntu (Will)
        else:
            args_filenames=['/home/william/Documents/GitHub/m4dvap/data/ScanPos002 - SINGLESCANS - 241002_155554.laz', '/home/william/Documents/GitHub/m4dvap/data/ScanPos002 - SINGLESCANS - 241002_155654.laz']
            args_config_folder = "/home/william/Documents/GitHub/m4dvap/config"
    # If not in debug mode, parse command-line arguments
    else:
        args = parse_args()
        args_filenames = args.filenames
        args_config_folder = args.config_folder
   
    # Iterate over all pairs of input files and all configuration files
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H-%M-%S")
    for i, t1_file in enumerate(args_filenames[:-1]):
        t2_file = args_filenames[i+1]
        for config_file in glob.glob(f"{args_config_folder}/*.json"):
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
            
            #Optional subsampling for M3C2
            if configuration["m3c2_settings"]["subsampling"]["voxel_size"] != 0:
                for tx_vapc_out_file in [t1_vapc_out_file, t2_vapc_out_file]:
                    if os.path.isfile(tx_vapc_out_file.replace(".laz", "_bk.laz")):
                        continue
                    vapc_command = {
                        "tool":"subsample",
                        "args":{
                            "sub_sample_method":"closest_to_center_of_gravity"
                            }
                        }
                    sspath = vapc.do_vapc_on_files(
                        file=tx_vapc_out_file,
                        out_dir=str(Path(tx_vapc_out_file).parent),
                        voxel_size=configuration["m3c2_settings"]["subsampling"]["voxel_size"],
                        vapc_command=vapc_command,
                        save_as=".laz")
                    os.rename(tx_vapc_out_file,tx_vapc_out_file.replace(".laz", "_bk.laz"))
                    os.rename(sspath,tx_vapc_out_file)

            #M3C2 - Change analysis module
            ChangeAnalysisM3C2.do_two_sided_bitemporal_m3c2(
                t1_vapc_out_file,
                t2_vapc_out_file,
                m3c2_out_file,
                configuration
                )
            
            #Add original points to M3C2 result
            if configuration["m3c2_settings"]["subsampling"]["voxel_size"] != 0:
                if os.path.exists(m3c2_out_file.replace(".laz", "_bk.laz")):
                    pass
                else:
                    os.rename(m3c2_out_file,m3c2_out_file.replace(".laz", "_bk.laz"))
                    
                    ChangeAnalysisM3C2.add_original_points_to_m3c2(m3c2_out_file.replace(".laz", "_bk.laz"),
                                                m3c2_out_file,
                                                t1_vapc_out_file.replace(".laz", "_bk.laz"),
                                                t2_vapc_out_file.replace(".laz", "_bk.laz"),
                                                configuration["m3c2_settings"]["subsampling"]["voxel_size"])


            #Cluster Changes
            if cluster(m3c2_out_file, 
                    m3c2_clustered,
                    configuration
                    ) is False:
                return
                
            
            #Change events of current clusters
            convert_cluster_to_change_events(
                m3c2_clustered,
                configuration
                )

            # #Merge change events to change event collection
            merge_change_events(change_event_folder)    # Outputs the change events JSON file path

            # Project the RBG point cloud to image
            pc_prj = PCloudProjection(configuration, project_name, projected_image_folder)
            pc_prj.project_pc()
            
            # Project the 3D change events point cloud to pixel and UTM 32N coordinates
            change_prj = ProjectChange(change_event_file, project_name,projected_image_folder,projected_events_folder)
            change_prj.project_change()

    loader.stop()
    end = datetime.datetime.now()

    t = (end.second - now.second)
    t_minute = np.floor(t/60)
    t_second = (t/60 - t_minute)*60
    print(f"Executed in {t_minute:02.0f} min {t_second:02.0f} sec")

if __name__ == "__main__":
    main()