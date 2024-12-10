import os
from pathlib import Path
import argparse
import glob
from helpers.utilities import setup_configuration
from helpers.change_analysis_m3c2 import do_two_sided_bitemporal_m3c2, add_original_points_to_m3c2
from helpers.cluster import cluster
from helpers.change_events import convert_cluster_to_change_events,merge_change_events
from bi_vapc_01 import compute_bitemporal_vapc
from pc_projection_03 import PCloudProjection
from change_projection_04 import ProjectChange
import vapc


def parse_args():
    parser = argparse.ArgumentParser(description="Use processing pipeline.")
    parser.add_argument("config_folder", help="Configuration of the processing pipeline.")
    parser.add_argument("t1_file", help="Path to reference file.")
    parser.add_argument("t2_file", help="Path to target file.")
    return parser.parse_args()

def main() -> None:
    """
    Main function to execute the full workflow.
    """
    
    # Get input 
    args = parse_args()

    # conf_folder = r"./config"

    # t1_file = r"E:\trier\hierarchical_analysis\infiles\trier\ScanPos001 - SINGLESCANS - 240826_000005.las"
    # t2_file = r"E:\trier\hierarchical_analysis\infiles\trier\ScanPos001 - SINGLESCANS - 240826_010006.las"

    # t1_file = r"/home/william/Downloads/ScanPos001 - SINGLESCANS - 240826_000005.las"
    # t2_file = r"/home/william/Downloads/ScanPos001 - SINGLESCANS - 240826_010006.las"
    
    for config_file in glob.glob(f"{args.config_folder}/*.json"):
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
        ) = setup_configuration(config_file, args.t1_file, args.t2_file)

        if configuration["project_setting"]["silent_mode"]:
            vapc.enable_trace(False)
            vapc.enable_timeit(False)
        # #BI-VAPC
        compute_bitemporal_vapc(
            args.t1_file,
            args.t2_file,
            t1_vapc_out_file,
            t2_vapc_out_file,
            configuration
            )
        
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

        #M3C2
        do_two_sided_bitemporal_m3c2(
            t1_vapc_out_file,
            t2_vapc_out_file,
            m3c2_out_file,
            configuration
            )
        
        # TODO: Add function to mask w.r.t. changes
        if configuration["m3c2_settings"]["subsampling"]["voxel_size"] != 0:
            add_original_points_to_m3c2(m3c2_out_file,
                                        t1_vapc_out_file.replace(".laz", "_bk.laz"),
                                        t2_vapc_out_file.replace(".laz", "_bk.laz"),
                                        configuration["m3c2_settings"]["subsampling"]["voxel_size"])


        #Cluster Changes
        cluster(m3c2_out_file, 
                m3c2_clustered,
                configuration
                )
        
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


if __name__ == "__main__":
    main()

