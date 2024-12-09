import os
import argparse
import glob
from helpers.utilities import setup_configuration
from helpers.change_analysis_m3c2 import do_two_sided_bitemporal_m3c2
from helpers.cluster import cluster
from helpers.change_events import convert_cluster_to_change_events,merge_change_events
from bi_vapc_01 import compute_bitemporal_vapc
from pc_projection_03 import PCloudProjection
from change_projection_04 import ProjectChange


def parse_args():
    parser = argparse.ArgumentParser(description="Use processing pipeline.")
    parser.add_argument("config_file", help="Configuration of the processing pipeline.")
    parser.add_argument("t1_file", help="Path to reference file.")
    parser.add_argument("t2_file", help="Path to target file.")
    return parser.parse_args()

def main() -> None:
    """
    Main function to execute the full workflow.
    """
    
    # Get input 
    # args = parse_args()

    conf_folder = r"./config"
    t1_file = r"E:\trier\hierarchical_analysis\infiles\trier\ScanPos001 - SINGLESCANS - 240826_000005.las"
    t2_file = r"E:\trier\hierarchical_analysis\infiles\trier\ScanPos001 - SINGLESCANS - 240826_010006.las"

    # t1_file = r"/home/william/Downloads/ScanPos001 - SINGLESCANS - 240826_000005.las"
    # t2_file = r"/home/william/Downloads/ScanPos001 - SINGLESCANS - 240826_010006.las"
    
    for config_file in glob.glob(f"{conf_folder}/*.json"):
        (
        configuration,
        t1_vapc_out_file,
        t2_vapc_out_file,
        m3c2_out_file,
        m3c2_clustered,
        change_event_folder,
        delta_t,
        project_name,
        projected_image_folder,
        polygons_change_event_folder
        ) = setup_configuration(config_file, t1_file, t2_file)


        # #BI-VAPC
        compute_bitemporal_vapc(
            t1_file,
            t2_file,
            t1_vapc_out_file,
            t2_vapc_out_file,
            configuration
            )

        #M3C2
        do_two_sided_bitemporal_m3c2(
            t1_vapc_out_file,
            t2_vapc_out_file,
            m3c2_out_file,
            configuration
            )
        
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
        change_event_file = merge_change_events(change_event_folder)    # Outputs the change events JSON file path

        # Project the RBG point cloud to image
        pc_prj = PCloudProjection(configuration, project_name, projected_image_folder)
        pc_prj.project_pc()
        
        # Project the 3D change events point cloud to pixel and UTM 32N coordinates
        change_prj = ProjectChange(configuration, change_event_file, project_name)
        change_prj.project_change()


if __name__ == "__main__":
    main()

