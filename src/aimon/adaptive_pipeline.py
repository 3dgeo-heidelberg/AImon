###########################################################################
# The code currently uses the following settings:
# 1) A fixed interval of 60 minutes for the standard mode
# 2) A fixed interval of 30 minutes for the adaptive mode
# 2.1) In the adaptive mode, the interval is reduced to 30 minutes if a change is detected
# 2.2) High resolution scans are made if a change is detected on a reduced FOV
# 2.3) The FOV is updated based on the extent of the detected change
# 2.4) An overview scan is always made with the old FOV, in adaptive mode after the high resolution scan
# 2.5) The overview scan is used for change detection if no high resolution scan is made
# 3) The first scan is at 12:00, the last scan is at 15:00
###########################################################################

import os
import numpy as np
import argparse

#Survey editing:
from aimon.helpers.lidar_sim import run_lidar_simulation, compute_angles, get_min_and_max_vertical_and_horizontal_angles, update_survey
#Change detection:
from aimon.voxel_wise_change_detection_01 import compute_bitemporal_vapc
from aimon.change_analysis_02 import ChangeAnalysisM3C2
from aimon.helpers.cluster import cluster
from aimon.helpers.change_events import process_m3c2_file_into_change_events
import vapc
#Change projection:
from aimon.pc_projection_03 import PCloudProjection
from aimon.change_projection_04 import ProjectChange
#Mute vapc 
vapc.enable_trace(False)
vapc.enable_timeit(False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Adaptive LiDAR & Change Analysis Pipeline")
    parser.add_argument('--helios_root', required=True, help="Path to the Helios++ root directory")
    parser.add_argument('--output_folder', required=True, help="Path to the output folder")
    parser.add_argument("--adaptive", action="store_true", help="Enable adaptive mode")
    args = parser.parse_args()

    # Set the path to the Helios++ root directory and output folder
    path_to_helios =  args.helios_root
    output_folder = args.output_folder
    adaptive_mode = args.adaptive

    ##########################################   Example usage:   ##########################################
    # python adaptive_pipeline.py --helios_root "D:\helios-plusplus-win" --output_folder "D:\output\test2" --adaptive

    ########################################################################################################

    #### These are the used files for the simulation. These can be changed to fit your needs. ####
    survey_template = os.path.join(path_to_helios, r"data_for_aimon/surveys/template.xml")
    available_scene_files = [
                os.path.join(path_to_helios,r"data_for_aimon/scenes/S0.xml"),
                os.path.join(path_to_helios,r"data_for_aimon/scenes/S1.xml"),
                os.path.join(path_to_helios,r"data_for_aimon/scenes/S2.xml")]
    
    #### This gives the description of the scenes. The first element is the scene number, the second is a description of the scene and the third is the path to the scene file. ####
    # The scene number is used to identify the scenefile in the scenes folder. The description is used for logging purposes.
    # Times are in the format HH_MM. 
    scene_descitption = {
        "12_00":["S0","Initial scene, no change",available_scene_files[0]], #initial scene
        "12_30":["S0","Initial scene, no change",available_scene_files[0]], #No intermediate change
        "13_00":["S1","Scene with change, rock topple",available_scene_files[1]], #small change, lets assume this rock topple occurs at 12:50, so it is first detected at 13:00
        "13_30":["S2","Scene with change, rock fall",available_scene_files[2]], #big change, lets assume this rock fall occurs at 13:10, so it is first detected at 13:30 (adaptive mode) or 14:00 (standard mode)
        "14_00":["S2","Scene with change, rock fall",available_scene_files[2]], #No intermediate change
        "14_30":["S2","Scene with change, rock fall",available_scene_files[2]], #No intermediate change
        "15_00":["S2","Scene with change, rock fall",available_scene_files[2]] #No intermediate change
    }

    # The configuration for the VAPC and M3C2 settings. 
    configuration = {
        "vapc_settings": {
            "vapc_config": {
                "voxel_size": 6,
                "origin": [
                    0,
                    0,
                    0
                ],
                "attributes": {},
                "filter": {
                    "filter_attribute": "point_count",
                    "min_max_eq": "greater_than",
                    "filter_value": 30
                },
                "compute": [],
                "return_at": "center_of_gravity"
            },
            "bi_temporal_vapc_config": {
                "signicance_threshold": 0.999
            },
            "vapc_mask": {
                "buffer_size": 0
            }
        },
        "m3c2_settings": {
            "corepoints": {
                "use": False,
                "point_spacing_m": 1
            },
            "m3c2": {
                "normal_radii": [
                    1,2,3
                ],
                "cyl_radii": 1,
                "max_distance": 10.0,
                "registration_error": 0.025
            }
        },
        "cluster_settings": {
            "cluster_method": "DBSCAN",
            "distance_threshold": 1,
            "cluster_by": [
                "X",
                "Y",
                "Z"
            ],
            "min_cluster_size": 25
        },
        "pc_projection": {
            "epsg": 32632,
            "make_range_image": True,
            "make_color_image": False,
            "top_view": False,
            "save_rot_pc": False,
            "resolution_cm": 15.0,
            "camera_position": [
                26.5,
                -240.015,
                135.411
                ],
            "rgb_light_intensity": 100,
            "range_light_intensity": 15,
        }
    }

    ###### These settings are used for the simulation. These can be changed to fit your needs. ######
    # Set the initial time for the simulation
    initial_time = "12_00"
    # Keep track of the processed scenes
    processed_overview_scenes = []
    processed_detail_scenes = []
    # Extract the hour and minute from the initial time
    h, m = map(int, initial_time.split('_'))
    # Extract the maximum hour and minute from the scene description
    max_h, max_m = map(int, list(scene_descitption.keys())[-1].split('_'))
    # Keep track if change was detected
    change_detected = False 
    # Used for adaptive mode to change the FOV. It is either false or a list of [min_theta, max_theta, min_phi, max_phi] for the new FOV
    change_fov = False

    while True:
        # Abort if the maximum time is exceeded
        if h > max_h or (h == max_h and m > max_m):
            print("Reached last epoch. Exiting loop.")
            break

        ########################### Start code here ###########################
        print("Working on time:", f"{h:02d}_{m:02d}")
        current_time = f"{h:02d}_{m:02d}"
        current_time_out_folder = os.path.join(output_folder, f"{current_time}")
        os.makedirs(current_time_out_folder, exist_ok=True)

        ########################### Start LiDAR Simulation ###########################
        # Check if the current time is in the scene description
        if current_time in scene_descitption:
            # Extract the scene file path
            scene_file = scene_descitption[current_time][2]
            scene_nr = scene_descitption[current_time][0]
            # Run the LiDAR simulation with the specified scene file
            if change_fov: # Make a high resolution scan with the new fov if change is detected
                current_survey = os.path.join(os.path.dirname(survey_template), current_time + "_high_res.xml")
                current_survey_laz = os.path.join(current_time_out_folder, current_time + "_high_res.laz")
                if os.path.exists(current_survey_laz):
                    print(f"Simulation for {current_time} already exists. Skipping...")
                    processed_detail_scenes.append(current_survey_laz)
                else:
                    update_survey(
                        survey_template_path=survey_template,
                        output_path=current_survey,
                        new_scene_name=scene_file,
                        change_fov=change_fov)
                    
                    run_lidar_simulation(current_survey, current_survey_laz, path_to_helios)
                    processed_detail_scenes.append(current_survey_laz)
            #Always run a low resolution scan with the old fov, even if change is detected. This ensures that 
            #we have a low resolution overview scan for the change detection in case we miss something with the selected FOV.
            current_survey = os.path.join(os.path.dirname(survey_template), current_time + "_overview.xml")
            current_survey_laz = os.path.join(current_time_out_folder, current_time + "_overview.laz")
            if os.path.exists(current_survey_laz):
                print(f"Simulation for {current_time} already exists. Skipping...")
                processed_overview_scenes.append(current_survey_laz)
            else:
                update_survey(
                    survey_template_path=survey_template,
                    output_path=current_survey,
                    new_scene_name=scene_file,
                    change_fov=False)
                run_lidar_simulation(current_survey, current_survey_laz, path_to_helios)
                processed_overview_scenes.append(current_survey_laz)
        else:
            print(f"No scene description for {current_time}. Skipping simulation.")

        ########################### Start Hierarchical Change Analysis Pipeline ###########################
        if len(processed_overview_scenes) == 1: # Create shaded range image of the first overview scan
            configuration["pc_projection"]["pc_path"] = current_survey_laz

            pc_prj = PCloudProjection(configuration=configuration,
                                      project_name="%s"%os.path.basename(current_survey_laz)[:-4],
                                      projected_image_folder = os.path.dirname(current_survey_laz))
            pc_prj.project_pc()
        if len(processed_overview_scenes) > 1:
            if m == 0: # Compare overview scans every full hour
                t1_file = processed_overview_scenes[-2]
                t2_file = processed_overview_scenes[-1]
            else: # Compare detail scan to overview scan if change was detected before
                t1_file = processed_overview_scenes[-2]
                t2_file = processed_detail_scenes[-1]

            t1_vapc_out_file = os.path.join(current_time_out_folder, f"{current_time}_t1_vapc.laz")
            t2_vapc_out_file = os.path.join(current_time_out_folder, f"{current_time}_t2_vapc.laz")
            #This is an intemediate solution
            t = "250101_"+"".join(os.path.basename(t1_file).split("_")[:2]) + "00__250101_" + "".join(os.path.basename(t2_file).split("_")[:2])+"00"
            m3c2_out_file = os.path.join(current_time_out_folder, f"{t}_m3c2.laz")
            m3c2_clustered = os.path.join(current_time_out_folder, f"{t}_m3c2_clustered.laz")
            change_event_folder = os.path.join(current_time_out_folder, f"{current_time}_change_events")

            if change_fov:
                #Reduce the extent of pc1 to the extent of pc2
                dh1 = vapc.DataHandler(t1_file)
                dh1.load_las_files()
                #Get the angles of the first point cloud
                phi,theta = compute_angles(np.array(dh1.df.X), np.array(dh1.df.Y), np.array(dh1.df.Z))
                dh1.df = dh1.df[(phi>= change_fov[2]) & (phi <= change_fov[3]) & (theta >= change_fov[0]) & (theta <= change_fov[1])]
                #Save the new file
                dh1.save_as_las(t1_file.replace(".laz", "_cropped.laz"))
                t1_file = t1_file.replace(".laz", "_cropped.laz")

            #BI-VAPC - Change detetction module
            compute_bitemporal_vapc(
                t1_file,
                t2_file,
                t1_vapc_out_file,
                t2_vapc_out_file,
                configuration
                )
            
            #Extract areas that are occupied in both epochs
            # mask_file = os.path.join(os.path.dirname(t1_vapc_out_file), "mask.las")
            # if os.path.exists(mask_file):
            #     

            #M3C2 - Change analysis module
            ChangeAnalysisM3C2.do_two_sided_bitemporal_m3c2(
                t1_vapc_out_file,
                t2_vapc_out_file,
                m3c2_out_file,
                configuration
                )
            
            if os.path.exists(m3c2_out_file):
                #Cluster Changes
                cluster(m3c2_out_file, 
                        m3c2_clustered,
                        configuration
                        )

                if os.path.isfile(m3c2_out_file):
                    # Create change events, a shaded range image, and .geojson files that show the change events projected into the shaded range image and in original coordinates
                    process_m3c2_file_into_change_events(m3c2_clustered)
                    # Project the RBG point cloud to image
                    change_prj = ProjectChange(change_event_file=os.path.dirname(m3c2_clustered)+"change_events.json",
                                               project_name="%s"%os.path.basename(current_survey_laz)[:-4],
                                               projected_image_path=os.path.dirname(current_survey_laz),
                                               projected_events_folder=os.path.dirname(m3c2_out_file),
                                               epsg= configuration["pc_projection"]["epsg"])
                    change_prj.project_change()

                    change_detected = True  # Set to True as changes are detected
                    if m == 30 or adaptive_mode is False: # If we are in standard mode we do not update the FOV
                        change_fov = False
                    else:
                        change_fov = get_min_and_max_vertical_and_horizontal_angles(m3c2_out_file)
                else:
                    change_detected = False
            else:
                print(f"M3C2 output file {m3c2_out_file} does not exist --> No change detected.")
                change_detected = False

        ############################ End code here ###########################

        # Choose step size based on adaptive mode and change detection
        step = 30 if ((adaptive_mode and change_detected) or m == 30) else 60
        m += step
        # Handle overflow of minutes to hours
        if m >= 60:
            h += m // 60
            m = m % 60