###########################################

#               Usage                 #

###########################################

# python change_analysis.py "change_analysis_config.json"

###########################################

import sys
import json
import os
import py4dgeo
from functools import partial
from .utilities import *
from vapc import Vapc, DataHandler
from multiprocessing import Pool
import pandas as pd


def read_json_file(file_path):
    """Read JSON data from a file.

    :param file_path:
        The path to the JSON file.
    :type file_path: str

    :return:
        A dictionary containing the JSON data if successful, None otherwise.
    :rtype: dict or None
    """

    try:
        with open(file_path, 'r') as file:
            json_data = json.load(file)
        return json_data
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return None

@timeit
@trace
def compute_m3c2(reference, target, corepoints, m3c2_config):
    epoch_refernce = py4dgeo.Epoch(reference)
    epoch_target = py4dgeo.Epoch(target)
    epoch_corepoints = py4dgeo.Epoch(corepoints)
    print("epoch_refernce: ",epoch_refernce.cloud.shape)
    print("corepoints: ",corepoints.shape)
    m3c2 = py4dgeo.M3C2(
        epochs=(epoch_refernce, epoch_target),
        corepoints=epoch_corepoints.cloud[::],
        normal_radii=tuple(m3c2_config["normal_radii"]),
        cyl_radii=(m3c2_config["cyl_radii"],),
        max_distance=m3c2_config["max_distance"],
        registration_error=m3c2_config["registration_error"],
    )
    # Run the distance computation
    m3c2_distances, uncertainties = m3c2.run()
    return m3c2_distances, uncertainties

@timeit
@trace
def do_bitemporal_m3c2(reference_file,tx_path,m3c2_config_path,project_name):
    # Read the JSON configuration file
    config_data = read_json_file(m3c2_config_path)

    #  Implement real functionality below
    m3c2_config = config_data.get("m3c2")
    corepoint_config = config_data.get("corepoints")
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # Write change event metadata to output folder '03_Change_analysis_UHD'
    output_folder_Vapc = os.path.join(current_script_dir,
                                "out_data",
                                project_name,
                                "01_Change_analysis_UHD_Vapc"
                                )
    output_folder = os.path.join(current_script_dir,
                                "out_data",
                                project_name,
                                "02_Change_analysis_UHD_M3C2"
                                )
    
    # Mask point clouds based on significant change detected within voxels
    outfile_vapc_t0 = os.path.join(output_folder_Vapc,"%s_t0.laz"%(os.path.basename(tx_path)[:-4]))
    outfile_vapc_tx = os.path.join(output_folder_Vapc,"%s_tx.laz"%(os.path.basename(tx_path)[:-4]))
    outfile_vapc_t0_no_change = os.path.join(output_folder_Vapc,"%s_t0.txt"%(os.path.basename(tx_path)[:-4]))
    if not os.path.isfile(outfile_vapc_t0) or os.path.isfile(outfile_vapc_t0_no_change):
        return
    
    outfile_m3c2 = os.path.join(output_folder,"%s_%s.laz"%(os.path.basename(reference_file)[:-4],os.path.basename(tx_path)[:-4]))
    if os.path.isfile(outfile_m3c2):
        print("Result for %s already computed."%outfile_m3c2)
        return
    
    vapc_config = {
			"voxel_size":corepoint_config["subsample_distance_m"],
			"origin":[0,0,0],
			"attributes":{
				"intensity":"mean"
			},
			"compute":[
			],
			"return_at":"closest_to_center_of_gravity"				
		}
    
    xyzs = []
    for i,masked_pc in enumerate([outfile_vapc_t0,outfile_vapc_tx]):
        data_handler = DataHandler(masked_pc,
                                    vapc_config["attributes"])
        data_handler.load_las_files()
        vapc = Vapc(vapc_config["voxel_size"],
                    vapc_config["origin"],
                    vapc_config["attributes"],
                    vapc_config["compute"],
                    vapc_config["return_at"])
        vapc.get_data_from_data_handler(data_handler)
        xyzs.append(np.array(vapc.df[["X","Y","Z"]]))
        if i == 0:
            vapc.reduce_to_voxels()
            corepoint_vapc = vapc
            corepoints = np.array(corepoint_vapc.df[["X","Y","Z"]])
    #Compute M3C2
    try:
        m3c2_distances, uncertainties = compute_m3c2(xyzs[0],xyzs[1],corepoints,m3c2_config)
        #Filter for change bigger then level of detection 
        significant_m3c2_change = np.abs(m3c2_distances) >= uncertainties["lodetection"]
        corepoint_vapc.df["lodetection"] = uncertainties["lodetection"]
        corepoint_vapc.df["m3c2_distance"] = m3c2_distances
        corepoint_vapc.df = corepoint_vapc.df[significant_m3c2_change]

        dh = DataHandler("",{})
        dh.df = corepoint_vapc.df
        dh.save_as_las(outfile_m3c2)
    except Exception as error:
        with open(outfile_vapc_t0_no_change,"w") as ef:
            ef.write("%s"%error)


def do_two_sided_bitemporal_m3c2(t1_file_vapc,t2_file_vapc,outfile_m3c2,config):

    m3c2_config = config["m3c2_settings"]["m3c2"]
    corepoint_config = config["m3c2_settings"]["corepoints"]

    
    # Mask point clouds based on significant change detected within voxels
    if not os.path.isfile(t1_file_vapc) or not os.path.isfile(t2_file_vapc):
        return

    outfile_m3c2_no_change = outfile_m3c2[:-4]+".txt"

    if os.path.isfile(outfile_m3c2) or os.path.isfile(outfile_m3c2_no_change):
        print("Result for %s already computed."%outfile_m3c2)
        return
    
    vapc_config = {
			"voxel_size":corepoint_config["subsample_distance_m"],
			"origin":[0,0,0],
			"attributes":{
				"intensity":"mean"
			},
			"compute":[
			],
			"return_at":"closest_to_center_of_gravity"				
		}
    
    xyzs = []
    corepoints = []
    distances = []
    lo_detections = []
    epoch_ids = []
    for i,masked_pc in enumerate([t1_file_vapc,t2_file_vapc]):
        data_handler = DataHandler(masked_pc
                                    )
        data_handler.load_las_files()
        vapc = Vapc(vapc_config["voxel_size"],
                    vapc_config["origin"],
                    vapc_config["attributes"],
                    vapc_config["compute"],
                    vapc_config["return_at"])
        vapc.get_data_from_data_handler(data_handler)
        xyzs.append(np.array(vapc.df[["X","Y","Z"]]))
        
        vapc.reduce_to_voxels()
        corepoints.append(np.array(vapc.df[["X","Y","Z"]]))
    #Compute M3C2
    try:
        for i,cp in enumerate(corepoints):
            m3c2_distances, uncertainties = compute_m3c2(xyzs[i],xyzs[-i-1],cp,m3c2_config)
            #Filter for change bigger then level of detection 
            rel_change_mask = np.abs(m3c2_distances) >= uncertainties["lodetection"]
            distance = m3c2_distances[rel_change_mask]
            distances.append(distance)
            lo_detections.append(uncertainties["lodetection"][rel_change_mask])
            ep = np.ones(shape = (distance.shape[0],1))*i
            epoch_ids.append(ep)
            corepoints[i] = cp[rel_change_mask]
        print("Computed M3C2")
    except Exception as error:
        with open(outfile_m3c2_no_change,"w") as ef:
            ef.write("%s"%error)

    dh = DataHandler("")
    corepoints = np.vstack(corepoints)
    distances = np.concatenate(distances)
    lo_detections = np.concatenate(lo_detections)
    epoch_ids = np.concatenate(epoch_ids)
    dh.df = pd.DataFrame(np.c_[corepoints,distances,lo_detections,epoch_ids
                        ], columns= ["X","Y","Z","M3C2_distance","M3C2_lodetection","epoch"])
    dh.save_as_las(outfile_m3c2)
    return dh.df


def add_original_points_to_m3c2(m3c2_out_file_in,m3c2_out_file_points_added,t1_vapc,t2_vapc,voxel_size):
    m3c2 = DataHandler(m3c2_out_file_in)
    m3c2.load_las_files()
    m3c2.df["epoch"] = m3c2.df["epoch"].astype(int)
    vp_m3c2 = Vapc(voxel_size)
    vp_m3c2.get_data_from_data_handler(m3c2)

    t1 = DataHandler(t1_vapc)
    t1.load_las_files()
    t1.df["epoch"] = 0
    vp_t1 = Vapc(voxel_size)
    vp_t1.get_data_from_data_handler(t1)

    t2 = DataHandler(t2_vapc)
    t2.load_las_files()
    t2.df["epoch"] = 1
    vp_t2 = Vapc(voxel_size)
    vp_t2.get_data_from_data_handler(t2)

    vp_t1.select_by_mask(vp_m3c2,mask_attribute = "voxel_index")
    vp_t2.select_by_mask(vp_m3c2,mask_attribute = "voxel_index")

    vp_m3c2.df = pd.concat([vp_m3c2.df,vp_t1.df,vp_t2.df])
    m3c2.df = vp_m3c2.df
    del m3c2.df["voxel_index"]
    del m3c2.df["voxel_x"]
    del m3c2.df["voxel_y"]
    del m3c2.df["voxel_z"]
    m3c2.save_as_las(m3c2_out_file_points_added)
