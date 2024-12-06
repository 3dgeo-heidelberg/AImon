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
from .data_handler import *
from .vasp import VASP
from multiprocessing import Pool


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
    output_folder_VASP = os.path.join(current_script_dir,
                                "out_data",
                                project_name,
                                "01_Change_analysis_UHD_VASP"
                                )
    output_folder = os.path.join(current_script_dir,
                                "out_data",
                                project_name,
                                "02_Change_analysis_UHD_M3C2"
                                )
    
    # Mask point clouds based on significant change detected within voxels
    outfile_vasp_t0 = os.path.join(output_folder_VASP,"%s_t0.laz"%(os.path.basename(tx_path)[:-4]))
    outfile_vasp_tx = os.path.join(output_folder_VASP,"%s_tx.laz"%(os.path.basename(tx_path)[:-4]))
    outfile_vasp_t0_no_change = os.path.join(output_folder_VASP,"%s_t0.txt"%(os.path.basename(tx_path)[:-4]))
    if not os.path.isfile(outfile_vasp_t0) or os.path.isfile(outfile_vasp_t0_no_change):
        return
    
    outfile_m3c2 = os.path.join(output_folder,"%s_%s.laz"%(os.path.basename(reference_file)[:-4],os.path.basename(tx_path)[:-4]))
    if os.path.isfile(outfile_m3c2):
        print("Result for %s already computed."%outfile_m3c2)
        return
    
    vasp_config = {
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
    for i,masked_pc in enumerate([outfile_vasp_t0,outfile_vasp_tx]):
        data_handler = DATA_HANDLER(masked_pc,
                                    vasp_config["attributes"])
        data_handler.load_las_files()
        vasp = VASP(vasp_config["voxel_size"],
                    vasp_config["origin"],
                    vasp_config["attributes"],
                    vasp_config["compute"],
                    vasp_config["return_at"])
        vasp.get_data_from_data_handler(data_handler)
        xyzs.append(np.array(vasp.df[["X","Y","Z"]]))
        if i == 0:
            vasp.reduce_to_voxels()
            corepoint_vasp = vasp
            corepoints = np.array(corepoint_vasp.df[["X","Y","Z"]])
    #Compute M3C2
    try:
        m3c2_distances, uncertainties = compute_m3c2(xyzs[0],xyzs[1],corepoints,m3c2_config)
        #Filter for change bigger then level of detection 
        significant_m3c2_change = np.abs(m3c2_distances) >= uncertainties["lodetection"]
        corepoint_vasp.df["lodetection"] = uncertainties["lodetection"]
        corepoint_vasp.df["m3c2_distance"] = m3c2_distances
        corepoint_vasp.df = corepoint_vasp.df[significant_m3c2_change]

        dh = DATA_HANDLER("",{})
        dh.df = corepoint_vasp.df
        dh.save_as_las(outfile_m3c2)
    except Exception as error:
        with open(outfile_vasp_t0_no_change,"w") as ef:
            ef.write("%s"%error)


def do_two_sided_bitemporal_m3c2(t1_file_vasp,t2_file_vasp,outfile_m3c2,config):
    m3c2_config = config["m3c2_settings"]["m3c2"]
    corepoint_config = config["m3c2_settings"]["corepoints"]

    
    # Mask point clouds based on significant change detected within voxels
    if not os.path.isfile(t1_file_vasp) or not os.path.isfile(t2_file_vasp):
        return

    outfile_m3c2_no_change = outfile_m3c2[:-4]+".txt"

    if os.path.isfile(outfile_m3c2) or os.path.isfile(outfile_m3c2_no_change):
        print("Result for %s already computed."%outfile_m3c2)
        return
    
    vasp_config = {
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
    for i,masked_pc in enumerate([t1_file_vasp,t2_file_vasp]):
        data_handler = DATA_HANDLER(masked_pc
                                    )
        data_handler.load_las_files()
        vasp = VASP(vasp_config["voxel_size"],
                    vasp_config["origin"],
                    vasp_config["attributes"],
                    vasp_config["compute"],
                    vasp_config["return_at"])
        vasp.get_data_from_data_handler(data_handler)
        xyzs.append(np.array(vasp.df[["X","Y","Z"]]))
        
        vasp.reduce_to_voxels()
        corepoints.append(np.array(vasp.df[["X","Y","Z"]]))
    #Compute M3C2
    try:
        for i,cp in enumerate(corepoints):
            m3c2_distances, uncertainties = compute_m3c2(xyzs[i],xyzs[-i-1],cp,m3c2_config)
            #Filter for change bigger then level of detection 
            rel_change_mask = np.abs(m3c2_distances) >= uncertainties["lodetection"]
            distance = m3c2_distances[rel_change_mask]
            distances.append(distance)
            print(distance)
            lo_detections.append(uncertainties["lodetection"][rel_change_mask])
            ep = np.ones(shape = (distance.shape[0],1))*i
            print(ep)
            epoch_ids.append(ep)
            corepoints[i] = cp[rel_change_mask]
        print("Computed M3C2")
    except Exception as error:
        with open(outfile_m3c2_no_change,"w") as ef:
            ef.write("%s"%error)

    dh = DATA_HANDLER("")
    corepoints = np.vstack(corepoints)
    distances = np.concatenate(distances)
    lo_detections = np.concatenate(lo_detections)
    epoch_ids = np.concatenate(epoch_ids)
    dh.df = pd.DataFrame(np.c_[corepoints,distances,lo_detections,epoch_ids
                        ], columns= ["X","Y","Z","M3C2_distance","M3C2_lodetection","epoch"])
    dh.save_as_las(outfile_m3c2)
    return dh.df