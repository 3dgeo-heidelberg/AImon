###########################################

#               Usage                 #

###########################################

# python change_analysis.py "change_analysis_config.json"

###########################################

import os
import numpy as np
import py4dgeo
from aimon.helpers.utilities import read_json_file
from vapc import Vapc, DataHandler, timeit, trace
import pandas as pd


class ChangeAnalysisM3C2:
    @timeit
    @trace
    def compute_m3c2(reference, target, corepoints, m3c2_config):
        """
        Compute M3C2 distances and uncertainties between reference and target point clouds.

        Parameters:
        ----------
        reference (str or numpy.ndarray): Path to the reference point cloud file or the point cloud data as a numpy array.
        target (str or numpy.ndarray): Path to the target point cloud file or the point cloud data as a numpy array.
        corepoints (str or numpy.ndarray): Path to the corepoints file or the corepoints data as a numpy array.
        m3c2_config (dict): Configuration dictionary for M3C2 parameters, containing:
            - "normal_radii" (list or tuple): Radii for normal computation.
            - "cyl_radii" (float): Radius for the cylindrical neighborhood.
            - "max_distance" (float): Maximum distance for M3C2 computation.
            - "registration_error" (float): Registration error for uncertainty computation.

        Returns:
        ----------
        tuple: A tuple containing:
            - m3c2_distances (numpy.ndarray): Computed M3C2 distances.
            - uncertainties (numpy.ndarray): Computed uncertainties.
        """
        epoch_refernce = py4dgeo.Epoch(reference)
        epoch_target = py4dgeo.Epoch(target)
        if corepoints is False:
            epoch_corepoints = reference
        else:
            epoch_corepoints = py4dgeo.Epoch(corepoints)
        m3c2 = py4dgeo.M3C2(
            epochs=(epoch_refernce, epoch_target),
            corepoints=epoch_corepoints.cloud[::],
            normal_radii=tuple(m3c2_config["normal_radii"]),
            cyl_radius=(m3c2_config["cyl_radii"],),
            max_distance=m3c2_config["max_distance"],
            registration_error=m3c2_config["registration_error"],
        )
        # Run the distance computation
        m3c2_distances, uncertainties = m3c2.run()
        normal_directions = m3c2.directions()
        normal_radii = m3c2.directions_radii()
        return m3c2_distances, uncertainties, normal_directions, normal_radii

    @timeit
    @trace
    def do_bitemporal_m3c2(reference_file,tx_path,m3c2_config_path,project_name):
        """
        Perform bitemporal M3C2 change analysis between two point clouds.
        Args:
        ----------
            reference_file (str): Path to the reference point cloud file.
            tx_path (str): Path to the target point cloud file.
            m3c2_config_path (str): Path to the JSON configuration file for M3C2 analysis.
            project_name (str): Name of the project for organizing output data.

        Returns:
        ----------
            None

        This function reads the configuration file, processes the point clouds, and computes the M3C2 distances.
        The results are saved in the specified output folder. If the results already exist, the function will
        print a message and return without reprocessing.
        """
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
            #print("Result for %s already computed."%outfile_m3c2)
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
            m3c2_distances, uncertainties = ChangeAnalysisM3C2.compute_m3c2(xyzs[0],xyzs[1],corepoints,m3c2_config)
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


    @timeit
    @trace
    def do_two_sided_bitemporal_m3c2(t1_file_vapc, t2_file_vapc, outfile_m3c2, config):
        """
        Perform two-sided bitemporal M3C2 change detection analysis on two point clouds.

        Parameters:
        ----------
            t1_file_vapc (str): File path to the first voxelized point cloud (VAPC) file.
            t2_file_vapc (str): File path to the second voxelized point cloud (VAPC) file.
            outfile_m3c2 (str): File path to save the M3C2 results.
            config (dict): Configuration dictionary containing M3C2 and corepoint settings.

        Returns:
        ----------
            pd.DataFrame: DataFrame containing the corepoints, M3C2 distances, level of detection, and epoch IDs.

        Notes:
        ----------
            - The function checks if the input files exist and if the output files already exist.
            - It loads the point clouds, reduces them to voxels, and computes M3C2 distances.
            - Results are saved in LAS format if significant changes are detected.
            - If an error occurs during computation, a no-change file is created with the error message.
        """
        outfile_m3c2_no_change = outfile_m3c2[:-4]+".txt"

        if os.path.isfile(outfile_m3c2):
            #print("Result for %s already computed."%outfile_m3c2)
            return
        if os.path.isfile(outfile_m3c2_no_change):
            #print("Result for %s already computed."%outfile_m3c2)
            return
        
        m3c2_config = config["m3c2_settings"]["m3c2"]
        corepoint_config = config["m3c2_settings"]["corepoints"]

        
        # Mask point clouds based on significant change detected within voxels
        if not os.path.isfile(t1_file_vapc) or not os.path.isfile(t2_file_vapc):
            return


        if os.path.isfile(outfile_m3c2) or os.path.isfile(outfile_m3c2_no_change):
            #print("Result for %s already computed."%outfile_m3c2)
            return
        
        vapc_config = {
                "voxel_size":corepoint_config["point_spacing_m"],
                "return_at":"closest_to_center_of_gravity"				
            }
        
        xyzs = []
        corepoints = []
        distances = []
        lo_detections = []
        nx = []
        ny = []
        nz = []
        n_radius = []
        spread1 = []
        spread2 =[]
        num_samples1 = []
        num_samples2 = []
        epoch_ids = []

        for i,masked_pc in enumerate([t1_file_vapc,t2_file_vapc]):
            # Load xyz data
            data_handler = DataHandler(masked_pc
                                        )
            data_handler.load_las_files()
            vapc = Vapc(voxel_size = vapc_config["voxel_size"],
                        return_at= vapc_config["return_at"])
            vapc.get_data_from_data_handler(data_handler)
            xyzs.append(np.array(vapc.df[["X","Y","Z"]]))
            
            # Optionally get corepoints for M3C2 by reducing to 
            # closest to center of gravity per voxels
            if corepoint_config["use"]:
                vapc.reduce_to_voxels()
                corepoints.append(np.array(vapc.df[["X","Y","Z"]]))
            else:
                corepoints.append(np.array(vapc.df[["X","Y","Z"]]))
        # Compute M3C2
        try:
            # Compute M3C2 for both epochs in both directions.
            for i,cp in enumerate(corepoints):
                m3c2_distances, uncertainties,normal_directions, normal_radii = ChangeAnalysisM3C2.compute_m3c2(xyzs[i],xyzs[-i-1],cp,m3c2_config)
                #Filter for change bigger then level of detection 
                rel_change_mask = np.abs(m3c2_distances) >= uncertainties["lodetection"]
                # Add points with significant change to the output
                distances.append(m3c2_distances[rel_change_mask])
                lo_detections.append(uncertainties["lodetection"][rel_change_mask])
                spread1.append(uncertainties["spread1"][rel_change_mask])
                spread2.append(uncertainties["spread2"][rel_change_mask])
                num_samples1.append(uncertainties["num_samples1"][rel_change_mask])
                num_samples2.append(uncertainties["num_samples2"][rel_change_mask])
                nx.append(normal_directions.T[0][rel_change_mask])
                ny.append(normal_directions.T[1][rel_change_mask])
                nz.append(normal_directions.T[2][rel_change_mask])
                n_radius.append(normal_radii[rel_change_mask])
                ep = np.ones(shape = (m3c2_distances[rel_change_mask].shape[0],1))*i
                epoch_ids.append(ep)
                corepoints[i] = cp[rel_change_mask]
        except Exception as error:
            with open(outfile_m3c2_no_change,"w") as ef:
                ef.write("%s"%error)
            return
        # Save the results
        dh = DataHandler("")
        corepoints = np.vstack(corepoints)
        # Distances
        distances = np.concatenate(distances)
        # Level of detection
        lo_detections = np.concatenate(lo_detections)
        spread1 = np.concatenate(spread1)
        spread2 = np.concatenate(spread2)
        num_samples1 = np.concatenate(num_samples1)
        num_samples2 = np.concatenate(num_samples2)
        # Normals
        nx = np.concatenate(nx)
        ny = np.concatenate(ny)
        nz = np.concatenate(nz)
        n_radius = np.concatenate(n_radius)

        epoch_ids = np.concatenate(epoch_ids)
        dh.df = pd.DataFrame(np.c_[corepoints,distances,lo_detections,epoch_ids,nx,ny,nz,n_radius,spread1,spread2,num_samples1,num_samples2
                            ], columns= ["X","Y","Z","M3C2_distance","M3C2_lodetection","epoch","nx","ny","nz","n_radius","spread1","spread2","num_samples1","num_samples2"])
        try:
            dh.save_as_las(outfile_m3c2)
        except Exception as error:
            with open(outfile_m3c2_no_change,"w") as ef:
                ef.write("%s"%error)
            return
        return dh.df

    @timeit
    @trace
    def add_original_points_to_m3c2(m3c2_out_file_in, m3c2_out_file_points_added, t1_vapc, t2_vapc, voxel_size):
        """
        Adds original points from two epochs to the M3C2 output file.

        Parameters:
        ----------
            m3c2_out_file_in (str): Path to the input M3C2 output file.
            m3c2_out_file_points_added (str): Path to save the M3C2 output file with added points.
            t1_vapc (str): Path to the VAPC file for the first epoch.
            t2_vapc (str): Path to the VAPC file for the second epoch.
            voxel_size (float): Size of the voxel to be used.

        Returns:
        ----------
            None
        """
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
