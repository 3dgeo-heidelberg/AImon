import vapc
import pyhelios
from pyhelios import outputToNumpy
import vapc
import pandas as pd
import os
import xml.etree.ElementTree as ET
import numpy as np

def initiate_survey(survey_path):
    survey_path = survey_path
    assets_dir = os.path.join(os.curdir, 'assets')
    output_dir = os.path.join(os.curdir, 'output')
    print(assets_dir)
    print(output_dir)

    simBuilder = pyhelios.SimulationBuilder(
        surveyPath=survey_path,
        assetsDir=assets_dir,
        outputDir=output_dir)
    
    simBuilder.setNumThreads(0)
    # simBuilder.setLasOutput(True)
    # simBuilder.setZipOutput(True)
    simBuilder.setFinalOutput(True)
    simBuilder.setExportToFile(False)  # Disable export point cloud to file
    # build the survey
    simB = simBuilder.build()
    simB.start()
    if simB.isStarted():
        print("Simulation is started!")
    while True:
        if simB.isFinished():
            print("\nSimulation has finished.")
            break
    return simB

def helios_sim_to_df(simBuilder):
    measurement_points, trajectory_points = outputToNumpy(simBuilder.join())
    cols_points = ("X","Y","Z","ORI_X","ORI_Y","ORI_Z","DIR_X","DIR_Y","DIR_Z","intensity","echoWidth","NumberOfReturns","ReturnNumber","FullwaveIndex","hitObjectId","classification","gpsTime")
    cols_trajectory = ("X","Y","Z","gpsTime","roll","pitch","yaw")
    df_points = pd.DataFrame(measurement_points, columns=cols_points)
    df_trajectory = pd.DataFrame(trajectory_points, columns=cols_trajectory)
    return df_points,df_trajectory

def save_df_to_laz(df, laz_file):
    # Save the DataFrame to a .laz file using vapc
    dh = vapc.DataHandler("")
    dh.df = df
    dh.save_as_las(laz_file)

def run_lidar_simulation(path_to_survey, 
                          laz_file,
                          path_to_helios = r"D:\helios-plusplus-win"):
    os.chdir(path_to_helios)
    # Initiate the survey and run the simulation
    simB = initiate_survey(path_to_survey)
    # Convert the simulation output to DataFrames
    df, _ = helios_sim_to_df(simB)
    # Save the DataFrame to a .laz file
    save_df_to_laz(df, laz_file)
    return laz_file


def compute_angles(xv,yv,zv,
                   x = 26.5, 
                   y = -240.015, 
                   z = 135.411,
                   buffer= 0 #m
                   ):
    # Calculate the angles for each point in the DataFrame
    phi = -90 + np.arctan2(yv+buffer - y, xv+buffer - x) * 180 / np.pi
    theta = np.arctan2(zv+buffer - z, np.sqrt((xv+buffer - x) ** 2 + (yv+buffer - y) ** 2)) * 180 / np.pi
    return phi, theta


def get_min_and_max_vertical_and_horizontal_angles(infile,scan_pos= [26.5,-240.015,135.411]):
    dh = vapc.DataHandler(infile)
    dh.load_las_files()

    phi_1, theta_1 = compute_angles(dh.df.X.min(), dh.df.Y.min(), dh.df.Z.min(), 
                                        scan_pos[0], scan_pos[1], scan_pos[2])
    phi_2, theta_2 = compute_angles(dh.df.X.max(), dh.df.X.max(), dh.df.Z.max(),
                                        scan_pos[0], scan_pos[1], scan_pos[2])

    min_phi = min(phi_1, phi_2)
    max_phi = max(phi_1, phi_2)
    min_theta = min(theta_1, theta_2)
    max_theta = max(theta_1, theta_2)

    delta_phi = max_phi - min_phi
    delta_theta = max_theta - min_theta

    min_phi = min_phi - delta_phi * 0.25
    max_phi = max_phi + delta_phi * 0.25
    min_theta = min_theta - delta_theta * 0.25
    max_theta = max_theta + delta_theta * 0.25

    print(f"min_phi: {min_phi}, min_theta: {min_theta}")    
    print(f"max_phi: {max_phi}, max_theta: {max_theta}")
    return min_theta, max_theta, min_phi, max_phi

def update_survey(
    survey_template_path,
    output_path,
    new_scene_name,
    change_fov,
    old_scene="data/scenes/aimon/S0.xml#t0"):
    # 1) Parse the input XML
    tree = ET.parse(survey_template_path)
    root = tree.getroot()

    # 2) Find the element whose TEXT is exactly "S0.xml"
    found = False
    for elem in root.iter():
        scene_attr = elem.get("scene")
        if scene_attr == old_scene:
            new_val = new_scene_name + "#t0"
            elem.set("scene", new_val)
            found = True
            break

    if not found:
        raise ValueError(f'No element found with scene="{old_scene}"')

    leg_scanner = root.find(".//leg/scannerSettings")
    if leg_scanner is None:
        raise ValueError("Could not find leg/scannerSettings element")
    # 3) Update the scanner settings
    # Set the vertical and horizontal angles
    if change_fov:
        vmin, vmax = change_fov[0], change_fov[1]
        leg_scanner.set("verticalAngleMin_deg", str(vmin))
        leg_scanner.set("verticalAngleMax_deg", str(vmax))
        hstart, hstop = change_fov[2], change_fov[3]
        leg_scanner.set("headRotateStart_deg", str(hstart))
        leg_scanner.set("headRotateStop_deg", str(hstop))

        top_scanner = root.find(".//scannerSettings[@id='profile1']")
        if top_scanner is None:
            raise ValueError('Could not find top-level scannerSettings with id="profile1"')
        vres, hres = 0.015/2, 0.015/2  # Example values for vertical and horizontal resolution
        top_scanner.set("verticalResolution_deg", str(vres))
        top_scanner.set("horizontalResolution_deg", str(hres))

    # 4) Write the modified XML to a new file
    tree.write(
        output_path,
        encoding="utf-8",
        xml_declaration=True
    )
