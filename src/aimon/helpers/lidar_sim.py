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

    # pyhelios.loggingSilent()
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
                          path_to_helios):
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
                   z = 135.411
                   ):
    # Vector differences from scan position
    z+=1.7 # because of helios platform and scanner height
    dx = xv - x
    dy = yv - y
    dz = zv - z
    # Horizontal angle (azimuth) in degrees [0, 360)
    theta = -90 + np.arctan2(dy, dx) * 180 / np.pi
    # Elevation angle in degrees from horizontal plane
    r = np.sqrt(dx**2 + dy**2)
    phi = np.degrees(np.arctan2(dz, r))
    return phi, theta 


def get_min_and_max_vertical_and_horizontal_angles(infile, scan_pos=[26.5,-240.015,135.411]):
    dh = vapc.DataHandler(infile)
    dh.load_las_files()

    # grab raw arrays
    xs = dh.df['X'].to_numpy()
    ys = dh.df['Y'].to_numpy()
    zs = dh.df['Z'].to_numpy()

    # compute arrays of phi, theta for _every_ point
    phi, theta = compute_angles(xs, ys, zs,
                                scan_pos[0], scan_pos[1], scan_pos[2])

    # true mins & maxs
    min_phi, max_phi = float(phi.min()), float(phi.max())
    min_theta, max_theta = float(theta.min()), float(theta.max())
    d_theta  = (max_theta - min_theta) * 0.2   # 20% vertical buffer
    d_phi    = (max_phi   - min_phi  ) * 0.2   # 20% horizontal buffer

    min_theta -= abs(d_theta) 
    max_theta += abs(d_theta) 
    min_phi   -= abs(d_phi)   
    max_phi   += abs(d_phi)   

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
        vmin, vmax = change_fov[2], change_fov[3]
        leg_scanner.set("verticalAngleMin_deg", str(vmin))
        leg_scanner.set("verticalAngleMax_deg", str(vmax))
        hstart, hstop = change_fov[0], change_fov[1]
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
