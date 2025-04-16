from functools import wraps
import time
import json
import os
from datetime import datetime
from PySide6.QtCore import QDate
import numpy as np

from itertools import cycle
from shutil import get_terminal_size
from threading import Thread


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


def create_project_structure(config) -> None:
    """
    Generate output folder structure if not existing.
    """
    sub_directories = [
        "01_Change_analysis_UHD_VAPC", 
        "02_Change_analysis_UHD_Change_Events",
        "03_Change_visualisation_UHD_Projected_Images",
        "04_Change_visualisation_UHD_Change_Events",
        "documentation"
    ]
    outdir = config["output_folder"]

    #Create Project Folder
    output_dir = os.path.join(outdir,
                              config["project_name"])
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    #Create Poject Subfolders
    out_folders = {}
    for sub_dir in sub_directories:
        sub_dir_path = os.path.join(output_dir,sub_dir)
        out_folders[sub_dir] = sub_dir_path
        if not os.path.isdir(sub_dir_path):
            os.mkdir(sub_dir_path)
    return config["project_name"], out_folders, config["temporal_format"]

def get_delta_t(t1_file, 
    t2_file, 
    temporal_format='%y%m%d_%H%M%S'):
    """
    Calculate the time difference in seconds between two files based on their filenames.

    Args:
        t1_file (str): Path to the first file.
        t2_file (str): Path to the second file.
        temporal_format (str): The datetime format used in the filenames.

    Returns:
        float: Time difference in seconds.
    """
    # Extract the base filenames
    t1_filename = os.path.basename(t1_file)
    t2_filename = os.path.basename(t2_file)
    
    # Extract the timestamp part from the filenames
    try:
        t1_time_str = t1_filename.split(" ")[-1][:-4]  # Adjust slicing if needed
        t2_time_str = t2_filename.split(" ")[-1][:-4]
        
        # Parse the timestamps into datetime objects
        t1_time = datetime.strptime(t1_time_str, temporal_format)
        t2_time = datetime.strptime(t2_time_str, temporal_format)
    except (IndexError, ValueError) as e:
        print(f"Error parsing filenames: {e}")
        return None
    
    # Calculate the difference in seconds
    delta_seconds = (t2_time - t1_time).total_seconds()
    
    #print(f"Time difference between the two epochs: {delta_seconds} seconds")
    return delta_seconds

def setup_configuration(config_file,t1_file,t2_file, timestamp):
    """
    Sets up the configuration for the change detection pipeline.

    Parameters:
    ----------
        config_file (str): Path to the JSON configuration file.
        t1_file (str): Path to the first timepoint file.
        t2_file (str): Path to the second timepoint file.
        timestamp (str): Timestamp to append to the project name if included.

    Returns:
    ----------
        tuple: A tuple containing:
            - configuration (dict): Loaded configuration settings.
            - t1_out_file (str): Path for the output t1 file.
            - t2_out_file (str): Path for the output t2 file.
            - m3c2_out_file (str): Path for the M3C2 output file.
            - m3c2_clustered (str): Path for the clustered M3C2 output file.
            - change_event_folder (str): Path to the change event folder.
            - change_event_file (str): Path to the change events JSON file.
            - delta_t (float): Time delta between t1 and t2.
            - project_name (str): Name of the project.
            - projected_image_folder (str): Path to the projected images folder.
            - projected_events_folder (str): Path to the projected events folder.
            
    Raises:
    ----------
        AssertionError: If any of the input files do not exist.
    """
    #Check if input is proper
    assert os.path.isfile(t1_file), "This file does not exist: %s"%t1_file
    assert os.path.isfile(t2_file), "This file does not exist: %s"%t2_file
    assert os.path.isfile(config_file), "Configuration file does not exist at: %s"%config_file

    configuration = read_json_file(config_file)
    if not os.path.isdir(configuration["project_setting"]["output_folder"]): os.mkdir(configuration["project_setting"]["output_folder"])
    if configuration["project_setting"]["include_timestamp"]:
        configuration["project_setting"]["project_name"] = configuration["project_setting"]["project_name"] + "_" + timestamp
    project_name, out_folders, temporal_format = create_project_structure(configuration["project_setting"])
    with open(os.path.join(out_folders["documentation"],configuration["project_setting"]["project_name"]+".json"), 'w') as f:
        json.dump(configuration, f,indent=4)

    delta_t = get_delta_t(t1_file,t2_file,temporal_format)

    combination_of_names = os.path.basename(t1_file)[:-4] + "_" + os.path.basename(t2_file)[:-4]
    t1_out_file = os.path.join(out_folders["01_Change_analysis_UHD_VAPC"],combination_of_names+"_t1.laz")
    t2_out_file = os.path.join(out_folders["01_Change_analysis_UHD_VAPC"],combination_of_names+"_t2.laz")
    m3c2_out_file = os.path.join(out_folders["02_Change_analysis_UHD_Change_Events"],combination_of_names+".laz")
    change_event_folder = os.path.join(out_folders["02_Change_analysis_UHD_Change_Events"])
    change_event_file = os.path.join(change_event_folder, "change_events.json")
    m3c2_clustered = os.path.join(change_event_folder,combination_of_names,"clustered.laz")

    projected_image_folder = out_folders["03_Change_visualisation_UHD_Projected_Images"]
    projected_events_folder =out_folders["04_Change_visualisation_UHD_Change_Events"]

    # m3c2_out_file = os.path.join(out_folders["02_Change_analysis_UHD_M3C2"],combination_of_names+".laz")
    # change_event_folder = os.path.join(out_folders["03_Change_analysis_UHD_Change_Events"])
    # m3c2_clustered = os.path.join(change_event_folder,combination_of_names,"clustered.laz")

    return configuration, t1_out_file,t2_out_file,m3c2_out_file,m3c2_clustered,change_event_folder, change_event_file, delta_t, project_name, projected_image_folder, projected_events_folder


############################################################################################################

def date_str2datetime(my_date):
    return datetime.strptime(my_date, "%y%m%d_%H")


def date_str2QDate(my_date):
    return QDate(int(str(20)+my_date[:2]), int(my_date[2:4]), int(my_date[4:6]))


def date_between(my_date, t_min, t_max):
    if my_date[0] <= t_min and t_max <= my_date[1]:
        in_between = True
    else: 
        in_between = False
    return in_between

def get_min_sec(start, end):
    t = (end.second - start.second)
    t_minute = int(t/60)
    t_second = int((t/60 - t_minute)*60)
    return t_minute, t_second

def get_event_color(df_legend_colors, event_type):
    color = df_legend_colors.loc[df_legend_colors["event_type"] == event_type]["color"]
    color = list(color)[0]
    return color


# Converting cartesian coordinates in sperical coordinates
def xyz_2_spherical(xyz):
    dxy = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2)
    r = np.sqrt(dxy**2 + xyz[:, 2]**2)          # radius r
    theta = np.arctan2(dxy, xyz[:, 2])          # theta θ   # for elevation angle defined from Z-axis down
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])      # phi ϕ
    return r, theta, phi                        # [m, rad, rad]


def loc2ref_TRIER():
    change_events = read_json_file("/home/william/Documents/GitHub/changeDetPipeline/data/trier/change_events_split.json")
    for change_event in change_events:
        if 'undefined' in str(change_event['event_type']): continue

        # Fetch contour points in WGS84 coordinate system
        change_event_pts_og = change_event['points_builing_convex_hulls'][0]
        change_event_pts_og = np.asarray(change_event_pts_og)

        mRT = np.array([[0.307283101602, 0.951597433800, 0.006278491405, 330022.325080771232],
            [-0.951613714032, 0.307254846897, 0.005079205073, 5516592.615982715972],
            [0.002904261597, -0.007535452413, 0.999967390579, 134.990167052281],
            [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]]).T
        print(change_event_pts_og.shape)
        o = np.ones(shape=(change_event_pts_og.shape[0], 1))
        change_event_pts_og = np.c_[change_event_pts_og, o]
        
        temp = list(change_event_pts_og@mRT)
        for enum,t in enumerate(temp):
            temp[enum] = list(t[:-1])
        change_event['points_builing_convex_hulls'][0] = temp
        
    with open("/home/william/Documents/GitHub/changeDetPipeline/data/trier/change_events_splitREF.json", 'w') as f:
        json.dump(change_events, f)
    

def rotate_to_top_view(xyz, mean_x, mean_y, mean_z):
        # If the user want a top view, we rotate the point cloud on the side instead of changing the camera view
        rot_angle = np.radians(90)
        rotation_matrix_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(rot_angle), -np.sin(rot_angle)],
                [0, np.sin(rot_angle), np.cos(rot_angle)],
            ]
        )
        rot_angle = np.radians(20)
        rotation_matrix_y = np.array(
            [
                [np.cos(rot_angle), 0, np.sin(rot_angle)],
                [0, 1, 0],
                [-np.sin(rot_angle), 0, np.cos(rot_angle)],
            ]
        )

        rotation_matrix = np.dot(rotation_matrix_y, rotation_matrix_x)
        xyz[:, 0] -= mean_x
        xyz[:, 1] -= mean_y
        xyz[:, 2] -= mean_z
        xyz = np.dot(xyz, rotation_matrix.T)
        xyz[:, 0] += mean_x
        xyz[:, 1] += mean_y
        xyz[:, 2] += mean_z

        return xyz

#loc2ref_TRIER()#


############################################################################################################


class Loader:
    def __init__(self, desc="Loading...", end="Done!", timeout=0.1):
        """
        A loader-like context manager

        Args:
            desc (str, optional): The loader's description. Defaults to "Loading...".
            end (str, optional): Final print. Defaults to "Done!".
            timeout (float, optional): Sleep time between prints. Defaults to 0.1.
        """
        self.desc = desc
        self.end = end
        self.timeout = timeout

        self._thread = Thread(target=self._animate, daemon=True)
        self.steps = ["⢿ ", "⣻ ", "⣽ ", "⣾ ", "⣷ ", "⣯ ", "⣟ ", "⡿ "]
        self.done = False

    def start(self):
        self._thread.start()
        return self

    def _animate(self):
        for c in cycle(self.steps):
            if self.done:
                print("\n")
                break
            print(f"\r{self.desc} {c}", flush=True, end="")
            time.sleep(self.timeout)

    def __enter__(self):
        self.start()

    def stop(self):
        self.done = True
        cols = get_terminal_size((80, 20)).columns
        print("\r" + " " * cols, end="", flush=True)
        print(f"\r{self.end}", flush=True)

    def __exit__(self, exc_type, exc_value, tb):
        # handle exceptions with those variables ^
        self.stop()


def build_pipeline_command(folder_path, config_file, default_cmd, use_every_xth_file = 1):
    """
    Scans the specified folder for las or laz files, orders them alphabetically,
    prints a command string with the default command followed by each las or laz file's full path,
    and returns the list of las or laz filenames.
    
    Parameters:
        folder_path (str): The path to the folder to scan.
        default_cmd (str): The command prefix to use.
        
    Returns:
        list: A sorted list of las or laz filenames found in the folder.
    """
    # List all files in the folder
    files = os.listdir(folder_path)
  
    file_paths = []
    # Build full file paths for each las or laz file
    counter = 0
    for file in files:
        if file.endswith('.last') or file.endswith('.laz'):
            counter += 1
            # Skip files based on the use_every_xth_file parameter
            if counter % use_every_xth_file != 0:
                continue
            full_path = os.path.join(folder_path, file)
            file_paths.append(full_path)
    
    # Create the command string with each file path in quotes
    command_str = default_cmd + " " + "-c \"%s\""%config_file
    command_str = command_str + " -f " + " ".join(f'"{fp}"' for fp in file_paths)
    
    # Print the command string
    print(len(file_paths), "files found")
    print(command_str)