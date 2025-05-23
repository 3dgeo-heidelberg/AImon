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

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import rasterio as rio
from rasterio.plot import show
import shapely as shp

from collections import defaultdict

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
            if counter % use_every_xth_file != 0 and counter != 1:
                continue
            full_path = os.path.join(folder_path, file)
            file_paths.append(full_path)
    
    # Create the command string with each file path in quotes
    command_str = default_cmd + " " + "-c \"%s\""%config_file
    command_str = command_str + " -f " + " ".join(f'"{fp}"' for fp in file_paths)
    
    # Print the command string
    print(len(file_paths), "files found")
    print(command_str)
    return command_str


def plot_change_events(vector, raster, event_type_col=None, colors=None):
    fig, ax = plt.subplots(1, figsize=(12, 12))

    with rio.open(raster) as rds:
        show(
            (rds, (3, 2, 1)),  # Read 3 bands of raster in R G B order (3,2,1)
            adjust=True,       # Rescale 0.0 - 1.0
            ax=ax,             # Use existing matplotlib axes
        )

    # Load vector data
    gdf = gpd.read_file(vector)

    # Flip Y-axis of geometry
    gdf['geometry'] = gdf['geometry'].map(lambda polygon: shp.ops.transform(lambda x, y: (x, -y), polygon))

    if event_type_col is not None:
        event_types = gdf[event_type_col].unique()
        if colors is not None:
            # Use provided colormap
            if type(colors) == list:
                if len(event_types) > len(colors):
                    raise ValueError(f"Only {len(colors)} colors defined but {len(event_types)} classes found.")

                color_map = dict(zip(event_types, colors))
            else:
                cmap = cm.get_cmap(colors, len(event_types))
                colors = [mcolors.to_hex(cmap(i)) for i in range(len(event_types))]
                color_map = dict(zip(event_types, colors))
        else:
            cmap = cm.get_cmap('summer', len(event_types))
            colors = [mcolors.to_hex(cmap(i)) for i in range(len(event_types))]
            color_map = dict(zip(event_types, colors))

        gdf['color'] = gdf[event_type_col].map(color_map)

        # Plot colored by class column
        gdf.plot(ax=ax, color=gdf['color'], edgecolor='black', linewidth=0.5, alpha=0.6)

        # Create dynamic legend
        legend_elements = [
            Patch(facecolor=color_map[cls], edgecolor='black', label=str(cls))
            for cls in event_types
        ]
        ax.legend(handles=legend_elements, loc='lower right')
    else:
        gdf.boundary.plot(ax=ax)  # Plot just the boundary

    plt.show()



############################
# For datamodel

class Geometry:
    def __init__(self, type: str, coordinates: list[list[float]]):
        self.type = type
        self.coordinates = np.flip(coordinates[0], axis=1).tolist()  # Reverse the order of coordinates from [X,Y] to [Y,X] to match data model format

class GeoObject:
    def __init__(self, id: str, type: str, dateTime: str, geometry: Geometry, customEntityData: dict[str, str]):
        self.id = id
        self.type = type
        self.dateTime = dateTime
        self.geometry = geometry
        self.customEntityData = customEntityData

class ImageData:
    def __init__(self, url: str, width: int, height: int):
        self.url = url
        self.width = width
        self.height = height

class Observation:
    def __init__(self, startDateTime: str, endDateTime: str, geoObjects: list[GeoObject], backgroundImageData: ImageData = {}):
        self.startDateTime = startDateTime
        self.endDateTime = endDateTime
        self.geoObjects = geoObjects
        self.backgroundImageData = backgroundImageData

class DataModel:
    def __init__(self, observations: list[Observation]):
        self.observations = observations
        
    def toJSON(self):
        return json.dumps(
            self,
            default=lambda o: o.__dict__,
            sort_keys=True,
            indent=4)
    

def convert_geojson_to_datamodel(geojson: dict, bg_img: str=None, width: int=None, height: int=None) -> DataModel:
    # Group features by timestamp
    grouped_features = defaultdict(list)
    if geojson is None:
        print("No change events found")
        return None
    
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        t_min_raw = props.get("t_min")
        t_max_raw = props.get("t_max")
        # Convert t_min and t_max to ISO 8601 format
        t_min = datetime.strptime(t_min_raw, "%y%m%d_%H%M%S").strftime("%Y-%m-%dT%H:%M:%SZ")
        t_max = datetime.strptime(t_max_raw, "%y%m%d_%H%M%S").strftime("%Y-%m-%dT%H:%M:%SZ")
        if t_min and t_max:
            grouped_features[(t_min, t_max)].append(feature)

    observations = []
    for (t_min, t_max), features in grouped_features.items():
        geo_objects = []
        for i, feature in enumerate(features):
            geometry_data = feature.get("geometry", {})
            geometry = Geometry(
                type=geometry_data.get("type", ""),
                coordinates=geometry_data.get("coordinates", [])
            )
            props = feature.get("properties", {})
            geo_object = GeoObject(
                id=props.get("object_id", f"obj_{i}"),
                type=props.get("type", "undefined"),
                dateTime=t_min,
                geometry=geometry,
                customEntityData={k: str(v) for k, v in props.items() if k not in {"object_id", "event_type", "t_min", "t_max", "geometry"}} # Add any properties you want to exclude from customEntityData
            )
            geo_objects.append(geo_object)

        image_data = ImageData(bg_img, width, height)  # Replace with actual image data if available

        observation = Observation(
            startDateTime=t_min,
            endDateTime=t_max,
            geoObjects=geo_objects,
            backgroundImageData=image_data
        )
        observations.append(observation)

    data_model = DataModel(observations=observations)
    datamodel_json = data_model.toJSON()

    return datamodel_json