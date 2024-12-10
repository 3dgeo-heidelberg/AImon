import json 
from vapc import DataHandler
import numpy as np
import os
from scipy.spatial import ConvexHull
import re
from datetime import datetime
import uuid

def extract_time_info(filepath, date_format="%y%m%d_%H%M%S"):
    """
    Extracts t_min, t_max, and delta_t_hours from the filepath.

    Args:
        filepath (str): The file path containing timestamps.
        date_format (str): The format of the timestamps in the filepath.

    Returns:
        dict: Dictionary with extracted time information.
    """
    # Pattern to match all timestamps in the format YYMMDD_HHMMSS
    pattern = r"(\d{6}_\d{6})"
    matches = re.findall(pattern, filepath)
    
    if len(matches) >= 2:
        t_1_str = matches[0]  # Assuming the second timestamp is t_min
        t_2_str = matches[-1]  # Assuming the third timestamp is t_max
        if datetime.strptime(t_1_str, date_format) < datetime.strptime(t_2_str, date_format): 
            t_min = datetime.strptime(t_1_str, date_format)
            t_max = datetime.strptime(t_2_str, date_format)
            t_min_str = t_1_str
            t_max_str = t_2_str
        else:
            t_max = datetime.strptime(t_1_str, date_format)
            t_min = datetime.strptime(t_2_str, date_format)
            t_max_str = t_1_str
            t_min_str = t_2_str
            
        delta_t = round((t_max - t_min).total_seconds() / 3600,3)  # in hours

        return {
            "t_min": [t_min_str],
            "t_max": [t_max_str],
            "delta_t_hours": [delta_t]
        }
    else:
        raise ValueError("Insufficient timestamps found in filepath.")
    

def get_change(points, stat):
    """
    Calculate a statistical measure of the absolute distances from the given points.

    Parameters:
    points (DataFrame): A DataFrame containing a column 'M3C2_distance' with distance values.
    stat (str): The statistical measure to calculate. Options are:
        - "std": Standard deviation of the absolute distances.
        - "mean": Mean of the absolute distances.
        - "min": Minimum of the absolute distances.
        - "max": Maximum of the absolute distances.
        - "median": Median of the absolute distances.
        - "quant90": 90th percentile of the absolute distances.
        - "quant95": 95th percentile of the absolute distances.
        - "quant99": 99th percentile of the absolute distances.

    Returns:
    float: The calculated statistical measure of the absolute distances.

    Raises:
    ValueError: If the provided stat is not one of the recognized options.
    """
    abs_dist = np.abs(points.M3C2_distance)
    if stat == "std":
        return np.nanstd(abs_dist)
    if stat == "mean":
        return np.nanmean(abs_dist)
    if stat == "min":
        return np.nanmin(abs_dist)
    if stat == "max":
        return np.nanmax(abs_dist)
    if stat == "median":
        return np.nanmedian(abs_dist)
    if stat == "quant90":
        return np.nanquantile(abs_dist,.90)
    if stat == "quant95":
        return np.nanquantile(abs_dist,.95)
    if stat == "quant99":
        return np.nanquantile(abs_dist,.99)
    else:
        print("Stat unknown")


def get_conv_hull_points(df):
    """
    Calculate the convex hull points, surface areas, volumes, and surface area to volume ratios 
    for a given DataFrame containing 3D points.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing columns 'X', 'Y', and 'Z' representing 3D points.

    Returns:
    tuple: A tuple containing four lists:
        - simplices_list (list): A list of lists of points that form the vertices of the convex hull.
        - surface_areas (list): A list of surface areas of the convex hulls.
        - volumes (list): A list of volumes of the convex hulls.
        - surface_area_to_volume_ratios (list): A list of surface area to volume ratios of the convex hulls.
    """
    points = df[["X","Y","Z"]].values
    simplices_list = []
    surface_areas = []
    volumes = []
    surface_area_to_volume_ratios = []
    if len(points) < 4:
        simplices_list.append([])
        surface_areas.append(0)
        volumes.append(0)
        surface_area_to_volume_ratios.append(0)
    else:
        hull = ConvexHull(points)
        simplices_list.append([list(pt) for pt in points[hull.vertices]])
        surface_areas.append(hull.area)
        volumes.append(hull.volume)
        surface_area_to_volume_ratios.append(hull.area/hull.volume)
    return simplices_list,surface_areas,volumes,surface_area_to_volume_ratios

def convert_cluster_to_change_events(m3c2_clustered, configuration):
    """
    Converts clustered M3C2 data into change events and saves them as JSON files.
    Args:
        m3c2_clustered (str): Path to the clustered M3C2 file.
        configuration (dict): Configuration dictionary for processing.
    Returns:
        None
    The function performs the following steps:
    1. Creates necessary folders to store change events and point clouds for each cluster.
    2. Loads the clustered M3C2 data.
    3. Iterates through each cluster and computes change event statistics.
    4. Saves the change events as a JSON file.
    5. Saves the point cloud data for each cluster as LAS files.
    6. Deletes the original clustered M3C2 file after processing.
    The change event template includes the following fields:
        - object_id: Identifier for the object.
        - event_type: Type of event (default is "undefined").
        - filepath: Path to the original M3C2 file.
        - start_date: Start date of the event.
        - number_of_points: Number of points in the cluster.
        - t_min: Minimum time value.
        - t_max: Maximum time value.
        - delta_t_hours: Time difference in hours.
        - change_magnitudes_mean: Mean change magnitude.
        - change_magnitudes_median: Median change magnitude.
        - change_magnitudes_std: Standard deviation of change magnitudes.
        - change_magnitudes_min: Minimum change magnitude.
        - change_magnitudes_max: Maximum change magnitude.
        - change_magnitudes_quant90: 90th percentile of change magnitudes.
        - change_magnitudes_quant95: 95th percentile of change magnitudes.
        - change_magnitudes_quant99: 99th percentile of change magnitudes.
        - surface_areas_from_convex_hulls: Surface areas computed from convex hulls.
        - volumes_from_convex_hulls: Volumes computed from convex hulls.
        - surface_area_to_volume_ratios_from_convex_hulls: Surface area to volume ratios from convex hulls.
        - points_builing_convex_hulls: Points building the convex hulls (list of lists).
    """
    change_event_template = {
                "object_id": "object_id",
                "event_type": "undefined",
                "filepath": "path_file",
                "start_date": "YYMMDD_HHMM",
                "number_of_points": "LIST",
                "t_min": "LIST",
                "t_max": "LIST",
                "delta_t_hours": "LIST",
                "change_magnitudes_mean": "LIST",
                "change_magnitudes_median": "LIST",
                "change_magnitudes_std": "LIST",
                "change_magnitudes_min": "LIST",
                "change_magnitudes_max": "LIST",
                "change_magnitudes_quant90": "LIST",
                "change_magnitudes_quant95": "LIST",
                "change_magnitudes_quant99": "LIST",
                "surface_areas_from_convex_hulls": "LIST",
                "volumes_from_convex_hulls": "LIST",
                "surface_area_to_volume_ratios_from_convex_hulls": "LIST",
                "points_builing_convex_hulls": "LIST of LISTS",
            }
    
    #create folder to store change events and point clouds for each cluster
    outfolder = os.path.dirname(m3c2_clustered)
    if not os.path.isdir(outfolder):
        os.makedirs(outfolder)
    pc_folder = os.path.join(outfolder,"point_clouds")
    ce_file = os.path.join(outfolder,"change_events.json")
    if not os.path.isdir(pc_folder):
        os.makedirs(pc_folder) 
    if os.path.isfile(ce_file):
        print("Change events already computed.")
        return
    # Load data
    dh = DataHandler(m3c2_clustered)
    dh.load_las_files()
    df = dh.df
    clusters = np.unique(df["cluster_id"])    
    change_events = []
    for cluster in clusters:
        cluster_df = df[df["cluster_id"] == cluster]
        change_event = change_event_template.copy()
        change_event["object_id"] = str(cluster)
        change_event["filepath"] = m3c2_clustered
        change_event["number_of_points"] = [len(cluster_df)]
        timedict = extract_time_info(m3c2_clustered)
        change_event.update(timedict)
        for stat_for_magnitude in ["mean",
                                   "std",
                                   "min",
                                   "max",
                                   "median",
                                   "quant90",
                                   "quant95",
                                   "quant99"]:
            change_magnitudes = round(get_change(cluster_df,stat_for_magnitude),3)
            change_event["change_magnitudes_%s"%stat_for_magnitude] = [float(change_magnitudes)]
        simplices_list,surface_areas,volumes,surface_area_to_volume_ratios = get_conv_hull_points(cluster_df)
        change_event["points_builing_convex_hulls"] = simplices_list
        change_event["surface_areas_from_convex_hulls"] = surface_areas
        change_event["volumes_from_convex_hulls"] = volumes
        change_event["surface_area_to_volume_ratios_from_convex_hulls"] = surface_area_to_volume_ratios

        change_events.append(change_event)
        # Save point cloud
        dh_pc = DataHandler("")
        dh_pc.df = cluster_df
        dh_pc.save_as_las(os.path.join(pc_folder,"%s.laz"%cluster))
    with open(ce_file, 'w') as f:
        json.dump(change_events, f, indent=4)
    #delete clustered m3c2 file as it is not needed anymore and split to point clouds
    os.remove(m3c2_clustered)



def merge_change_events(change_event_folder):
    """
    Merges change event JSON files from subfolders within the specified folder into a single JSON file.
    This function processes each subfolder within the given `change_event_folder`, reads the 
    "change_events.json" file from each subfolder, and merges the events into a single JSON file 
    named "change_events.json" in the `change_event_folder`. It also keeps track of processed 
    folders to avoid reprocessing them and ensures that each event has a unique `object_id`.
    Args:
        change_event_folder (str): The path to the folder containing subfolders with change event files.
    Returns:
        None
    """
    merged_file = os.path.join(change_event_folder, "change_events.json")
    processed_file = os.path.join(change_event_folder, "processed_folders.json")
    existing_object_ids = set()
    processed_folders = set()

    # Load existing merged change events if the file exists
    if os.path.isfile(merged_file):
        with open(merged_file, 'r') as f:
            change_events_all = json.load(f)
            existing_object_ids = {event["object_id"] for event in change_events_all}
    else:
        change_events_all = []

    # Load processed folders if the file exists
    if os.path.isfile(processed_file):
        with open(processed_file, 'r') as f:
            processed_folders = set(json.load(f))
    
    for folder in os.listdir(change_event_folder):
        folder_path = os.path.join(change_event_folder, folder)
        if not os.path.isdir(folder_path):
            continue

        if folder in processed_folders:
            continue  # Skip already processed folders

        change_event_file = os.path.join(folder_path, "change_events.json")
        if not os.path.isfile(change_event_file):
            continue

        with open(change_event_file, 'r') as f:
            change_events = json.load(f)

        for event in change_events:
            original_id = event.get("object_id")
            if original_id in existing_object_ids:
                # Generate a new unique object_id
                new_id = f"{uuid.uuid4().hex[:8]}"
                event["object_id"] = new_id
            existing_object_ids.add(event["object_id"])
            change_events_all.append(event)
        # Mark folder as processed
        processed_folders.add(folder)

    # Save the merged change events
    with open(merged_file, 'w') as f:
        json.dump(change_events_all, f, indent=4)

    # Save the list of processed folders
    with open(processed_file, 'w') as f:
        json.dump(list(processed_folders), f, indent=4)
    return merged_file
