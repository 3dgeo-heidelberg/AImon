from datetime import datetime
import json
from PySide6.QtCore import QDate
import numpy as np


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
        with open(file_path, "r") as file:
            json_data = json.load(file)
        return json_data
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return None
    

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
