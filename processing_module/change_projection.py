import argparse

import sys, os
import numpy as np
from scipy.spatial import ConvexHull
from shapely import Polygon
import cv2
import piexif
import json
import rasterio
from itertools import islice

from shapely.geometry import mapping, Polygon
import fiona

sys.path.append('../')
from changeDetPipeline.helpers import utils


class ProjectChange:
    """
    Change Projection Module.

    This module processes spatial change events, projects them onto images,
    and generates GeoJSON and kml files for visualization in GIS tools.

    Classes:
        - ProjectChange: Handles loading change data, projecting onto images, 
                        and generating GeoJSON outputs.

    Methods:
        - __init__: Initializes the ProjectChange class with input parameters.
        - project_change: Main function to project changes and create GeoJSON files.
        - project_gis_layer: Helper function to handle GIS layer projection.
    """
    
    def __init__(self, project, bg_img_path, path_change_events):
        ##############################
        ### INITIALIZING VARIABLES ###
        self.project = project
        self.bg_img_path = bg_img_path
        self.path_change_events = path_change_events
        self.img = None
        self.pts = []
        ##############################


    def project_change(self):
        # Load EXIF data from an image
        try:
            # Retrieve the metadata
            with rasterio.open(self.bg_img_path) as src:
                image_metadata_loaded = dict(src.tags().items())
        except:
            print("Missing some information, cannot project change into image")

        # Get metadata of the image. Necessary for the projection of the change event points
        camera_position_x = float(image_metadata_loaded['camera_position_x'])
        camera_position_y = float(image_metadata_loaded['camera_position_y'])
        camera_position_z = float(image_metadata_loaded['camera_position_z'])
        h_img_res = float(image_metadata_loaded['h_img_res'])
        v_img_res = float(image_metadata_loaded['v_img_res'])
        h_fov_x = float(image_metadata_loaded['h_fov_x'])
        h_fov_y = float(image_metadata_loaded['h_fov_y'])
        v_fov_x = float(image_metadata_loaded['v_fov_x'])
        v_fov_y = float(image_metadata_loaded['v_fov_y'])
        res = float(image_metadata_loaded['res'])
        #top_view = bool(image_metadata_loaded['top_view'])

        # Get change events dictionnary in json file
        with open(self.path_change_events) as json_data:
            change_events = json.load(json_data)

        # Create output folder file if not existant
        output_folder_path = f"output/geojson/{self.project}"
        if not os.path.exists(output_folder_path):
            os.makedirs(f"output/geojson/{self.project}")
        # Name geojson according to the project name written in the json file
        geojson_name = f"{output_folder_path}/{self.project}_change_events.geojson"
        geojson_name_gis = f"{output_folder_path}/{self.project}_change_events_gis.geojson"
        
        # Create the schema for the attributes of the geojson
        schema = {
            'geometry': 'Polygon',
            'properties': {
                'event_type': 'str',
                'object_id': 'str',
                'X_centroid': 'float',
                'Y_centroid': 'float',
                'Z_centroid': 'float',
                't_min': 'str',
                't_max': 'str',
                'change_magnitudes_avg': 'float',
                'volumes_from_convex_hulls': 'float'
                }
            }
        # Open the shapefile to be able to write each polygon in it
        geojson = fiona.open(geojson_name, 'w', 'GeoJSON', schema, fiona.crs.CRS.from_epsg(4979), 'binary')
        geojson_gis = fiona.open(geojson_name_gis, 'w', 'GeoJSON', schema, fiona.crs.CRS.from_epsg(25832))

        for change_event in change_events:
            if 'undefined' in str(change_event['event_type']): continue

            # Fetch contour points in WGS84 coordinate system
            change_event_pts_og = change_event['points_builing_convex_hulls'][0]
            change_event_pts_og = np.asarray(change_event_pts_og)
            
            # Handle the empty array, if any
            if change_event_pts_og.shape[0] == 0:
                continue
            
            # GIS layer
            self.project_gis_layer(change_event_pts_og)
            # Add the polygon to the main geojson file
            geojson_gis.write({
                'geometry': mapping(self.polygon_gis),
                'properties': {
                    'event_type': str(change_event['event_type'][0]),
                    'object_id': str(change_event['object_id']),
                    'X_centroid': float(self.centroid_gis[0]),
                    'Y_centroid': float(self.centroid_gis[1]),
                    'Z_centroid': float(self.centroid_gis[2]),
                    't_min': str(change_event['t_min']),
                    't_max': str(change_event['t_min']),
                    'change_magnitudes_avg': float(change_event['change_magnitudes_avg'][0]),
                    'volumes_from_convex_hulls': float(change_event['volumes_from_convex_hulls'][0])
                }
            })

            # Translation of point cloud coordinates for the scanner position of (0, 0, 0)
            change_event_pts = change_event_pts_og - np.asarray([camera_position_x, camera_position_y, camera_position_z])

            # Transformation from cartesian coordinates (x, y, z) to spherical coordinates (r, θ, φ)
            r, theta, phi = utils.xyz_2_spherical(change_event_pts)
            theta, phi = np.rad2deg(theta), np.rad2deg(phi)

            # Transformation from spherical coordinates (r, θ, φ) to pixel coordinates (u, v)
            u = np.round((theta - h_fov_x) / res).astype(int)
            v = np.round((phi - v_fov_x) / res).astype(int)
            change_points_uv = np.c_[u, v]

            # Create the convex hull
            hull = ConvexHull(change_points_uv)

            # Order the points anti-clockwise
            list_points = []
            for simplex in hull.vertices:
                list_points.append([int(v_img_res - change_points_uv[simplex, 1]), -int(change_points_uv[simplex, 0])])
            
            # Create the polygon
            polygon = Polygon(np.array(list_points))

            # Compute centroid
            centroid = np.mean(change_event_pts_og, axis=0)

            # Add the polygon to the main shapefile
            geojson.write({
                'geometry': mapping(polygon),
                'properties': {
                    'event_type': str(change_event['event_type'][0]),
                    'object_id': str(change_event['object_id']),
                    'X_centroid': float(centroid[0]),
                    'Y_centroid': float(centroid[1]),
                    'Z_centroid': float(centroid[2]),
                    't_min': str(change_event['t_min']),
                    't_max': str(change_event['t_min']),
                    'change_magnitudes_avg': float(change_event['change_magnitudes_avg'][0]),
                    'volumes_from_convex_hulls': float(change_event['volumes_from_convex_hulls'][0])
                }
            })
        geojson.close()


    def project_gis_layer(self, change_event_pts_og):
        change_event_pts_xy = change_event_pts_og[:,:2]
        # Create the convex hull
        hull = ConvexHull(change_event_pts_xy)
        # Order the points anti-clockwise
        list_points = []
        for simplex in hull.vertices:
            list_points.append([int(change_event_pts_xy[simplex, 1]), -int(change_event_pts_xy[simplex, 0])])
        
        # Create the polygon
        list_points = np.asarray(list_points)
        list_points.T[[0, 1]] = list_points.T[[1, 0]]
        list_points[:, 0] *= -1
        self.polygon_gis = Polygon(np.array(list_points))
        # Compute centroid
        self.centroid_gis = np.mean(change_event_pts_og, axis=0)

            

if __name__ == "__main__":
    #config_file = r"config/Trier_2d_projection_config.json"
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Project config file containing information for the projection of the point cloud and change events.", type=str)
    args = parser.parse_args()
    config = utils.read_json_file(args.config)

    img = ProjectChange(
        project = config["pc_projection"]["project"],
        bg_img_path = config["change_projection"]["bg_img_path"],
        path_change_events = config["change_projection"]["path_change_events"]
    )

    img.project_change()

