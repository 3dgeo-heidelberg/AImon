import sys, os
import numpy as np
from scipy.spatial import ConvexHull
from shapely import Polygon
import cv2
import piexif
import json

from shapely.geometry import mapping, Polygon
import fiona

sys.path.append('../')
from changeDetPipeline.helpers import utils


class ProjectChange:
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
        exif_data = piexif.load(self.bg_img_path)
        try:
            user_comment = exif_data["Exif"].get(piexif.ExifIFD.UserComment, b'')
            metadata_json = user_comment.decode('utf-8')
            image_metadata_loaded = json.loads(metadata_json)
        except:
            print("Missing some information, cannot project change into image")
            pass

        # Get metadata of the image. Necessary for the projection of the change event points
        camera_position = image_metadata_loaded['camera_position']
        h_img_res = image_metadata_loaded['h_img_res']
        v_img_res = image_metadata_loaded['v_img_res']
        h_fov = image_metadata_loaded['h_fov']
        v_fov = image_metadata_loaded['v_fov']
        res = image_metadata_loaded['res']

        # Get change events dictionnary in json file
        with open(self.path_change_events) as json_data:
            change_events = json.load(json_data)

        # Create output folder file if not existant
        output_folder_path = f"output/geojson/{self.project}"
        if not os.path.exists(output_folder_path):
            os.makedirs(f"output/geojson/{self.project}")
        # Name shapefile according to the project name written in the json file
        geojson_name = f"{output_folder_path}/{self.project}_change_events.geojson"
        
        # Create the schema for the attributes of the shapefile
        schema = {
            'geometry': 'Polygon',
            'properties': {
                'event_type': 'str',
                'object_id': 'str',
                'X_centroid': 'float',
                'Y_centroid': 'float',
                'Z_centroid': 'float'
                }
            }
        # Open the shapefile to be able to write each polygon in it
        geojson = fiona.open(geojson_name, 'w', 'GeoJSON', schema, fiona.crs.CRS.from_epsg(4979), 'binary')

        for change_event in change_events:
            if 'undefined' in str(change_event['event_type']): continue

            # Fetch contour points in WGS84 coordinate system
            change_event_pts_og = change_event['points_builing_convex_hulls'][0]
            change_event_pts_og = np.asarray(change_event_pts_og)
            
            # Handle the empty array, if any
            if change_event_pts_og.shape[0] == 0:
                continue

            # Translation of point cloud coordinates for the scanner position of (0, 0, 0)
            change_event_pts = change_event_pts_og - np.asarray(camera_position)

            # Transformation from cartesian coordinates (x, y, z) to spherical coordinates (r, θ, φ)
            r, theta, phi = utils.xyz_2_spherical(change_event_pts)
            theta, phi = np.rad2deg(theta), np.rad2deg(phi)

            # Transformation from spherical coordinates (r, θ, φ) to pixel coordinates (u, v)
            u = np.round((theta - h_fov[0]) / res).astype(int)
            v = np.round((phi - v_fov[0]) / res).astype(int)
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
                    'Z_centroid': float(centroid[2])
                }
            })

        geojson.close()
            

if __name__ == "__main__":
    config_file = r"./config/Obergurgl_2d_projection_config.json"
    config = utils.read_json_file(config_file)
    img = ProjectChange(
        project = config["pc_projection"]["project"],
        bg_img_path = config["change_projection"]["bg_img_path"],
        path_change_events = config["change_projection"]["path_change_events"]
    )

    img.project_change()

