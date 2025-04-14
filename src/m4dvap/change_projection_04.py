import argparse
import os
import numpy as np
from scipy.spatial import ConvexHull
from shapely import Polygon
import rasterio
from shapely.geometry import mapping, Polygon
import fiona
from helpers import utilities
from helpers.change_events import ChangeEventCollection
import json

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

    def __init__(self, change_event_file, project_name, projected_image_folder, projected_events_folder):
        ##############################
        ### INITIALIZING VARIABLES ###
        self.project = project_name
        self.bg_img_folder = projected_image_folder
        self.path_change_events = change_event_file
        self.img = None
        self.pts = []
        self.geojson_name = os.path.join(projected_events_folder,"%s_change_events_pixel.geojson"%self.project)
        self.geojson_name_gis = os.path.join(projected_events_folder,"%s_change_events_gis.geojson"%self.project)
        ##############################
        if not os.path.isdir(self.bg_img_folder):
            print("Missing some information, cannot find %s"%self.bg_img_folder)
            return 
        else:
            self.bg_img_path = os.path.join(self.bg_img_folder, os.listdir(self.bg_img_folder)[0])


    def project_change(self):
        # Load EXIF data from an image
        try:
            # Retrieve the metadata
            #self.bg_img_path = os.path.join(os.getcwd(), self.bg_img_path)
            with rasterio.open(self.bg_img_path) as src:
                image_metadata_loaded = dict(src.tags().items())
        except:
            print("Missing some information, cannot project change into image")
            return

        # Get metadata of the image. Necessary for the projection of the change event points
        pc_mean_x = float(image_metadata_loaded['pc_mean_x'])
        pc_mean_y = float(image_metadata_loaded['pc_mean_y'])
        pc_mean_z = float(image_metadata_loaded['pc_mean_z'])
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
        top_view = json.loads(image_metadata_loaded['top_view'].lower()) # Using json.loads() method to convert the string "True"/"False" to a boolean
        
        # Get change events dictionnary in json file
        # change_events = utilities.read_json_file(self.path_change_events)
        change_events = ChangeEventCollection()
        change_events = change_events.load_from_file(self.path_change_events)
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
                'change_magnitudes_mean': 'float',
                'volumes_from_convex_hulls': 'float',
                'cluster_point_cloud': 'str',
                'cluster_point_cloud_chull': 'str'
                }
            }
        # Open the shapefile to be able to write each polygon in it
        geojson = fiona.open(self.geojson_name, 'w', 'GeoJSON', schema, fiona.crs.CRS.from_epsg(4979), 'binary')
        geojson_gis = fiona.open(self.geojson_name_gis, 'w', 'GeoJSON', schema, fiona.crs.CRS.from_epsg(25832))
        print(change_events.events)
        for change_event in change_events.events:
            
            #if 'undefined' in str(change_event['event_type']): continue

            # Fetch contour points in WGS84 coordinate system
            change_event_pts_og = change_event.convex_hull["points_building"]
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
                    'event_type': str(change_event.event_type),
                    'object_id': str(change_event.object_id),
                    'X_centroid': float(self.centroid_gis[0]),
                    'Y_centroid': float(self.centroid_gis[1]),
                    'Z_centroid': float(self.centroid_gis[2]),
                    't_min': str(change_event.t_min),
                    't_max': str(change_event.t_max),
                    'change_magnitudes_mean': float(change_event.change_magnitudes["mean"]),
                    'volumes_from_convex_hulls': float(change_event.convex_hull["volume"]),
                    'cluster_point_cloud': str(change_event.cluster_point_cloud),
                    'cluster_point_cloud_chull': str(change_event.cluster_point_cloud_chull)
                }
            })

            
            # If top_view is True, rotate the change events the same way the point cloud was rotated to make the top view
            if top_view:
                change_event_pts = utilities.rotate_to_top_view(change_event_pts_og, 
                                                                pc_mean_x,
                                                                pc_mean_y,
                                                                pc_mean_z
                                                                )
            else:
                change_event_pts = change_event_pts_og.copy()
            
            # Translation of point cloud coordinates for the scanner position of (0, 0, 0)
            change_event_pts = change_event_pts - np.asarray([camera_position_x, camera_position_y, camera_position_z])

            # Transformation from cartesian coordinates (x, y, z) to spherical coordinates (r, θ, φ)
            r, theta, phi = utilities.xyz_2_spherical(change_event_pts)
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
                    'event_type': str(change_event.event_type),
                    'object_id': str(change_event.object_id),
                    'X_centroid': float(centroid[0]),
                    'Y_centroid': float(centroid[1]),
                    'Z_centroid': float(centroid[2]),
                    't_min': str(change_event.t_min),
                    't_max': str(change_event.t_max),
                    'change_magnitudes_mean': float(change_event.change_magnitudes["mean"]),
                    'volumes_from_convex_hulls': float(change_event.convex_hull["volume"]),
                    'cluster_point_cloud': str(change_event.cluster_point_cloud),
                    'cluster_point_cloud_chull': str(change_event.cluster_point_cloud_chull)
                }
            })
        geojson.close()
        geojson_gis.close()

        #self.geojson2kml()


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


    # TODO: make it work
    def geojson2kml(self):
        self.kml_name_gis = self.geojson_name_gis.replace('.geojson', ".kml")
        self.kml_name_gis = f"{os.path.abspath('.')}/{self.kml_name_gis}"
        geojson_data = utilities.read_json_file(self.geojson_name_gis)

        from xml.etree.ElementTree import Element, SubElement, tostring, ElementTree
        import xml.dom.minidom
        #############################
        
        # Helper function to create XML elements
        def create_simple_field(parent, name, field_type):
            simple_field = SubElement(parent, "SimpleField", name=name, type=field_type)
            return simple_field

        # Initialize KML structure
        kml = Element('kml', xmlns="http://www.opengis.net/kml/2.2")
        document = SubElement(kml, 'Document', id="root_doc")

        # Schema definition
        schema = SubElement(document, 'Schema', name="trier_change_events_gis", id="trier_change_events_gis")
        fields = [
            ("event_type", "string"),
            ("object_id", "string"),
            ("X_centroid", "float"),
            ("Y_centroid", "float"),
            ("Z_centroid", "float")
        ]
        for name, field_type in fields:
            create_simple_field(schema, name, field_type)

        # Folder for Placemarks
        folder = SubElement(document, 'Folder')
        folder_name = SubElement(folder, 'name')
        folder_name.text = "trier_change_events_gis"

        # Generate Placemarks from GeoJSON
        for feature in geojson_data.get("features", []):
            properties = feature.get("properties", {})
            geometry = feature.get("geometry", {})
            
            # Extract values from properties
            event_type = properties.get("event_type", "Unknown")
            object_id = properties.get("object_id", "Unknown")
            x_centroid = str(properties.get("X_centroid", 0))
            y_centroid = str(properties.get("Y_centroid", 0))
            z_centroid = str(properties.get("Z_centroid", 0))
            
            # Extract coordinates
            coordinates = ""
            if geometry.get("type") == "Polygon":
                for ring in geometry.get("coordinates", []):
                    coordinates += " ".join(f"{lon},{lat}" for lon, lat in ring) + " "
            
            # Create Placemark
            placemark = SubElement(folder, 'Placemark')
            style = SubElement(placemark, 'Style')
            line_style = SubElement(style, 'LineStyle')
            line_width = SubElement(line_style, 'width')
            line_width.text = "4"
            line_color = SubElement(line_style, 'color')
            line_color.text = "ff0000ff"
            poly_style = SubElement(style, 'PolyStyle')
            fill = SubElement(poly_style, 'fill')
            fill.text = "0"
            
            extended_data = SubElement(placemark, 'ExtendedData')
            schema_data = SubElement(extended_data, 'SchemaData', schemaUrl="#trier_change_events_gis")
            
            # Add properties to SchemaData
            for name, value in [
                ("event_type", event_type),
                ("object_id", object_id),
                ("X_centroid", x_centroid),
                ("Y_centroid", y_centroid),
                ("Z_centroid", z_centroid)
            ]:
                simple_data = SubElement(schema_data, 'SimpleData', name=name)
                simple_data.text = value

            # Add Polygon geometry
            if coordinates:
                polygon = SubElement(placemark, 'Polygon')
                outer_boundary = SubElement(polygon, 'outerBoundaryIs')
                linear_ring = SubElement(outer_boundary, 'LinearRing')
                coord_element = SubElement(linear_ring, 'coordinates')
                coord_element.text = coordinates.strip()
        
        ########################
        
        # Beautify the output XML
        kml_str = xml.dom.minidom.parseString(tostring(kml)).toprettyxml(indent="  ")
        with open(self.kml_name_gis, 'w') as file:
            file.write(kml_str)

        #############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Project config file containing information for the projection of the point cloud and change events.", type=str)
    args = parser.parse_args()
    config = utilities.read_json_file(args.config)

    """config_file = r"config/Trier_2d_projection_config.json"
    config = utilities.read_json_file(config_file)"""

    img = ProjectChange(
        project = config["pc_projection"]["project"],
        bg_img_path = config["change_projection"]["bg_img_path"],
        path_change_events = config["change_projection"]["path_change_events"]
    )

    img.project_change()

