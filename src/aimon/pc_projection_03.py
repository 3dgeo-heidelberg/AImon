import argparse
import os
import laspy
import numpy as np
import cv2
import rasterio
from aimon.helpers import utilities


class PCloudProjection:
    """
    Point Cloud Projection Module.

    This module processes point clouds, creating 2D projections (color and range images)
    from either a top-down or scanner-based perspective.

    Classes:
        - PCloudProjection: Main class for handling point cloud processing and projection.

    Methods:
        - __init__: Initializes the PCloudProjection class with configuration parameters.
        - project_pc: Main function to execute the projection process.
        - load_pc_file: Loads point cloud data from .las or .laz files.
        - create_top_view: Rotates the point cloud for top-down projection.
        - main_projection: Projects the point cloud into 2D image space.
        - create_shading: Calculates surface normals for image shading.
        - apply_shading_to_color_img: Applies lighting effects to color images.
        - apply_shading_to_range_img: Applies lighting effects to range images.
        - apply_smoothing: Smoothens images using Gaussian blur.
        - save_image: Saves generated images with metadata.
    """

    def __init__(
        self,
        configuration,
        project_name,
        projected_image_folder
    ):
        ##############################
        ### INITIALIZING VARIABLES ###
        self.project_name = project_name
        self.projected_image_folder = projected_image_folder
        self.pc_path = configuration["pc_projection"]["pc_path"]
        self.make_range_image = configuration["pc_projection"]["make_range_image"]
        self.make_color_image = configuration["pc_projection"]["make_color_image"]
        self.top_view = configuration["pc_projection"]["top_view"]
        self.resolution_cm = configuration["pc_projection"]["resolution_cm"]
        self.camera_position = configuration["pc_projection"]["camera_position"]
        self.rgb_light_intensity = configuration["pc_projection"]["rgb_light_intensity"]
        self.range_light_intensity = configuration["pc_projection"]["range_light_intensity"]
        ### INITIALIZING VARIABLES ###
        ##############################


    def project_pc(self):
        self.load_pc_file()
        if self.top_view:
            self.xyz = utilities.rotate_to_top_view(self.xyz, self.mean_x, self.mean_y, self.mean_z)
        self.main_projection()
        self.create_shading()
        if self.make_color_image:
            self.apply_shading_to_color_img()
            self.save_image()
        if self.make_range_image:
            self.apply_shading_to_range_img()
            self.save_image()


    # Define a function to remove isolated black pixels - Only for RGB image
    def remove_isolated_black_pixels(self, image, threshold=np.array([0.0, 0.0, 0.0])):
        """Function to process each pixel neighborhood"""

        # Convert the image in float
        image = image.astype(np.float32)
        # Create a kernel to compute the mean of neighboring pixels (same weight each)
        # And exclude the center pixel
        kernel = np.ones((3, 3), np.float32) / 8.0
        kernel[1, 1] = 0

        # Split the image into its color channels
        channels = cv2.split(image)

        # Apply the convolution to each channel (RGB) separately
        mean_channels = [cv2.filter2D(channel, -1, kernel) for channel in channels]

        # Merge the channels back together
        mean_image = cv2.merge(mean_channels)

        # Identify black pixels (all channels are zero)
        # A pixel is black if each of its RGB value is less than 41
        try:
            black_pixels_mask = np.all(image <= [40, 40, 40], axis=-1)
        except:
            # If the image only has one chanel (black and white image)
            black_pixels_mask = np.any(image <= 40, axis=-1)

        # Replace black pixels with the corresponding values from the mean
        image[black_pixels_mask] = mean_image[black_pixels_mask]

        # Convert image back to integer
        result_image = np.clip(image, 0, 255).astype(np.uint8)

        return result_image


    def save_image(self):
        # Save image with the current time
        if not os.path.exists(self.projected_image_folder):
            os.makedirs(self.projected_image_folder)
        filename = os.path.join(self.projected_image_folder,f"{self.project_name}_{self.image_type}Image.tif")

        raster = np.moveaxis(self.shaded_image, [0, 1, 2], [2, 1, 0])
        raster = np.rot90(raster, k=-1, axes=(1, 2))
        raster = np.flip(raster, axis=2)

        meta = {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'nodata': None,
            'height': self.shaded_image.shape[0],
            'width': self.shaded_image.shape[1],
            'count': 3,  # number of bands
            "tiled": False,
            "compress": 'lzw'
        }
        
        custom_tags = {
                "pc_path": self.pc_path,
                "image_path": filename,
                "make_range_image": self.make_range_image,
                "make_color_image": self.make_color_image,
                "resolution_cm": self.resolution_cm,
                "top_view": self.top_view,
                "camera_position_x": self.camera_position[0],
                "camera_position_y": self.camera_position[1],
                "camera_position_z": self.camera_position[2],
                "pc_mean_x": self.mean_x,
                "pc_mean_y": self.mean_y,
                "pc_mean_z": self.mean_z,
                "rgb_light_intensity": self.rgb_light_intensity,
                "range_light_intensity": self.range_light_intensity,
                "h_img_res": self.h_img_res,
                "v_img_res": self.v_img_res,
                "h_fov_x": self.h_fov[0],
                "h_fov_y": self.h_fov[1],
                "v_fov_x": self.v_fov[0],
                "v_fov_y": self.v_fov[1],
                "res": self.v_res
            }

        # Write the raster
        with rasterio.open(filename, "w", **meta) as dest:
            dest.write(raster, [1,2,3])
            dest.update_tags(**custom_tags)


    def load_pc_file(self):
        # Load the .las/.laz file
        with laspy.open(self.pc_path) as las_file:
            self.las_f = las_file.read()
        x = np.array(self.las_f.x)
        y = np.array(self.las_f.y)
        z = np.array(self.las_f.z)
        if self.make_color_image:
            self.red = np.array(self.las_f.red)
            self.green = np.array(self.las_f.green)
            self.blue = np.array(self.las_f.blue)

            # Normalize RGB values if necessary (assuming they are in the range 0-65535)
            if self.red.max() > 255:
                self.red = (self.red / 65535.0 * 255).astype(np.uint8)
                self.green = (self.green / 65535.0 * 255).astype(np.uint8)
                self.blue = (self.blue / 65535.0 * 255).astype(np.uint8)

        self.xyz = np.vstack((x, y, z)).T

        # Computing xyz coord means
        self.mean_x = np.mean(x)
        self.mean_y = np.mean(y)
        self.mean_z = np.mean(z)


    def main_projection(self):
        # Shift the point cloud by the camera position' coordinates so the latter is positionned on the origin
        self.xyz -= self.camera_position
        # Range between camera and the mean point of the point cloud
        range = np.sqrt(
            (
                (self.camera_position[0] - self.mean_x) ** 2
                + (self.camera_position[1] - self.mean_y) ** 2
                + (self.camera_position[2] - self.mean_z) ** 2
            )
        )
        # Getting vertical and horizontal resolutions in degrees. Both calculated with the range and the pixel dimension
        alpha_rad = np.arctan2(self.resolution_cm / 100, range)
        self.v_res = self.h_res = np.rad2deg(alpha_rad)

        # Get spherical coordinates
        r, theta, phi = utilities.xyz_2_spherical(self.xyz)  # Outputs r, theta (radians), phi (radians)
        # Convert radians to degrees
        theta_deg, phi_deg = np.rad2deg(theta), np.rad2deg(phi)

        # Discretize angles to image coordinates
        if np.floor(min(theta_deg)) == -180 or np.floor(max(theta_deg)) == 180:
            mask = theta_deg < 0
            theta_deg[mask] += 360
        
        self.h_fov = (np.floor(min(theta_deg)), np.ceil(max(theta_deg)))


        if np.floor(min(phi_deg)) == -180 or np.floor(max(phi_deg)) == 180:
            mask = phi_deg < 0
            phi_deg[mask] += 360
        
        self.v_fov = (np.floor(min(phi_deg)), np.ceil(max(phi_deg)))

        self.h_img_res = int((self.h_fov[1] - self.h_fov[0]) / self.h_res)
        self.v_img_res = int((self.v_fov[1] - self.v_fov[0]) / self.v_res)

        # Initialize range and color image
        self.range_image = np.full(
            (self.h_img_res, self.v_img_res, 3), 0, dtype=np.float32
        )
        self.color_image = np.full(
            (self.h_img_res, self.v_img_res, 3), 0, dtype=np.uint8
        )

        # Map angles to pixel indices
        u = np.round((theta_deg - self.h_fov[0]) / self.h_res).astype(int)
        v = np.round((phi_deg - self.v_fov[0]) / self.v_res).astype(int)

        # Filter points within range
        valid_indices = (
            (u >= 0) & (u < self.h_img_res) & (v >= 0) & (v < self.v_img_res)
        )
        self.u = u[valid_indices]
        self.v = v[valid_indices]
        self.r = r[valid_indices]
        self.r = (self.r-np.min(self.r))*255/np.max(self.r-np.min(self.r))
        if self.make_color_image:
            self.red = self.red[valid_indices]
            self.green = self.green[valid_indices]
            self.blue = self.blue[valid_indices]

        # Shift the point cloud back to its original coordinates
        self.xyz += self.camera_position


    def create_shading(self):
        # Compute surface normals' components (gradient approximation)
        z_img = np.zeros((self.h_img_res, self.v_img_res))
        #self.r = self.r * 255 / np.max(self.r)
        z_img[self.u, self.v] = self.r
        dz_dv, dz_du = np.gradient(z_img)

        # Compute normals with components
        self.normals = np.dstack((-dz_du, -dz_dv, np.ones_like(z_img)))
        self.norms = np.linalg.norm(self.normals, axis=2, keepdims=True)
        self.normals /= self.norms  # Normalize


    def apply_shading_to_color_img(self):
        # Populate
        self.color_image[self.u, self.v, 0] = self.red
        self.color_image[self.u, self.v, 1] = self.green
        self.color_image[self.u, self.v, 2] = self.blue
        # Compute shading (Lambertian model)
        # Light direction for the image to have the right shading
        light_dir_x = abs(self.camera_position[0] - self.mean_x)
        light_dir_y = abs(self.camera_position[1] - self.mean_y)
        light_dir_z = abs(self.camera_position[2] - self.mean_z)
        light_direction = np.array(
            [light_dir_x, light_dir_y, light_dir_z]
        )  # Direction of the light source
        light_direction = light_direction / np.linalg.norm(light_direction)  # Normalize

        dot_product = np.sum(self.norms * light_direction, axis=2)
        shading = np.clip(dot_product * self.rgb_light_intensity, 0, 1)

        # Apply smoothed shading to the color image
        shaded_color_image = (self.color_image.astype(np.float32) * shading[..., np.newaxis])
        shaded_color_image = np.clip(shaded_color_image, 0, 255).astype(np.uint8)
        # Apply median filter to selectively remove isolated black pixels
        shaded_color_image = self.remove_isolated_black_pixels(shaded_color_image)

        self.shaded_image = self.apply_smoothing(shaded_color_image)

        # Call save_image function
        self.image_type = "Color"


    def apply_shading_to_range_img(self):
        # Populate the range image with the radius (scanner to point distance)
        self.range_image[self.u, self.v, 0] = \
            self.range_image[self.u, self.v, 1] = \
            self.range_image[self.u, self.v, 2] = \
        self.r+ self.range_light_intensity
        
        # Shade the range image with the normals
        shaded_range_image = (
            self.range_image.astype(np.float32)
            * (
                self.normals[:, :, -1] + self.normals[:, :, -2] + self.normals[:, :, -3]
            )[..., np.newaxis]
        )

        filter_255 = shaded_range_image>255.
        shaded_range_image[filter_255] = 255.

        filter_0 = shaded_range_image<0.
        shaded_range_image[filter_0] = 0.

        self.shaded_image = self.apply_smoothing(shaded_range_image)

        # Call save_image function
        self.image_type = "Range"

        
    def apply_smoothing(self, input_image):
        blur = cv2.GaussianBlur(input_image, (3, 3), 0)
        # Flip the image left to right
        output_image = np.fliplr(np.asarray(blur))

        return output_image


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("config", help="Project config file containing information for the projection of the point cloud and change events.", type=str)
    # args = parser.parse_args()
    # config = utilities.read_json_file(args.config)

    config_file = r"/home/william/Documents/DATA/TRIER/project_settings_trier.json"
    config = utilities.read_json_file(config_file)

    prj = PCloudProjection(
        configuration=config,
        project_name=config["project_setting"]["project_name"],
        projected_image_folder=os.path.join(config["project_setting"]["output_folder"], "Trier_vs6_av0_999/03_Change_visualisation_UHD_Projected_Images")
    )
    prj.project_pc()