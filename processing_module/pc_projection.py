import sys
import os
from datetime import datetime
import laspy
import numpy as np
import cv2
import PIL
from PIL import Image
from PIL.ExifTags import TAGS
import piexif
import json
sys.path.append('../')
from changeDetPipeline.helpers import utils


class PCloudProjection:
    def __init__(
        self,
        project,
        pc_path,
        make_range_image,
        make_color_image,
        top_view,
        save_rot_pc,
        outfolder,
        resolution_cm,
        camera_position,
        rgb_light_intensity,
        range_light_intensity,
        sigma,
        factor_anti_aliasing
    ):
        ##############################
        ### INITIALIZING VARIABLES ###
        self.project = project
        self.pc_path = pc_path  # Point cloud path
        self.make_range_image = make_range_image  # True if range image is wanted, False if not
        self.make_color_image = make_color_image  # True if color image is wanted, False if not
        self.top_view = top_view  # True if the wanted view is from the top, False if the wanted view is from the scanner's perspective
        self.save_rot_pc = save_rot_pc  # If the user wants to save the rotated point cloud for the top view
        self.outfolder = outfolder
        self.resolution_cm = resolution_cm     # Pixel dimension in cm
        self.camera_position = camera_position # Position of the point of view
        self.rgb_light_intensity = rgb_light_intensity # For the colored image: adjust this factor to increase/decrease light intensity
        self.range_light_intensity = range_light_intensity # For the range image: adjust this factor to increase/decrease light intensity
        self.sigma = sigma  # Standard deviation for Gaussian kernel
        self.factor_anti_aliasing = factor_anti_aliasing
        ### INITIALIZING VARIABLES ###
        ##############################

    def project_pc(self):
        self.load_pc_file()
        if self.top_view:
            self.create_top_view()
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
        """
        Saving the image with OpenCV and than saving it with piexif to write metadata

        - type (str) : "Color" or "Range" for the type of image to save
        """
        # Save image with the current time
        if not os.path.exists(self.outfolder):
            os.makedirs(self.outfolder)
        filename = os.path.join(self.outfolder,f"{self.project}_{self.image_type}Image.jpg")
        image_metadata = {
            "image_path": filename,
            "pc_path": self.pc_path,
            "make_range_image": self.make_range_image,
            "make_color_image": self.make_color_image,
            "resolution_cm": self.resolution_cm,
            "top_view": self.top_view,
            "save_rot_pc": self.save_rot_pc,
            "camera_position": self.camera_position,
            "rgb_light_intensity": self.rgb_light_intensity,
            "range_light_intensity": self.range_light_intensity,
            "sigma": self.sigma,
            "h_img_res": self.h_img_res,
            "v_img_res": self.v_img_res,
            "h_fov": self.h_fov,
            "v_fov": self.v_fov,
            "res": self.v_res,
        }

        filename = image_metadata["image_path"]
        # Reversing image array because Opencv reads and writes color channels in BGR instead of RGB
        BGR_img = self.shaded_range_image[..., ::-1]
        cv2.imwrite(filename, BGR_img)

        # Opening image with Pillow
        img = Image.open(image_metadata["image_path"])

        # Preparing metadata
        metadata_json = json.dumps(image_metadata)
        metadata_bytes = metadata_json.encode("utf-8")

        # Load existing EXIF data (if any)
        exif_dict = (
            piexif.load(img.info["exif"]) if "exif" in img.info else {"Exif": {}}
        )

        # Add the custom metadata to the UserComment field in the EXIF data
        exif_dict["Exif"][piexif.ExifIFD.UserComment] = metadata_bytes

        # Convert the EXIF dictionary back to binary EXIF data
        exif_bytes = piexif.dump(exif_dict)

        # Save the image with the new EXIF data
        img.save(image_metadata["image_path"], "jpeg", exif=exif_bytes, quality='keep')

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

    def create_top_view(self):
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
        self.xyz[:, 0] -= self.mean_x
        self.xyz[:, 1] -= self.mean_y
        self.xyz[:, 2] -= self.mean_z
        self.xyz = np.dot(self.xyz, rotation_matrix.T)
        self.xyz[:, 0] += self.mean_x
        self.xyz[:, 1] += self.mean_y
        self.xyz[:, 2] += self.mean_z

        # In case the user indicated to write the rotated point cloud for the top  view
        if self.save_rot_pc:
            self.las_f.x = self.xyz[:, 0]
            self.las_f.y = self.xyz[:, 1]
            self.las_f.z = self.xyz[:, 2]
            self.las_f.write(self.pc_path[:-4] + "_rotated" + self.pc_path[-4:])

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
        r, theta, phi = utils.xyz_2_spherical(self.xyz)  # Outputs r, theta (radians), phi (radians)
        # Convert radians to degrees
        theta_deg = np.rad2deg(theta)
        phi_deg = np.rad2deg(phi)

        # Discretize angles to image coordinates
        self.h_fov = (np.floor(min(theta_deg)), np.ceil(max(theta_deg)))
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
        shaded_color_image = (
            self.color_image.astype(np.float32) * shading[..., np.newaxis]
        )
        shaded_color_image = np.clip(shaded_color_image, 0, 255).astype(np.uint8)
        # If the image is flipped right to left, apply the fliplr() numpy function
        final_image = np.fliplr(shaded_color_image)
        # Apply median filter to selectively remove isolated black pixels
        shaded_color_image = self.remove_isolated_black_pixels(final_image)

        self.shaded_color_image = self.apply_smoothing(shaded_color_image)

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

        self.shaded_range_image = self.apply_smoothing(shaded_range_image)

        # Call save_image function
        self.image_type = "Range"
        
    def apply_smoothing(self, input_image):
        im = Image.fromarray(input_image.astype('uint8'), 'RGB')
        h = int(self.h_img_res/self.factor_anti_aliasing)
        w = int(self.v_img_res/self.factor_anti_aliasing)
        blur = cv2.GaussianBlur(input_image, (3, 3), 0)
        # Flip the image left to right
        output_image = np.fliplr(np.asarray(blur))

        return output_image


if __name__ == "__main__":
    config_file = r"./config/Obergurgl_2d_projection_config.json"
    config = utils.read_json_file(config_file)
    prj = PCloudProjection(
        project=config["pc_projection"]["project"],
        pc_path=config["pc_projection"]["pc_path"],
        make_range_image=config["pc_projection"]["make_range_image"],
        make_color_image=config["pc_projection"]["make_color_image"],
        top_view=config["pc_projection"]["top_view"],
        save_rot_pc=config["pc_projection"]["save_rot_pc"],
        outfolder=config["pc_projection"]["outfolder"],
        resolution_cm = config["pc_projection"]["resolution_cm"],
        camera_position = config["pc_projection"]["camera_position"],
        rgb_light_intensity = config["pc_projection"]["rgb_light_intensity"],
        range_light_intensity = config["pc_projection"]["range_light_intensity"],
        sigma = config["pc_projection"]["sigma"],
        factor_anti_aliasing = config["pc_projection"]["factor_anti_aliasing"]
    )
    prj.project_pc()

