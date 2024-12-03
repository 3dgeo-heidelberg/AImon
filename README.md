# ChangeDetPipeline

## Configuration File Structure
The configuration file `config.json` is used to process and project a point cloud dataset. The configuration is in JSON format and can be customized for specific projects, such as generating range and color images, adjusting the image resolution and camera and lighting settings.

### `pc_projection`

This section defines the parameters for projecting a point cloud to create range and color images, as well as configuring the view and saving options.

- **project**: Name of the project.
- **pc_path**: Path to the input point cloud file (.las / .laz)
- **make_range_image**: Boolean flag indicating whether to generate a range image (`true`/`false`).
- **make_color_image**: Boolean flag indicating whether to generate a color image (`true`/`false`).
- **top_view**: Boolean to enable or disable top-down projection view (`true`/`false`).
- **save_rot_pc**: Boolean flag indicating whether to save the rotated point cloud used for top_view (`true`/`false`).
- **outfolder**: Output folder where the generated images will be saved.
- **resolution_cm**: Resolution of the image in centimeters.
- **camera_position**: Camera position specified by a 3D coordinate `[x, y, z]`.
- **rgb_light_intensity**: Light intensity for color image generation.
- **range_light_intensity**: Light intensity for range image generation.
- **sigma**: Value used to apply Gaussian blur. Set to `0.0` if no blur is needed.

### `change_projection`

This section defines parameters for applying changes or transformations based on pre-defined events.

- **bg_img_path**: Path to the background image used for the projection.
- **path_change_events**: Path to the JSON file containing labeled change events.

## Example Usage

Ensure that the configuration JSON is properly set up with paths and parameters before executing the point cloud processing. To change or customize the parameters, edit the respective values as needed.
Indicate the configuration JSON file when running the `main.py` file as followed:

```console
~/Documents/GitHub/changeDetPipeline$ python main.py "config/Trier_2d_projection_config.json"
```
