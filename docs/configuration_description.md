# Configuration Description

This document details each configuration parameter and its role in the processing pipeline for point cloud change detection. The pipeline integrates voxelization (VAPC), change analysis via M3C2 (py4dgeo), clustering, and projections for visualization. The settings described below are used across various modules to ensure reproducibility.

*For each parameter, the allowed formats are indicated. When a parameter is fixed to a set of options, those acceptable values are listed.*

In summary, the configuration parameters described bellow establish a framework for point cloud analysis:

- <a href="#project_setting">**Global project settings**</a> define naming, logging, and file output behaviors.
- <a href="#vapc_settings">**VAPC settings**</a> determine voxelization parameters and computational procedures for extracting meaningful geometric properties.
- <a href="#m3c2_settings">**M3C2 settings**</a> include corepoint generation, normal computation, and registration errors to facilitate precise change quantification.
- <a href="#cluster_settings">**Clustering parameters**</a> aggregate significant changes into spatially coherent groups.
- <a href="#pc_projection">**Point cloud projection parameters**</a> ensure that processed data is accurately visualized and georeferenced.

Collectively, these settings enable a workflow for detecting, quantifying, and visualizing changes in 3D time series datasets.


<details>
<summary>Example of a configuration file</summary>
```json
{
    "project_setting": {
        "project_name": "Trier_vs6_av0_999",
        "output_folder": "./test_data/out",
        "temporal_format": "%y%m%d_%H%M%S",
        "silent_mode": true,
        "include_timestamp": false
    },
    "vapc_settings": {
        "vapc_config": {
            "voxel_size": 6,
            "origin": [
                0, 0, 0
            ],
            "attributes": {},
            "filter": {
                "filter_attribute": "point_count",
                "min_max_eq": "greater_than",
                "filter_value": 30
            },
            "compute": [],
            "return_at": "center_of_gravity"
        },
        "bi_temporal_vapc_config": {
            "signicance_threshold": 0.999
        },
        "vapc_mask": {
            "buffer_size": 0
        }
    },
    "m3c2_settings": {
        "corepoints": {
            "use": false,
            "point_spacing_m": 1
        },
        "m3c2": {
            "normal_radii": [
                1, 2, 3
            ],
            "cyl_radii": 1,
            "max_distance": 10.0,
            "registration_error": 0.02
        }
    },
    "cluster_settings": {
        "cluster_method": "DBSCAN",
        "distance_threshold": 1,
        "cluster_by": [
            "X", "Y", "Z"
        ],
        "min_cluster_size": 100
    },
    "pc_projection": {
        "pc_path": "./test_data/240826_000005.laz",
        "make_range_image": true,
        "make_color_image": false,
        "create_kml": true,
        "top_view": false,
        "resolution_cm": 15.0,
        "camera_position": [
            330599.6068, 5515785.9140, 135.4113
        ],
        "rgb_light_intensity": 100,
        "range_light_intensity": 15,
        "epsg": 32632
    }
}
```
</details>

---

## <span style="color:#c92434">**project_setting**</span>

- **project_name**:  
  This specifies the project identifier (e.g. `Trier_vs6_av0_999`).
  *Allowed format*: Any non-empty string; typically alphanumeric with underscores or hyphens.  
  All outputs are stored in a subdirectory with the same name as the project within the designated output folder.

- **output_folder**:  
  The base directory where all output files are stored (e.g., `./test_data/out`).  
  *Allowed format*: A valid relative or absolute file path as a string.

- **temporal_format**:  
  A date-time formatting string (e.g., `%y%m%d_%H%M%S`) used for parsing timestamps from filenames and for generating consistent output file names.  
  *Allowed format*: Formats compatible with Python’s `strftime` and `strptime`; only valid format specifiers (e.g., `%Y`, `%m`, `%d`, `%H`, `%M`, `%S`) are accepted.

- **silent_mode**:  
  If set to `true`, the process minimizes console output.  
  *Allowed format*: Boolean (`true` or `false`).

- **include_timestamp**:  
  When enabled, appends timestamps to output folder. If disabled, previous outputs may be overwritten.  
  *Allowed format*: Boolean (`true` or `false`).


## <span style="color:#c92434">**vapc_settings**</span>

<h3>vapc_config</h3>

- **voxel_size**:  
  Defines the size of voxels used to subdivide the point cloud. Smaller voxels result in higher spatial resolution at increased computational cost, while larger voxels aggregate more points.  
  *Allowed format*: A positive float.

- **origin**:  
  A three-dimensional coordinate `[x, y, z]` that establishes the reference point for the voxel grid. Defaults to `[0, 0, 0]`.  
  *Allowed format*: A list or array of three numeric values (floats or integers).

- **attributes**:  
  A dictionary serving as a placeholder for supplemental parameters (e.g., intensity averaging) required during preprocessing.  
  *Allowed format*: A JSON/dictionary with string keys and corresponding values.

- **filter**:
    - **filter_attribute**:
    The DataFrame column used to filter voxels. For example, `"point_count"` restricts processing to voxels that meet a specified point density.  
    *Allowed format*: A string representing a valid column name.
    - **min_max_eq**:  
    The comparison operator (e.g., `"greater_than"`, `"=="`, `"<"`, etc.) that quantifies how the attribute is compared against a threshold.  
    *Allowed options*: `"equal_to"`, `"=="`, `"not_equal_to"`, `"!="`, `"greater_than"`, `">"`, `"greater_than_or_equal_to"`, `">="`, `"less_than"`, `"<"`, `"less_than_or_equal_to"`, `"<="`.
    - **filter_value**:  
    Numeric threshold (e.g., 30) applied to the corresponding attribute to select voxels for further analysis.  
    *Allowed format*: An integer or float, matching the data type of the attribute field.

- **compute**:  
  A list of computational operations to be executed for each voxel.  
  *Allowed options*: A list of strings from a fixed set such as `"voxel_index"`, `"point_count"`, `"point_density"`, `"percentage_occupied"`, `"covariance_matrix"`, `"eigenvalues"`, `"geometric_features"`, `"center_of_gravity"`, `"distance_to_center_of_gravity"`, `"std_of_cog"`, `"closest_to_center_of_gravity"`, `"center_of_voxel"`, `"corner_of_voxel"`.

- **return_at**:  
  Indicates which computed attribute should represent the voxel’s location. For instance, using `"center_of_gravity"` selects the voxel centroid as the representative coordinate.  
  *Allowed options*: A string value chosen from the attributes computed (e.g., `"center_of_gravity"`, `"closest_to_center_of_gravity"`, etc).

<h3>bi_temporal_vapc_config</h3>

- **signicance_threshold**:  
  A statistical threshold (e.g., 0.999) used during bi-temporal comparisons. This value is often used alongside measures such as the Mahalanobis distance to determine whether changes observed between epochs are statistically significant.  
  *Allowed format*: A float between 0 and 1.

<h3>vapc_mask</h3>

- **buffer_size**:  
  Specifies an optional buffer zone (in voxel units) around the computed grid to accommodate edge effects. A zero value indicates no additional padding.  
  *Allowed format*: A non-negative integer.

*Additional Context:*  
The VAPC module performs voxelization and computes metrics (e.g., covariance matrices, eigenvalues) essential for detecting subtle geometric variations. These voxel properties underpin change detection procedures implemented in the change analysis modules.

---

## <span style="color:#c92434">**m3c2_settings**</span>

<h3>corepoints</h3>

- **use**:  
  Boolean flag indicating whether to generate a reduced subset of core points from the voxelized data. If set to `false`, analysis utilizes the full or pre-voxelized point set.  
  *Allowed format*: Boolean (`true` or `false`).

- **point_spacing_m**:  
  Specifies the inter-point spacing (in meters) for core points. Smaller spacing yields a denser representation, which may improve resolution at the expense of increased computational load.  
  *Allowed format*: A positive float.

<h3>m3c2</h3>

- **normal_radii**:  
  A list (e.g., `[1, 2, 3]`) defining the radii used to calculate surface normals at multiple scales. These normals capture local surface orientations, crucial for quantifying changes.  
  *Allowed format*: A list of positive floats.
  
- **cyl_radii**:  
  Sets the radius of the cylindrical neighborhood used in M3C2 distance computation. This parameter influences the selection of points for comparing corresponding regions across epochs.  
  *Allowed format*: A positive float.
  
- **max_distance**:  
  The maximum allowable distance for point comparisons.  
  *Allowed format*: A positive float.
  
- **registration_error**:  
  The computed error when aligning the two point cloud datasets.  
  *Allowed format*: A positive float, typically derived from registration uncertainty measurements.

*Additional Context:*  
The M3C2 analysis uses these settings to compute precise distance measures and to quantify uncertainties (e.g., level-of-detection values) that inform subsequent clustering and change detection.

---

## <span style="color:#c92434">**cluster_settings**</span>

- **cluster_method**:  
  Specifies the clustering algorithm used to aggregate significant change detections. DBSCAN is typically employed, although alternative methods, such as connected components, are supported.  
  *Allowed options*: `"DBSCAN"` or `"connected_components"`.
  
- **distance_threshold**:  
  The maximum distance within which points are considered part of the same cluster. This parameter is analogous to the “eps” parameter in DBSCAN.  
  *Allowed format*: A positive float.

- **cluster_by**:  
  Defines which dimensions (commonly X, Y, and Z coordinates) are used during the clustering process.  
  *Allowed format*: A list of strings. Fixed options are typically `"X"`, `"Y"`, and `"Z"`.
  
- **min_cluster_size**:  
  The minimum number of points required for a group to be recognized as a valid cluster. Clusters with fewer points are deemed insignificant.  
  *Allowed format*: A positive integer.

*Additional Context:*  
Clustering, as implemented, aggregates spatially correlated change detections into meaningful groups for further analysis or visualization.

---

## <span style="color:#c92434">**pc_projection**</span>

- **pc_path**:  
  The file path to the input point cloud (e.g., `./test_data/240826_000005.laz`).  
  *Allowed format*: A valid file path string; the file must exist and be in LAS/LAZ format.

- **make_range_image**:  
  Boolean flag to create a range image from the point cloud.  
  *Allowed format*: Boolean (`true` or `false`).

- **make_color_image**:  
  Specifies whether to generate a color image from the point cloud data.  
  *Allowed format*: Boolean (`true` or `false`).

- **create_kml**:  
  If `true`, a KML file is generated to facilitate rapid geospatial visualization.  
  *Allowed format*: Boolean (`true` or `false`).

- **top_view**:  
  Indicates whether a top-down (bird’s-eye) view is produced in the projection.  
  *Allowed format*: Boolean (`true` or `false`).

- **resolution_cm**:  
  Specifies the resolution (in centimeters) of the output image—the lower the value, the higher the resolution.  
  *Allowed format*: A positive float.

- **camera_position**:  
  The [x, y, z] coordinates defining the camera’s viewpoint for rendering, which directly impacts the visualization perspective.  
  *Allowed format*: A list or array of three numeric values (floats or integers).

- **rgb_light_intensity** and **range_light_intensity**:  
  Control the intensity of lighting in the RGB and range images, respectively.  
  *Allowed format*: Positive integers or floats, typically defined within a practical range (e.g., 0–255 for RGB).

- **epsg**:  
  The EPSG code representing the coordinate reference system of the point cloud data (e.g., 32632).  
  *Allowed format*: An integer corresponding to a valid EPSG code.

*Additional Context:*  
Projection settings are essential for generating accurate visual representations. They ensure that spatial data is correctly rendered and georeferenced for analysis and reporting.