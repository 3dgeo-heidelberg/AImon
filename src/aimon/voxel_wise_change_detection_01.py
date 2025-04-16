import vapc
from vapc.vapc_tools import extract_areas_with_change_using_mahalanobis_distance
import os

#silent mode
# vapc.enable_trace(False)
# vapc.enable_timeit(False)

# def subsample_vapc(vapc_file, voxel_size):
#     vapc_command = {
#         "tool":"subsample",
#         "args":{
#             "sub_sample_method":"closest_to_center_of_gravity"
#             }
#         }
#     sspath = vapc.do_vapc_on_files(
#         file=vapc_file,
#         out_dir=os.path.dirname(vapc_file),
#         voxel_size=voxel_size,
#         vapc_command=vapc_command,
#         save_as=".laz")
#     os.rename(vapc_file,vapc_file.replace(".laz", "_non_subsampled.laz"))
#     os.rename(sspath,vapc_file)

def compute_bitemporal_vapc(
            t1_file,
            t2_file,
            t1_vapc_out_file,
            t2_vapc_out_file,
            configuration
            ):
    if os.path.isfile(t1_vapc_out_file) and os.path.isfile(t2_vapc_out_file):
        # print("VAPC files already exist. Skipping VAPC computation.")
        return
    #Mask file
    mask_file = os.path.join(os.path.dirname(t1_vapc_out_file), "mask.las")
    #Extract areas
    alpha_value = configuration["vapc_settings"]["bi_temporal_vapc_config"]["signicance_threshold"]
    voxel_size = configuration["vapc_settings"]["vapc_config"]["voxel_size"]
    extract_areas_with_change_using_mahalanobis_distance(t1_file, t2_file, mask_file, t1_vapc_out_file, t2_vapc_out_file, voxel_size, alpha_value, delete_mask_file=False)

    #Optionally... Subsample if required. Removed for now but can be added back if needed.
    # if configuration["m3c2_settings"]["subsampling"]["voxel_size"] != 0:
    #     subsample_vapc(t1_vapc_out_file, configuration["m3c2_settings"]["subsampling"]["voxel_size"])
    #     subsample_vapc(t2_vapc_out_file, configuration["m3c2_settings"]["subsampling"]["voxel_size"])



