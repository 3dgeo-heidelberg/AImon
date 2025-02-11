from functools import partial
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.neighbors import BallTree
from vapc.utilities import *
from scipy.stats import chi2
from vapc import DataHandler
from vapc import Vapc
import laspy
import copy
import os

def compute_bitemporal_vapc(t1_file,
                       t2_file,
                       t1_out_file,
                       t2_out_file,
                       config):
    def _update_vapc_compute_based_on_bivapc_compute(bi_temporal_vapc_config,
                                                     vapc_config):
                    if bi_temporal_vapc_config["z_score"] is True:
                        if not "center_of_gravity" in vapc_config["compute"]:
                            vapc_config["compute"].append("center_of_gravity")
                        if not "std_of_cog" in vapc_config["compute"]:
                            vapc_config["compute"].append("std_of_cog")
                    if bi_temporal_vapc_config["fahle"] is True or bi_temporal_vapc_config["mahalanobis"] is True:
                        if not "center_of_gravity" in vapc_config["compute"]:
                            vapc_config["compute"].append("center_of_gravity")
                        if not "covariance_matrix" in vapc_config["compute"]:
                            vapc_config["compute"].append("covariance_matrix")
                    return vapc_config
    
    def _open_with_data_handler(config,file):
        if not type(file) is list:
            file = [file]
        data_handler = DataHandler(file)
        data_handler.load_las_files()
        vapc = Vapc(config["voxel_size"],
                    config["origin"],
                    config["attributes"],
                    config["compute"],
                    config["return_at"])
        vapc.get_data_from_data_handler(data_handler)
        vapc.voxelize()
        vapc.compute_point_count()
        vapc.filter_attributes(filter_attribute=config["filter"]["filter_attribute"],
                                filter_value=config["filter"]["filter_value"],
                                min_max_eq=config["filter"]["min_max_eq"]
                                )
        vapc.compute_requested_attributes()
        vapc.original_attributes = []
        vapc.compute_requested_statistics_per_attributes()
        vapc.reduce_to_voxels()
        return vapc


    # Read the JSON configuration file
    bi_temporal_vapc_config = config["vapc_settings"]["bi_temporal_vapc_config"]
    vapc_mask_config = config["vapc_settings"]["vapc_mask"]

    if (os.path.isfile(t1_out_file) or os.path.isfile(t1_out_file[:-4]+".txt")) and (os.path.isfile(t2_out_file) or os.path.isfile(t2_out_file[:-4]+".txt")) :
        #print("Result for \n%s and \n%s already computed."%(t1_out_file, t2_out_file))
        return

    # Hierarchichal approach
    # Update values to compute, based on bi-temporal vapc requirements:
    vapc_config = _update_vapc_compute_based_on_bivapc_compute(bi_temporal_vapc_config,config["vapc_settings"]["vapc_config"])
    # Apply voxelisation:
    get_voxelized_data = partial(_open_with_data_handler,vapc_config)
    vapcs = []
    i = 0
    for pc in [t1_file,t2_file]:
        # of = os.path.join("E:/trier/hierarchical_analysis/results",str(i)+".laz")
        i+=1
        vapcs.append(get_voxelized_data(pc))
        dh_t = DataHandler("")
        dh_t.df = vapcs[-1].df
        # dh_t.save_as_las(of)

    vapcs[0].compute_voxel_index()
    vapcs[1].compute_voxel_index()

    # Unique to df1
    #unique_df1 = vapcs[0].df[~vapcs[0].df['voxel_index'].isin(vapcs[1].df['voxel_index'])]
    unique_df1 = vapcs[0].df[~vapcs[0].df.index.isin(vapcs[1].df.index)]
    # Unique to df2
    #unique_df2 = vapcs[1].df[~vapcs[1].df['voxel_index'].isin(vapcs[0].df['voxel_index'])]
    unique_df2 = vapcs[1].df[~vapcs[1].df.index.isin(vapcs[0].df.index)]



    # Get Bi-Temporal Vapc:
    bi_vapc = BI_TEMPORAL_Vapc([vapcs[0].df,vapcs[1].df])
    # Compute Bi-temporal changes:
    compute_bi_temporal_statistics(bi_vapc,bi_temporal_vapc_config)
    bi_vapc.clean_merged_vapc()

    # Return to single Vapc after computations are done
    vapc = Vapc(vapc_config["voxel_size"],
                origin= vapc_config["origin"],
                attributes={})
    vapc.df = bi_vapc.df

    if (vapc.df['mahalanobi_significance'] == 0).all():
        #print("No change detected.")
        # Mask point clouds based on significant change detected within voxels
        outfile_t1 = t1_out_file[:-4]+".txt"
        outfile_t2 = t2_out_file[:-4]+".txt"
        with open(outfile_t1,"w") as f:
            f.write("No change detected")
        with open(outfile_t2,"w") as f:
            f.write("No change detected")
        return

    # Cluster by distance, given detected change is significant
    clusters_from_attribute(vapc,
                            cluster_attribute=bi_temporal_vapc_config["cluster_attribute"],
                            min_max_eq=bi_temporal_vapc_config["min_max_eq"],
                            filter_value = bi_temporal_vapc_config["filter_value"],
                            cluster_distance = bi_temporal_vapc_config["cluster_distance"],
                            cluster_by=bi_temporal_vapc_config["cluster_by"],
                            min_cluster_size = bi_temporal_vapc_config["min_cluster_size"])
    
    vapc.df['change_type'] = 0  # Significant changes
    unique_df1['change_type'] = 1  # Unique to epoch 1
    unique_df2['change_type'] = 2  # Unique to epoch 2
    # Add voxels from delta octree
    vapc.df = pd.concat([vapc.df,unique_df1,unique_df2])


    extract_by_mask(t1_file,copy.deepcopy(vapc),t1_out_file,buffer_size = vapc_mask_config["buffer_size"])
    extract_by_mask(t2_file,copy.deepcopy(vapc),t2_out_file,buffer_size = vapc_mask_config["buffer_size"])

class BI_TEMPORAL_Vapc:
    def __init__(self,
                 dfs:list = [],
                 compute:list = [],
                 return_at:dict = {"first":"center_of_gravity"}):
        """
        Parameters:
        - dfs (list): Two DataFrames representing the two temporal point clouds.
        - compute (list): List containing names of attributes to be calculated.
        - return_at (dict): Specifies what point the data will be returned at.
        """
        self.dfs = dfs
        self.compute = compute
        self.return_at = return_at
        self.distance = False
        self.merged = False
        self.z_score = False
        self.chi_squared_f = False
        self.chi_squared_mh = False
        # Temporary solution:
        self.overwrite_cog_with_coords()
        
    def overwrite_cog_with_coords(self):
        """
        Temporary solution as there are some issues to solve with laspy.
        Maps 'X', 'Y', 'Z' to 'cog_x', 'cog_y', 'cog_z' for both dataframes.
        """
        for i in range(2):
            self.dfs[i]["cog_x"] = self.dfs[i]["X"]
            self.dfs[i]["cog_y"] = self.dfs[i]["Y"]
            self.dfs[i]["cog_z"] = self.dfs[i]["Z"]
    
    
    def compute_closest_points_and_distance(self):
        df1 = self.dfs[0]
        df2 = self.dfs[1]
        # Precompute the columns
        df1_cols = df1[["cog_x", "cog_y", "cog_z"]].values
        df2_cols = df2[["cog_x", "cog_y", "cog_z"]].values
        # Create KD Trees
        tree_1 = KDTree(df1_cols)
        tree_2 = KDTree(df2_cols)
        ind_1, ind_2 = [], []
        # Vectorized operation to find nearest points
        distances, indices = tree_2.query(df1_cols, k=1)
        back_distances, back_indices = tree_1.query(df2_cols[indices], k=1)
        # Filter the indices where the nearest point in df1 is the point itself
        mask = np.arange(len(df1)) == back_indices
        ind_1 = np.arange(len(df1))[mask].tolist()
        ind_2 = indices[mask].tolist()
        df1_red = df1.iloc[ind_1].copy()
        df2_red = df2.iloc[ind_2].copy()
        tree_1 = KDTree(df1_red[["cog_x", "cog_y", "cog_z"]])
        tree_2 = KDTree(df2_red[["cog_x", "cog_y", "cog_z"]])
        distances1, indices1 = tree_2.query(df1_red[["cog_x", "cog_y", "cog_z"]], k=1)
        distances2, indices2 = tree_1.query(df2_red[["cog_x", "cog_y", "cog_z"]], k=1)
        df1_red.columns = [i+"_x" for i in df1_red.columns]
        df2_red.columns = [i+"_y" for i in df2_red.columns]
        df1_red["distance"] = distances1
        df1_red.reset_index(drop=True, inplace=True)
        df2_red.reset_index(drop=True, inplace=True)
        self.df_merged = pd.concat([df1_red,df2_red],axis = 1)
        self.distance = True
        self.merged = True
    

    def merge_vapcs(self):
        self.df_merged = self.vapcs[0].df.merge(self.vapcs[1].df, 
                                                how="left", 
                                                on=["voxel_index"])
    def clean_merged_vapc(self,
                          on = "left"):
        new_cols = ["X","Y","Z"]
        if on == "left":
            old_cols = [col+"_x" for col in new_cols]
        elif on == "right":
            old_cols = [col+"_y" for col in new_cols]
        else:
            return False
        
        if self.distance:
            new_cols.append("distance")
            old_cols.append("distance")
        if self.z_score:
            new_cols.append("z_score_significance")
            old_cols.append("z_score_significance")
        if self.chi_squared_f:
            new_cols.append("fahle_significance")
            old_cols.append("fahle_significance")
        if self.chi_squared_mh:
            new_cols.append("mahalanobi_significance")
            old_cols.append("mahalanobi_significance")

        self.df = self.df_merged[old_cols]
        self.df.columns = new_cols

    @trace
    @timeit
    def compute_chi_squared_fahle(self, alpha = 0.005):
        if not self.chi_squared_f:
            chi_squared_fahle_with_alpha = partial(chi_squared_fahle,alpha)
            self.df_merged = self.df_merged.apply(chi_squared_fahle_with_alpha,
                                                axis = 1)
            self.chi_squared_f = True
            self.df_merged["fahle_significance"] = self.df_merged["fahle_significance"].astype(int)
    
    @trace
    @timeit
    def compute_chi_squared_mahalanobis_old(self, alpha = 0.005):
        """
        Mahalanobis distance for significance testing:
        Calculates the Mahalanobis distance for a given data point x from the mean of the dataset.
        This is done in both directions. First the mean of df1 is taken and the mahlanobis distance 
        from each point in df2 to the means is calculated. Then the same is done in the other way.
        It is measured, how may standard deviations away a point is from the mean of a distribution.
        alpha defines the risk of labelling a point as an outlier, when it is not.
        """
        if not self.chi_squared_mh:
            chi_squared_mahalanobis_with_alpha = partial(chi_squared_mahalanobis,alpha)
            self.df_merged = self.df_merged.apply(chi_squared_mahalanobis_with_alpha,
                                              axis = 1)
            self.chi_squared_mh= True
            self.df_merged["mahalanobi_significance"] = self.df_merged["mahalanobi_significance"].astype(int)

    @trace
    @timeit
    def compute_chi_squared_mahalanobis(self, alpha=0.005):
        """
        Vectorized Mahalanobis chi-squared significance testing.
        This method replaces the row-by-row computation. It expects that:
          - The first epoch’s center-of-gravity is in columns 'cog_x_x', 'cog_y_x', 'cog_z_x'
          - The second epoch’s center-of-gravity is in 'cog_x_y', 'cog_y_y', 'cog_z_y'
          - The covariance matrix entries for the first epoch are in columns with names starting with 'cov_'
            and ending with '_x'
          - For the second epoch, the covariance entries are in columns ending with '_y'
        """
        if not self.chi_squared_mh:
            df = self.df_merged

            # Extract the center-of-gravity coordinates as (N, 3) arrays:
            x1 = df[['cog_x_x', 'cog_y_x', 'cog_z_x']].to_numpy()
            x2 = df[['cog_x_y', 'cog_y_y', 'cog_z_y']].to_numpy()

            # Extract covariance matrix entries.
            # IMPORTANT: Ensure the covariance columns are ordered correctly (e.g. row-major order)
            cov_cols_x = sorted([col for col in df.columns if col.startswith('cov_') and col.endswith('_x')])
            cov_cols_y = sorted([col for col in df.columns if col.startswith('cov_') and col.endswith('_y')])
            cov_x = df[cov_cols_x].to_numpy().reshape(-1, 3, 3)
            cov_y = df[cov_cols_y].to_numpy().reshape(-1, 3, 3)

            # Avoid singular matrices by adding a tiny value to the diagonal.
            eps = 1e-10
            cov_x = cov_x + np.eye(3) * eps
            cov_y = cov_y + np.eye(3) * eps

            # Batch inversion of the covariance matrices:
            inv_cov_x = np.linalg.inv(cov_x)
            inv_cov_y = np.linalg.inv(cov_y)

            # Compute the differences between the centers:
            diff = x1 - x2  # shape: (N, 3)

            # Compute the squared Mahalanobis distances:
            # d1: distance from x1 relative to x2’s covariance; d2: distance from x2 relative to x1’s covariance.
            d1 = np.einsum('ij,ijk,ik->i', diff, inv_cov_y, diff)
            d2 = np.einsum('ij,ijk,ik->i', -diff, inv_cov_x, -diff)

            # Compute p-values from the chi-squared distribution (with 3 degrees of freedom):
            p_val1 = 1 - chi2.cdf(d1, df=3)
            p_val2 = 1 - chi2.cdf(d2, df=3)

            # Decide which test to use per row (adjust logic as needed):
            is_outlier1 = p_val1 < alpha
            is_outlier2 = p_val2 < alpha

            # Use test 1 if it flags an outlier; otherwise test 2.
            significance = np.where(is_outlier1, 1, np.where(is_outlier2, 1, 0))
            p_value = np.where(is_outlier1, p_val1, p_val2)

            # Add the results to the DataFrame:
            df['mahalanobi_significance'] = significance.astype(int)
            df['p_value'] = p_value

            self.df_merged = df
            self.chi_squared_mh = True

    def compute_distance(self):
        if not self.distance:
            self.df_merged = self.df_merged.apply(distance,
                                                axis = 1)
            self.distance = True
    
    @trace
    @timeit
    def compute_z_score(self,signicance_threshold = 1.96):
        if not self.z_score:
            z_score_with_alpha = partial(z_score,signicance_threshold)
            self.df_merged = self.df_merged.apply(z_score_with_alpha,
                                    axis = 1)
            self.z_score = True
            self.df_merged["z_score_significance"] = self.df_merged["z_score_significance"].astype(float)




def compute_bi_temporal_statistics(bi_temporal_vapc,statistics_to_compute,save_mahlanobis_output = False):
    bi_temporal_vapc.compute_closest_points_and_distance()
    if statistics_to_compute["mahalanobis"] is True:
        bi_temporal_vapc.compute_chi_squared_mahalanobis(alpha =  statistics_to_compute["signicance_threshold"])
    if statistics_to_compute["fahle"] is True:
        bi_temporal_vapc.compute_chi_squared_fahle(alpha = statistics_to_compute["signicance_threshold"])
    if statistics_to_compute["z_score"] is True:
        bi_temporal_vapc.compute_z_score(signicance_threshold = statistics_to_compute["signicance_threshold"])

    if save_mahlanobis_output:
        #print(bi_temporal_vapc.df_merged)
        bi_temporal_vapc.df_merged.to_csv(save_mahlanobis_output, columns=['mahalanobis_distance', 'distance', 'p_value'], index=False)

    return bi_temporal_vapc

def get_cov_matrices_from_row(row):
    cov_columns_1 = []
    cov_columns_2 = []
    for col in row.index:
        if "cov_" in col:
            if col.endswith("_x"):
                cov_columns_1.append(col)
            else:
                cov_columns_2.append(col)
    cov_1 = np.array(row[cov_columns_1]).reshape(3, 3).astype(float)
    cov_2 = np.array(row[cov_columns_2]).reshape(3, 3).astype(float)
    return cov_1,cov_2

def get_cogs_and_covs_from_row(row):
    cog_1 = np.array(row[["cog_x_x","cog_y_x","cog_z_x"]]).T
    cog_2 = np.array(row[["cog_x_y","cog_y_y","cog_z_y"]]).T
    
    cov_1, cov_2 = get_cov_matrices_from_row(row)
    return cog_1,cog_2,cov_1,cov_2

def calculate_mean_covariance_matrix(cov_matrix_x1, cov_matrix_x2):
    """
    Calculate the inverse of the mean covariance matrix for two sets of vectors.
    :param x1: A set of vectors representing the first point cloud.
    :param x2: A set of vectors representing the second point cloud.
    :return: Inverse of the mean covariance matrix of x1 and x2.
    """

    # Calculate the mean of the two covariance matrices
    mean_cov_matrix = (cov_matrix_x1 + cov_matrix_x2) / 2
    # Calculate the inverse of the mean covariance matrix
    # mean_inv_cov_matrix = np.linalg.inv(mean_cov_matrix)
    return mean_cov_matrix

def calculate_xmean(mean_covariance_matrix, x1, x2,cov1,cov2):
    """
    Calculate the mean of the inverse of the covariance matrix.
    :param inv_covariance_matrix: Inverse of the covariance matrix
    :param x1: Mean vector for E1
    :param x2: Mean vector for E2
    :return: xmean
    """
    #check if the determinant of a matrix is too close to zero
    threshold = 1e-10
    if np.linalg.det(cov1) < threshold:
        cov1 = cov1 + np.eye(3) * threshold
    if np.linalg.det(cov2) < threshold:
        cov2= cov2 + np.eye(3) * threshold
    return mean_covariance_matrix @ (np.linalg.inv(cov1)@x1 + np.linalg.inv(cov2)@x2)/2

def compute_chi_squared_values(x, xmean, inv_mean_covariance_matrix):
    """
    Compute the chi-squared values.
    :param x: Mean vector
    :param xmean: xmean from calculate_xmean function
    :param inv_covariance_matrix: Inverse of the covariance matrix
    :return: Chi-squared values
    """
    return (x - xmean).T @ inv_mean_covariance_matrix @ (x - xmean)


def is_chi_squared_significant(chi_squared_value, degrees_of_freedom, alpha=0.05):
    """
    Compare the Chi-Squared value to the critical value from the Chi-Squared distribution.
    
    :param chi_squared_value: The smallest Chi-Squared value calculated from the test.
    :param degrees_of_freedom: The degrees of freedom for the test.
    :param alpha: The significance level (default is 0.05).
    :return: True if the Chi-Squared value is significant, False otherwise.
    """
    # Find the critical Chi-Squared value for the given degrees of freedom and alpha level
    critical_value = chi2.ppf(1 - alpha, degrees_of_freedom)
    
    # Determine if the calculated Chi-Squared value is greater than the critical value
    is_significant = chi_squared_value > critical_value
    
    return is_significant, critical_value


def compute_chi_squared_from_cov_and_cog(cog_1,cog_2,cov_1,cov_2, alpha):
    mean_covariance_matrix = calculate_mean_covariance_matrix(cov_1,cov_2)
    x_mean = calculate_xmean(mean_covariance_matrix,cog_1,cog_2,cov_1,cov_2)
    xE1 = compute_chi_squared_values(cog_1,x_mean,np.linalg.inv(mean_covariance_matrix))
    xE2 = compute_chi_squared_values(cog_2,x_mean,np.linalg.inv(mean_covariance_matrix))
    xEmin = np.min([xE1,xE2])
    # print(xEmin)
    significant, critical_value = is_chi_squared_significant(xEmin, 3,alpha=alpha)
    # print(f"Is the Chi-Squared value significant? {significant}")
    # print(f"Critical value at alpha = 0.05: {critical_value}")

    return pd.Series([significant,xEmin,critical_value],
                     index=["fahle_significance","xEmin","critical_value"])



def compute_3d_distance(p1,p2):
    d = np.linalg.norm(p2-p1)
    return pd.Series([d],
                     index=["distance"])

def chi_squared_fahle(alpha,df):
    # print(df)
    cog_1,cog_2,cov_1,cov_2 = get_cogs_and_covs_from_row(df)
    # print(cov_1,cov_2)
    res_series = compute_chi_squared_from_cov_and_cog(cog_1,cog_2,cov_1,cov_2,alpha)
    df = pd.concat([df,res_series])
    return df

def distance(df):
    cog_1,cog_2,_,_ = get_cogs_and_covs_from_row(df)
    distance = compute_3d_distance(cog_1,cog_2)
    df = pd.concat([df,distance])
    return df

def z_score(significance_threshold,df):
    X_1 = np.array(df[["cog_x_y","cog_y_y","cog_z_y"]])
    mu_1 = np.array(df[["cog_x_x","cog_y_x","cog_z_x"]])
    sigma_1 = np.array(df[["std_x_x","std_y_x","std_z_x"]])
    if 0 in sigma_1:
        sigma_1 += 1e-10
    Z_1 = np.abs((X_1-mu_1)/sigma_1)
    z_1_max = Z_1.max()

    X_2 = np.array(df[["cog_x_x","cog_y_x","cog_z_x"]])
    mu_2 = np.array(df[["cog_x_y","cog_y_y","cog_z_y"]])
    sigma_2 = np.array(df[["std_x_y","std_y_y","std_z_y"]])
    if 0 in sigma_2:
        sigma_2 += 1e-10
    Z_2 = np.abs((X_2-mu_2)/sigma_2)
    z_2_max = Z_2.max()


    if z_1_max > z_2_max:
        sign = z_2_max >= significance_threshold
        df = pd.concat([df,pd.Series([Z_2[0],Z_2[1],Z_2[2],sign],["ZS_x","ZS_y","ZS_z","z_score_significance"])])

    else:
        sign = z_1_max >= significance_threshold
        df = pd.concat([df,pd.Series([Z_1[0],Z_1[1],Z_1[2],sign],["ZS_x","ZS_y","ZS_z","z_score_significance"])])
    return df

def get_cov_matrices_from_row(row):
    cov_columns_1 = []
    cov_columns_2 = []
    for col in row.index:
        if "cov_" in col:
            if col.endswith("_x"):
                cov_columns_1.append(col)
            else:
                cov_columns_2.append(col)
    cov_1 = np.array(row[cov_columns_1]).reshape(3, 3).astype(float)
    cov_2 = np.array(row[cov_columns_2]).reshape(3, 3).astype(float)
    return cov_1,cov_2

def get_cogs_and_covs_from_row(row):
    cog_1 = np.array(row[["cog_x_x","cog_y_x","cog_z_x"]]).T
    cog_2 = np.array(row[["cog_x_y","cog_y_y","cog_z_y"]]).T
    
    cov_1, cov_2 = get_cov_matrices_from_row(row)
    return cog_1,cog_2,cov_1,cov_2
    
def mahalanobis_distance(x, mean, cov):
    x_minus_mean = x - mean
    return np.sqrt(x_minus_mean.T @ np.linalg.inv(cov) @ x_minus_mean)

# def chi_squared_mahalanobis(alpha,df):
#     x1,x2,cov_x1,cov_x2 = get_cogs_and_covs_from_row(df)
#     p_value1 = 1 - chi2.cdf(mahalanobis_distance(x1, x2, cov_x2), len(x2))
#     outlier1 = p_value1 < alpha
#     p_value2 = 1 - chi2.cdf(mahalanobis_distance(x2, x1, cov_x1), len(x1))
#     outlier2 = p_value2 < alpha
    
#     if outlier1:
#         res_series = pd.Series([outlier1,p_value1],
#                      index=["mahalanobi_significance","p_value"])
#         return pd.concat([df,res_series])
#     else:
#         res_series = pd.Series([outlier2,p_value2],
#                      index=["mahalanobi_significance","p_value"])
#         return pd.concat([df,res_series])
    
def chi_squared_mahalanobis(alpha,df):
    x1,x2,cov_x1,cov_x2 = get_cogs_and_covs_from_row(df)
    p_value1 = 1 - chi2.cdf(mahalanobis_distance(x1, x2, cov_x2)**2, len(x2))
    outlier1 = p_value1 < alpha
    p_value2 = 1 - chi2.cdf(mahalanobis_distance(x2, x1, cov_x1)**2, len(x1))
    outlier2 = p_value2 < alpha
    
    if outlier1:
        res_series = pd.Series([outlier1,p_value1],
                     index=["mahalanobi_significance","p_value"])
        return pd.concat([df,res_series])
    else:
        res_series = pd.Series([outlier2,p_value2],
                     index=["mahalanobi_significance","p_value"])
        return pd.concat([df,res_series])



def write_bi_temporal_to_laz(outfile, bi_temporal_vapc, coords_of = "first",voxel_size = 1):
    #Check calculations:
    distance = bi_temporal_vapc.distance
    chi_squared_f = bi_temporal_vapc.chi_squared_f
    z_score = bi_temporal_vapc.z_score
    chi_squared_mg = bi_temporal_vapc.chi_squared_mh
    if coords_of == "first":
        add_str = "x"
    elif coords_of == "second":
        add_str = "y"

    X = bi_temporal_vapc.df_merged["cog_x_%s"%add_str]
    Y = bi_temporal_vapc.df_merged["cog_y_%s"%add_str]
    Z = bi_temporal_vapc.df_merged["cog_z_%s"%add_str]
    relevant_output = [X,Y,Z]
    column_names = ["X","Y","Z"]
    if distance:
        relevant_output.append(np.array(bi_temporal_vapc.df_merged["distance"]))
        column_names += ["distance"]
    if chi_squared_f:
        relevant_output.append(np.array(bi_temporal_vapc.df_merged["fahle_significance"]))
        relevant_output.append(np.array(bi_temporal_vapc.df_merged["xEmin"]))
        relevant_output.append(np.array(bi_temporal_vapc.df_merged["critical_value"]))
        column_names += ["fahle_significance","xEmin","critical_value"]
    if chi_squared_mg:
        relevant_output.append(np.array(bi_temporal_vapc.df_merged["mahalanobi_significance"]))
        relevant_output.append(np.array(bi_temporal_vapc.df_merged["p_value"]))
        column_names += ["mahalanobi_significance","p_value"]
    if z_score:
        relevant_output.append(np.array(bi_temporal_vapc.df_merged["ZS_x"]))
        relevant_output.append(np.array(bi_temporal_vapc.df_merged["ZS_y"]))
        relevant_output.append(np.array(bi_temporal_vapc.df_merged["ZS_z"]))
        relevant_output.append(np.array(bi_temporal_vapc.df_merged["z_score_significance"]))
        column_names+=["z_score_x","z_score_y","z_score_z","z_score_significance"]

    out_data = pd.DataFrame(np.array(relevant_output).T,columns = column_names, dtype=float)
    dh = DataHandler([])
    dh.df = out_data
    dh.las_header = laspy.LasHeader()
    dh.save_as_las(outfile)
    dh.save_as_ply(outfile[:-4]+".ply",voxel_size)
    return out_data


def clusters_from_attribute(vapc,cluster_attribute,min_max_eq,filter_value,cluster_distance,cluster_by,min_cluster_size):
            vapc.filter_attributes(filter_attribute=cluster_attribute,
                                    min_max_eq=min_max_eq,
                                    filter_value=filter_value)

            vapc.compute_clusters(cluster_distance=cluster_distance,
                                    cluster_by = cluster_by)
            vapc.filter_attributes(filter_attribute="cluster_size",
                                    min_max_eq="greater_than_or_equal_to",
                                    filter_value=min_cluster_size)


def extract_by_mask(pc_file,vapc_mask,pc_file_masked,buffer_size = 2):
                #print(buffer_size)
                dh = DataHandler([pc_file])
                dh.load_las_files()
                print("a")
                vapc_pc = Vapc(float(vapc_mask.voxel_size))
                
                vapc_pc.get_data_from_data_handler(dh)
                print("b")

                vapc_mask.compute_voxel_buffer(buffer_size = int(buffer_size))
                print("c")
                vapc_mask.df = vapc_mask.buffer_df
                #Select by mask
                vapc_pc.select_by_mask(vapc_mask,"voxel_index")
                print("d")
                #Undo offset
                # vapc_pc.compute_offset()
                #Save Point Cloud
                dh.df = vapc_pc.df
                dh.save_as_las(pc_file_masked)
                return vapc_pc