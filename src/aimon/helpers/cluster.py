import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
# from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from vapc import DataHandler, timeit,trace
import os

@trace
@timeit
def compute_clusters_connected_components(
                    m3c2_out_file,
                    m3c2_clustered,
                    config
                    ):
    """
    Compute clusters using connected components algorithm.
    This function reads point cloud data from a LAS file, builds a k-d tree to find pairs of points within a specified distance,
    creates a sparse adjacency matrix, and then finds connected components to determine clusters. The resulting clusters are saved
    back to a LAS file.
    Args:
        m3c2_out_file (str): Path to the input LAS file containing point cloud data.
        m3c2_clustered (str): Path to the output LAS file where clustered point cloud data will be saved.
        config (dict): Configuration dictionary containing clustering settings:
            - "cluster_settings": {
                - "distance_threshold" (float): Maximum distance between points to be considered in the same cluster.
                - "cluster_by" (str): Column name in the DataFrame to use for clustering.
                - "min_cluster_size" (int): Minimum number of points required for a cluster to be considered valid.
            }
    Returns:
        bool: True if clustering was successful, False otherwise.
    """
      
    field_name="cluster_id"
    
    distance_threshold = config["cluster_settings"]["distance_threshold"]
    cluster_by=config["cluster_settings"]["cluster_by"]
    min_cluster_sie=config["cluster_settings"]["min_cluster_size"]
    dh = DataHandler(m3c2_out_file)
    dh.load_las_files()
    df = dh.df
    pts = np.array(df[cluster_by])
    #Build kdtree
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=distance_threshold)



    # tree = BallTree(pts)
    # indices = tree.query_radius(pts,r=distance_threshold)
    # # Extract unique pairs
    # pairs = set()
    # for i, neighbors in enumerate(indices):
    #     for j in neighbors:
    #         if i < j:
    #             pairs.add((i, j))
    # pairs = list(pairs)

    # Prepare for creating a sparse adjacency matrix
    row_indices, col_indices = zip(*pairs)
    # Create a sparse adjacency matrix
    adj_matrix = csr_matrix((np.ones(len(row_indices)), (row_indices, col_indices)), shape=(len(pts), len(pts)))

    # Find connected components
    _, labels = connected_components(csgraph=adj_matrix, directed=False)

    df[field_name] = labels
    # Count sizes of each final cluster
    oids, cts = np.unique(labels, return_counts=True)
    ct_df = pd.DataFrame(data=np.array((oids, cts)).T, columns=[field_name, "cluster_size"])
    df_out = df.merge(ct_df, on=field_name, how="left", validate="many_to_one")
    df_out = df_out[df_out[field_name] > -1]
    dh.df = df_out[df_out["cluster_size"] >= min_cluster_sie]
    dh.save_as_las(m3c2_clustered)
    return True


@trace
@timeit
def compute_clusters_dbscan(
                    m3c2_out_file,
                    m3c2_clustered,
                    config
                    ):
    """
    Computes clusters using the DBSCAN algorithm and saves the clustered data.
    Parameters:
    m3c2_out_file (str): Path to the input file containing the data to be clustered.
    m3c2_clustered (str): Path to the output file where the clustered data will be saved.
    config (dict): Configuration dictionary containing clustering settings:
        - cluster_settings (dict): Dictionary containing DBSCAN parameters:
            - distance_threshold (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
            - cluster_by (str): The column name in the data to be used for clustering.
            - min_cluster_size (int): The number of samples in a neighborhood for a point to be considered as a core point.
    Returns:
    bool: True if the clustering process is successful.
    """
    field_name = "cluster_id"
    
    distance_threshold = config["cluster_settings"]["distance_threshold"]
    cluster_by = config["cluster_settings"]["cluster_by"]
    min_cluster_size = config["cluster_settings"]["min_cluster_size"]
    
    # Load data
    dh = DataHandler(m3c2_out_file)
    dh.load_las_files()
    df = dh.df
    pts = np.array(df[cluster_by])
    
    # Apply DBSCAN
    db = DBSCAN(eps=distance_threshold, min_samples=min_cluster_size)
    labels = db.fit_predict(pts)
    
    df[field_name] = labels
    # Count sizes of each final cluster
    oids, cts = np.unique(labels, return_counts=True)
    ct_df = pd.DataFrame(data=np.array((oids, cts)).T, columns=[field_name, "cluster_size"])
    df_out = df.merge(ct_df, on=field_name, how="left", validate="many_to_one")
    df_out = df_out[df_out[field_name] > -1]
    dh.df = df_out[df_out["cluster_size"] >= min_cluster_size]
    if dh.df.shape[0] == 0:
        print("No clusters found")
        return False
    dh.save_as_las(m3c2_clustered)
    return True

def cluster(
        m3c2_out_file,
        m3c2_clustered,
        config):
    """
    Clusters the M3C2 output file based on the specified clustering method in the configuration.

    Parameters:
    m3c2_out_file (str): Path to the M3C2 output file.
    m3c2_clustered (str): Path to the output file where the clustered results will be saved.
    config (dict): Configuration dictionary containing clustering settings.

    Returns:
    bool: True if clustering was successful or if the result was already computed, False otherwise.
    """
    outdir = os.path.dirname(m3c2_clustered)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    if os.path.isfile(m3c2_clustered):
        #print("Result for %s already computed."%m3c2_clustered)
        return True
    if os.path.isfile(os.path.join(outdir,"change_events.json")):
        #print("Change events already computed.")
        return True
    elif config["cluster_settings"]["cluster_method"].lower() == "connected_components" or config["cluster_settings"]["cluster_method"].lower() == "cc":
        return compute_clusters_connected_components(m3c2_out_file,m3c2_clustered,config)
    elif config["cluster_settings"]["cluster_method"].lower() == "dbscan":
        return compute_clusters_dbscan(m3c2_out_file,m3c2_clustered,config)
    else:
        print("Cluster method not implemented")
        return False