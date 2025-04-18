import json
import os
import re
import uuid
import numpy as np
from datetime import datetime
from scipy.spatial import ConvexHull
from vapc import DataHandler
import pandas as pd
import matplotlib.pyplot as plt

#Rule based classification and filtering
from aimon.helpers.classification import extract_features_all,classify_event

###### Dimensionality Reduction ######
from aimon.helpers.classification import extract_features_umap
from sklearn.mixture import GaussianMixture
import umap.umap_ as umap  # UMAP for dimensionality reduction

#Random forest classification
import joblib
from aimon.helpers.classification import save_model, load_model, extract_features_for_random_forest
from sklearn.ensemble    import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics        import (
    classification_report,
    jaccard_score,
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

###########################################################################
# Helper Functions
###########################################################################
def extract_time_info(filepath, date_format="%y%m%d_%H%M%S"):
    """
    Extracts t_min, t_max, and delta_t_hours from the filepath.
    Returns scalar values rather than lists.
    """
    pattern = r"(\d{6}_\d{6})"
    matches = re.findall(pattern, filepath)
    if len(matches) >= 2:
        t1_str = matches[0]
        t2_str = matches[-1]
        dt1 = datetime.strptime(t1_str, date_format)
        dt2 = datetime.strptime(t2_str, date_format)
        if dt1 < dt2:
            t_min_str, t_max_str = t1_str, t2_str
            t_min, t_max = dt1, dt2
        else:
            t_min_str, t_max_str = t2_str, t1_str
            t_min, t_max = dt2, dt1
        delta_t = round((t_max - t_min).total_seconds() / 3600, 3)
        return {"t_min": t_min_str,
                "t_max": t_max_str,
                "delta_t_hours": delta_t}
    else:
        raise ValueError("Insufficient timestamps found in filepath.")

def get_change(points, stat):
    """
    Calculate a statistical measure of the absolute distances from the given points.
    """
    abs_dist = np.abs(points.M3C2_distance)
    if stat == "std":
        return np.nanstd(abs_dist)
    if stat == "mean":
        return np.nanmean(abs_dist)
    if stat == "min":
        return np.nanmin(abs_dist)
    if stat == "max":
        return np.nanmax(abs_dist)
    if stat == "median":
        return np.nanmedian(abs_dist)
    if stat == "quant90":
        return np.nanquantile(abs_dist, .90)
    if stat == "quant95":
        return np.nanquantile(abs_dist, .95)
    if stat == "quant99":
        return np.nanquantile(abs_dist, .99)
    raise ValueError("Unknown stat option: " + stat)


def get_geometric_features(points):
    """
    Calculate a statistical measure of the absolute distances from the given points. 
    https://doi.org/10.5194/isprsannals-II-5-W2-313-2013
    """
    def _safe_log(x):
        # Return 0 when x is 0; otherwise return the log.
        return np.where(x > 0, np.log(x), 0)
    epoch1 = points[points.epoch == 0]
    epoch2 = points[points.epoch == 1]
    both_epochs = points
    epochs = [epoch1, epoch2, both_epochs]
    geometric_features = []
    for epoch in epochs:
        epoch_points = epoch[["X", "Y", "Z"]]
        if len(epoch_points) < 3:
            geometric_features.append({
            })
            continue
        #Compute covarivance matrix
        cov = np.cov(epoch_points.values.T)
        #Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        #Solve by eigenvalues descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        eigenvalue1 = eigenvalues[0]
        eigenvalue2 = eigenvalues[1]
        eigenvalue3 = eigenvalues[2]
        sum_of_eigenvalues = np.sum(eigenvalues)
        eigenvalue1_normalized = eigenvalue1 / sum_of_eigenvalues
        eigenvalue2_normalized = eigenvalue2 / sum_of_eigenvalues
        eigenvalue3_normalized = eigenvalue3 / sum_of_eigenvalues
        #Compute geometric features
        linearity = (eigenvalue1 - eigenvalue2) / eigenvalue1
        planarity = (eigenvalue2 - eigenvalue3) / eigenvalue1
        sphericity = eigenvalue3 / eigenvalue1
        omnivariance = (eigenvalue1 * eigenvalue2 * eigenvalue3) ** (1/3)
        anisotropy = (eigenvalue1 - eigenvalue3) / eigenvalue1
        eigentropy = - (
            eigenvalue1_normalized * _safe_log(eigenvalue1_normalized)
            + eigenvalue2_normalized * _safe_log(eigenvalue2_normalized)
            + eigenvalue3_normalized * _safe_log(eigenvalue3_normalized)
        )
        surface_variation = eigenvalue3 / sum_of_eigenvalues # surface variation (http://dx.doi.org/10.1109/CVPR.2016.178)
        verticality = 1 - eigenvectors[2][2]
        gf = {"sum_of_eigenvalues": round(sum_of_eigenvalues,5),
                "omnivariance": round(omnivariance,5),
                "eigentropy": round(eigentropy,5),
                "anisotropy": round(anisotropy,5),
                "planarity": round(planarity,5),
                "linearity": round(linearity,5),
                "surface_variation": round(surface_variation,5),
                "sphericity": round(sphericity,5),
                "verticality": round(verticality,5)}
        geometric_features.append(gf)
    return geometric_features[0], geometric_features[1], geometric_features[2]

def hull_of_points_to_obj(points, obj_file):
    """
    Calculate the convex hull, reorient its faces if needed, and write an OBJ file.
    """
    if len(points) < 4:
        return None

    hull = ConvexHull(points)
    hull_centroid = np.mean(points[hull.vertices], axis=0)
    faces = []
    for simplex in hull.simplices:
        pts = points[simplex]
        face_center = pts.mean(axis=0)
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        normal = np.cross(v1, v2)
        if np.dot(normal, face_center - hull_centroid) < 0:
            simplex = simplex[::-1]
        faces.append(simplex)
    faces_arr = np.array(faces)
    unique_indices = np.unique(faces_arr)
    index_mapping = {old_idx: new_idx + 1 for new_idx, old_idx in enumerate(unique_indices)}

    with open(obj_file, "w") as file:
        for old_idx in unique_indices:
            x, y, z = points[old_idx]
            file.write("v {:.6f} {:.6f} {:.6f}\n".format(x, y, z))
        for face in faces:
            mapped = [index_mapping[idx] for idx in face]
            file.write("f {} {} {}\n".format(*mapped))
    return hull

def get_conv_hull_points(df):
    """
    Given a DataFrame of 3D points, compute convex hull properties.
    Returns a tuple with points used, surface areas, volumes and area/volume ratios.
    """
    points = df[["X", "Y", "Z"]].values
    if len(points) < 4:
        return ([], 0, 0, 0)
    hull = ConvexHull(points)
    simplices_list = [list(pt) for pt in points[hull.vertices]]
    return (simplices_list, hull.area, hull.volume, hull.area / hull.volume)

# ---------------------------------------------------------------------------
# Object-Based Classes
# ---------------------------------------------------------------------------
class ChangeEvent:
    def __init__(self, object_id, event_type="undefined", cluster_point_cloud=None,
                 cluster_point_cloud_chull=None, start_date=None, number_of_points=None,
                 t_min=None, t_max=None, delta_t_hours=None, change_magnitudes=None,
                 convex_hull=None, geometric_features_epoch_1=None, geometric_features_epoch_2=None, geometric_features_both_epochs=None):
        self.object_id = object_id
        self.event_type = event_type
        self.cluster_point_cloud = cluster_point_cloud
        self.cluster_point_cloud_chull = cluster_point_cloud_chull
        self.start_date = start_date
        self.number_of_points = number_of_points
        self.t_min = t_min
        self.t_max = t_max
        self.delta_t_hours = delta_t_hours
        # Store each statistical measure as a scalar
        self.change_magnitudes = change_magnitudes if change_magnitudes is not None else {}
        # convex_hull is a dictionary with keys: 'points_building', 'surface_areas', 'volumes', 'ratios'
        self.convex_hull = convex_hull if convex_hull is not None else {}
        self.geometric_features_epoch_1 = geometric_features_epoch_1 if geometric_features_epoch_1 is not None else {}
        self.geometric_features_epoch_2 = geometric_features_epoch_2 if geometric_features_epoch_2 is not None else {}
        self.geometric_features_both_epochs = geometric_features_both_epochs if geometric_features_both_epochs is not None else {}

    @classmethod
    def from_cluster(cls, cluster_df, cluster, m3c2_file, pc_folder, obj_folder):
        """
        Create a ChangeEvent from a cluster of points.
        """
        # Generate a unique object ID for the event
        object_id = str(uuid.uuid4())

        cluster_pc = os.path.join(pc_folder, f"{cluster}.laz")
        cluster_obj = os.path.join(obj_folder, f"{cluster}.obj")
        number_of_points = len(cluster_df)

        # Extract time info from the filename (each field will be scalar, not wrapped in a list)
        times = extract_time_info(m3c2_file)
        t_min = times["t_min"]
        t_max = times["t_max"]
        delta_t = times["delta_t_hours"]

        # Compute change magnitude statistics
        stats = ["mean", "std", "min", "max", "median", "quant90", "quant95", "quant99"]
        change_stats = {stat: round(get_change(cluster_df, stat), 3) for stat in stats}
        # Add geometric features to the geometric features dictionary.
        geo_f = ["Sum_of_Eigenvalues","Omnivariance", "Eigentropy", "Anisotropy", "Planarity", "Linearity","Surface_Variation", "Sphericity"]
        geo_f_epoch_1, geo_f_epoch_2, geo_f_both_epochs = get_geometric_features(cluster_df)
        
        # Compute convex hull properties
        simplices_list, area, volume, surface_area_to_volume_ratios = get_conv_hull_points(cluster_df)
        convex_data = {
            "surface_area": area,
            "volume": volume,
            "surface_area_to_volume_ratio": surface_area_to_volume_ratios,
            "points_building": simplices_list
        }
        # For start_date you might either extract it from the file or use current timestamp.
        start_date = datetime.now().strftime("%y%m%d_%H%M%S")

        return cls(object_id=object_id, event_type="undefined",
                   cluster_point_cloud=cluster_pc,
                   cluster_point_cloud_chull=cluster_obj,
                   start_date=start_date,
                   number_of_points=number_of_points,
                   t_min=t_min,
                   t_max=t_max,
                   delta_t_hours=delta_t,
                   change_magnitudes=change_stats,
                   convex_hull=convex_data,
                   geometric_features_epoch_1=geo_f_epoch_1,
                   geometric_features_epoch_2=geo_f_epoch_2,
                   geometric_features_both_epochs=geo_f_both_epochs)

    def to_dict(self):
        """
        Serialize the ChangeEvent as a dictionary. Scalar values remain as such.
        """
        return {
            "object_id": self.object_id,
            "event_type": self.event_type,
            "cluster_point_cloud": self.cluster_point_cloud,
            "cluster_point_cloud_chull": self.cluster_point_cloud_chull,
            "start_date": self.start_date,
            "number_of_points": self.number_of_points,
            "t_min": self.t_min,
            "t_max": self.t_max,
            "delta_t_hours": self.delta_t_hours,
            "change_magnitudes": self.change_magnitudes,
            "convex_hull": self.convex_hull,
            "geometric_features_epoch_1": self.geometric_features_epoch_1,
            "geometric_features_epoch_2": self.geometric_features_epoch_2,
            "geometric_features_both_epochs": self.geometric_features_both_epochs
        }
    
    def matches(self, conditions):
        """
        Return True if this event satisfies every feature threshold
        in the `conditions` dict, e.g.:
           {"change_mean": {"min": 0.1},
            "hull_volume": {"max": 50},
            "event_type": {"exact": "undefined"}}.
        """
        # Build a one‑row features DataFrame
        # df = extract_features_all([self])
        # row = df.iloc[0].to_dict()
        row = self.to_dict()
        for feature, thr in conditions.items():
            val = row.get(feature, float("nan"))
            if pd.isna(val):
                return False
            if "min" in thr and val < thr["min"]:
                return False
            if "max" in thr and val > thr["max"]:
                return False
            if "exact" in thr and val != thr["exact"]:
                return False
            if "in" in thr and val not in thr["in"]:
                return False
            if "not_in" in thr and val in thr["not_in"]:
                return False
        return True

    @classmethod
    def from_dict(cls, d):
        """
        Create a ChangeEvent from a dictionary.
        """
        return cls(
            object_id=d.get("object_id"),
            event_type=d.get("event_type", "undefined"),
            cluster_point_cloud=d.get("cluster_point_cloud"),
            cluster_point_cloud_chull=d.get("cluster_point_cloud_chull"),
            start_date=d.get("start_date"),
            number_of_points=d.get("number_of_points"),
            t_min=d.get("t_min"),
            t_max=d.get("t_max"),
            delta_t_hours=d.get("delta_t_hours"),
            change_magnitudes=d.get("change_magnitudes"),
            convex_hull=d.get("convex_hull"),
            geometric_features_epoch_1=d.get("geometric_features_epoch_1"),
            geometric_features_epoch_2=d.get("geometric_features_epoch_2"),
            geometric_features_both_epochs=d.get("geometric_features_both_epochs")
        )

    
    def __repr__(self):
        return f"<ChangeEvent {self.object_id}>"

class ChangeEventCollection:
    def __init__(self, events=None):
        self.events = events if events is not None else []

    def add_event(self, event):
        """
        Add a change event if its object_id is not already present.
        """
        if not any(ev.object_id == event.object_id for ev in self.events):
            self.events.append(event)
            # print("Added event:", event.object_id)
        # else:
            # print("Event already exists:", event.object_id)

    
    def add_event_type_label(self, object_id, event_type):
        """
        Add a change event if its object_id is not already present.
        """
        # Check where ev.object_id is object_id, add event_type to that event
        for ev in self.events:
            if ev.object_id == object_id:
                ev.event_type = event_type
                # print("Added event type:", event_type, "to event:", object_id)
                return

    def to_list(self):
        """
        Convert the collection to a list of dictionaries.
        """
        return [event.to_dict() for event in self.events]

    def to_dataframe(self):
        self.df = extract_features_all(self.events)
        return self.df
    
    def save_to_file(self, filename):
        """
        Save the change event collection as a JSON file.
        """
        with open(filename, 'w') as f:
            json.dump(self.to_list(), f, indent=4)

    @classmethod
    def load_from_file(cls, filename):
        """
        Load a collection of change events from a JSON file.
        """
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
            # events = []
            # for item in data:
            #     events.append(ChangeEvent.from_dict(item))
            events = [ChangeEvent.from_dict(item) for item in data]
        else:
            events = []
        return cls(events)

    def attach_from_file(self, filename):
        """
        Attach new change events from an external file to this collection,
        only adding events that are not already present.
        """
        new_collection = ChangeEventCollection.load_from_file(filename)
        for event in new_collection.events:
            # print("Adding:",event.object_id)
            self.add_event(event)

    def merge_from_folder(self, folder):
        """
        Iterate over subfolders within a folder, looking for change_events.json files,
        and merge the events into this collection.
        """
        for subfolder in os.listdir(folder):
            subfolder_path = os.path.join(folder, subfolder)
            if os.path.isdir(subfolder_path):
                file_path = os.path.join(subfolder_path, "change_events.json")
                if os.path.exists(file_path):
                    new_coll = ChangeEventCollection.load_from_file(file_path)
                    for event in new_coll.events:
                        self.add_event(event)


    def filter_events_rule_based(self,
                                 filter_rule
                                ):
        """
        Filter each ChangeEvent in this collection according
        to the provided rules, and write the remaining change events back into
        a new ChangeEventCollection.
        
        filter_rules should be a dict of the form:
        
          {
            "filter": {
               "change_mean":   {"min": 0.1, "max": 1.0},
               "hull_volume":   {"max": 50},
               …
          }
        """
        # Start with all events, then pare down
        filtered = self.events
        filtered = [ev for ev in filtered if ev.matches(filter_rule)]
        return filtered


    def classify_events_rule_based(self, classification_rules):
        """
        Classify each ChangeEvent in this collection according
        to the provided rules, and write the label back into
        each event.event_type.  Returns the feature‐matrix DataFrame.
        
        classification_rules should be a dict of the form:
        
          {
            "labelA": {
               "change_mean":   {"min": 0.1, "max": 1.0},
               "hull_volume":   {"max": 50},
               …
            },
            "labelB": { … }
            …
          }
        """
        # 1) build a feature‐matrix DataFrame from the events
        features_df = extract_features_all(self.events)

        # 2) classify each row & update the corresponding ChangeEvent
        for _, row in features_df.iterrows():
            object_id = row["object_id"]
            label = classify_event(row, classification_rules)
            self.add_event_type_label(object_id, label)
            # Update the event in the features_df
            features_df.at[_, "event_type"] = label
        return features_df
    
    def train_random_forest(
        self,
        ignore_labels: list[str] = None,
        param_grid: dict         = None,
        test_size: float         = 0.2,
        random_state: int        = 42
    ) -> RandomForestClassifier:
        """
        Train a RandomForestClassifier on this.collection.events.
        - ignore_labels: drop any events whose .event_type is in this list
        - param_grid:     sklearn‐style hyperparam grid for GridSearchCV
        Returns the best‐estimator.
        """
        # 1) optionally filter out unwanted labels
        events = [e for e in self.events
                  if not ignore_labels or e.event_type not in ignore_labels]

        # 2) extract features & labels; assumes extract_features_all yields 'event_type'
        df = extract_features_for_random_forest(events)
        X = df.drop(columns=["object_id", "event_type"])
        y = df["event_type"]

        # 3) train/test split (stratify to keep class balance)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        # 4) default param_grid if none provided
        if param_grid is None:
            param_grid = {
                "n_estimators":       [100, 200],
                "max_depth":          [None, 10, 20],
                "min_samples_split":  [2, 5]
            }

        # 5) grid‑search
        grid = GridSearchCV(
            RandomForestClassifier(random_state=random_state),
            param_grid,
            cv=5,
            n_jobs=-1,
            scoring="f1_macro"
        )
        grid.fit(X_train, y_train)

        best_rf = grid.best_estimator_
        print(f"Best params: {grid.best_params_}")
        print("Train  F1:", grid.score(X_train, y_train))
        print("Test   F1:", grid.score(X_test, y_test))

        # 6) classification report
        y_pred = best_rf.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # 7) confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=best_rf.classes_)
        disp = ConfusionMatrixDisplay(cm, display_labels=best_rf.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()

        # 8) feature importance
        importances = best_rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        feat_names = X.columns
        plt.figure()
        plt.title("Feature Importances")
        plt.bar(range(len(indices)), importances[indices], align="center")
        plt.xticks(range(len(indices)),
                   [feat_names[i] for i in indices],
                   rotation=90)
        plt.tight_layout()
        plt.show()

        # 9) extra metrics
        jaccard = jaccard_score(y_test, y_pred, average='macro')
        accuracy = accuracy_score(y_test, y_pred)
        f1       = f1_score(y_test, y_pred, average='macro')
        mcc      = matthews_corrcoef(y_test, y_pred)
        kappa    = cohen_kappa_score(y_test, y_pred)

        print(f"Jaccard Score:                  {jaccard:.3f}")
        print(f"Accuracy Score:                 {accuracy:.3f}")
        print(f"F1 Score (macro):               {f1:.3f}")
        print(f"Matthews Corr. Coef.:           {mcc:.3f}")
        print(f"Cohen's Kappa:                  {kappa:.3f}")
        self.model = best_rf

        
    def save_model(self, file_path: str):
        """
        Save the trained model to disk using joblib.
        """
        if not hasattr(self, 'model'):
            raise ValueError("Model not trained yet. Please train the model before saving.")
        
        joblib.dump(self.model, file_path)
        print(f"Model saved to {file_path}")

    def load_model(self,file_path: str):
        """
        Load a machine learning model from disk using joblib.
        """
        self.model = joblib.load(file_path)
        print(f"Model loaded from {file_path}")

    def apply_random_forest(
        self,
        model: str = None):
        """
        Given either a fitted model or a filepath, predict new event_types for
        every event in this collection (in‑place).
        """
        # 1) load the model if we were given a path
        if isinstance(model, str):
            model = load_model(model)
        else:
            model = self.model

        # 2) re‑extract features
        df = extract_features_for_random_forest(self.events)
        X = df.drop(columns=["object_id", "event_type"])
        preds = model.predict(X)

        # 3) map them back into each ChangeEvent
        for ev, label in zip(self.events, preds):
            ev.event_type = label


    
    ######### UMAP ##########

    def prep_data_for_umap(self, 
                           ignore_features = ["object_id","event_type","delta_t_hours","hull_surface_area","hull_volume"],
                           supervised_label = None):
        if not hasattr(self, 'df'):
            self.to_dataframe()

        # Handle missing values by dropping them
        self.df = self.df.dropna()
        if self.df.empty:
            raise ValueError("No complete data available after dropping missing values.")
    
        X = self.df
        
        if supervised_label is not None:
            # Check if the supervised label exists in the DataFrame
            if supervised_label not in X.columns:
                raise ValueError(f"Supervised label '{supervised_label}' not found in DataFrame columns.")

            y = X[supervised_label]
            y = y.astype("category").cat.codes  # Convert to categorical codes
            # Dict to map labels to integers
            self.y_label_map = {code:label  for label, code in zip(X[supervised_label].unique(), y.unique())}

        # 1) optionally filter out unwanted features
        for col in ignore_features+[supervised_label]:
            if col in X.columns:
                X = X.drop(columns=[col])

        self.X_umap = X.values
        self.y_umap = y.values if supervised_label is not None else None

    def fit_UMAP(
        self,
        n_neighbors,
        min_dist,
        n_components,
        metric = "euclidean",
        random_state = 3,
    ):
        """
        Train a RandomForestClassifier on this.collection.events.
        - ignore_labels: drop any events whose .event_type is in this list
        - param_grid:     sklearn‐style hyperparam grid for GridSearchCV
        Returns the best‐estimator.
        """

        if not hasattr(self, 'X_umap'):
            raise ValueError("Data not prepared for UMAP. Please call prep_data_for_umap() first.")

        # Standardize the feature set
        # scaler = StandardScaler()
        # X_scaled = scaler.fit_transform(features_df_without_id.values)

        # UMAP
        reducer = umap.UMAP(n_neighbors=n_neighbors,
                            min_dist=min_dist,
                            n_components=n_components, 
                            random_state=random_state,
                            metric=metric)

        # Fit UMAP to the data
        if self.y_umap is not None:
            # If supervised label is provided, fit UMAP with labels
            reducer.fit(self.X_umap, self.y_umap)
        else:
            # If no labels are provided, fit UMAP without labels
            reducer.fit(self.X_umap)
        self.umap_reducer = reducer

    def transform_UMAP(self):
        if not hasattr(self, 'umap_reducer'):
            raise ValueError("UMAP not fitted. Please call fit_UMAP() first.")
        # Transform the data using the fitted UMAP model
        self.X_umap_transformed = self.umap_reducer.transform(self.X_umap)


    def plot_UMAP(self, save_path=None):
        if not hasattr(self, 'X_umap_transformed'):
            raise ValueError("UMAP not transformed. Please call transform_UMAP() first.")
        if self.y_umap is None:
            plt.scatter(self.X_umap_transformed[:, 0], self.X_umap_transformed[:, 1], s=.5, alpha=1)
        else:
            for label in np.unique(self.y_umap):
                plt.scatter(self.X_umap_transformed[self.y_umap == label, 0], self.X_umap_transformed[self.y_umap == label, 1], label=self.y_label_map[label], s=.5, alpha=1)
        plt.title("UMAP projection of the change events")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        if self.y_umap is not None:
            plt.legend()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

    def save_UMAP_model(self, file_path: str):
        """
        Save the trained UMAP model to disk using joblib.
        """
        if not hasattr(self, 'umap_reducer'):
            raise ValueError("UMAP not trained yet. Please train the model before saving.")
        
        joblib.dump(self.umap_reducer, file_path)
        print(f"UMAP model saved to {file_path}")

    def load_UMAP_model(self, file_path: str):
        """
        Load a UMAP model from disk using joblib.
        """
        self.umap_reducer = joblib.load(file_path)
        print(f"UMAP model loaded from {file_path}")


    def plot_feature_expression_heatmap(
        self,
        normalize: str = "minmax",
        cmap: str = "Reds",
        figsize: tuple = (10, 10),
        fontsize_xtick: int = 6,
        fontsize_ytick: int = 12,
        cbar_label: str = "Normalized value",
        exclude = {"object_id", "event_type", "delta_t_hours", "hull_surface_area", "hull_volume","number_of_points"}
    ):
        """
        Draw a heatmap of each event (rows) across every feature (columns),
        with per‑feature normalization, one y‑tick per event_type,
        and horizontal lines marking each class boundary.

        Parameters
        ----------
        normalize : {'minmax', 'zscore', None}
            How to normalize each feature (column).  
            'minmax' → (x - min)/(max - min)  
            'zscore' → (x - μ)/σ  
            None     → leave raw values

        cmap : str
            A matplotlib colormap name.

        figsize : (width, height)
            Figure size in inches.

        fontsize_xtick, fontsize_ytick : int
            Font sizes for the tick labels.

        cbar_label : str
            Label for the colorbar.
        """
        import numpy as _np
        import matplotlib.pyplot as _plt

        # 1) build or refresh the DataFrame
        if not hasattr(self, "df"):
            self.to_dataframe()
        df = self.df.dropna()

        # 2) select numeric features, excluding IDs and known non‑features
        numeric_cols = df.select_dtypes(include=[_np.number]).columns
        feature_cols = [c for c in numeric_cols if c not in exclude]
        if not feature_cols:
            raise ValueError("No feature columns found to plot.")

        X = df[feature_cols].values
        classes = df["event_type"].values

        # 3) normalize per feature (axis=0)
        if normalize == "minmax":
            mins = X.min(axis=0, keepdims=True)
            maxs = X.max(axis=0, keepdims=True)
            X_norm = (X - mins) / (maxs - mins + 1e-8)
        elif normalize == "zscore":
            means = X.mean(axis=0, keepdims=True)
            stds = X.std(axis=0, keepdims=True)
            X_norm = (X - means) / (stds + 1e-8)
        elif normalize is None:
            X_norm = X
        else:
            raise ValueError(f"Unsupported normalize: {normalize!r}")

        # 4) sort events by class for grouping
        order = _np.argsort(classes)
        X_plot = X_norm[order]
        class_plot = classes[order]

        # 5) compute block boundaries & tick positions
        #    mask marks where each new class block starts
        mask = _np.r_[True, class_plot[1:] != class_plot[:-1]]
        starts = _np.nonzero(mask)[0]
        ends = _np.r_[starts[1:], len(class_plot)]
        midpoints = (starts + ends - 1) / 2
        labels = class_plot[starts]

        # 6) plot
        _plt.figure(figsize=figsize)
        im = _plt.imshow(X_plot, aspect="auto", origin="lower", cmap=cmap)

        ax = _plt.gca()
        for boundary in ends[:-1]:
            ax.hlines(boundary - 0.5,
                      xmin=-0.5,
                      xmax=len(feature_cols) - 0.5,
                      colors="black",
                      linewidth=1)

        _plt.xticks(
            _np.arange(len(feature_cols)),
            feature_cols,
            rotation=90,
            fontsize=fontsize_xtick
        )
        _plt.yticks(midpoints, labels, fontsize=fontsize_ytick, rotation=90)

        _plt.xlabel("Feature")
        _plt.ylabel("Event (grouped by class)")
        _plt.title(
            f"Feature Expression across Events\n(normalized per feature: {normalize})"
        )

        cbar = _plt.colorbar(im, orientation="vertical", pad=0.01)
        cbar.set_label(cbar_label)

        _plt.tight_layout()
        _plt.show()


    def __repr__(self):
        return f"<ChangeEventCollection size={len(self.events)}>"

# ---------------------------------------------------------------------------
# Functions to Process m3c2 File into Change Event Objects
# ---------------------------------------------------------------------------
def process_m3c2_file(m3c2_clustered):
    """
    Load a clustered M3C2 file, compute change events for each cluster,
    write per‑cluster files (point cloud and convex hull), and save a JSON file
    with the change event collection.
    """
    outfolder = os.path.dirname(m3c2_clustered)
    pc_folder = os.path.join(outfolder, "point_clouds")
    obj_folder = os.path.join(outfolder, "convex_hulls")
    os.makedirs(pc_folder, exist_ok=True)
    os.makedirs(obj_folder, exist_ok=True)
    ce_file = os.path.join(outfolder, "change_events.json")

    # Load data using DataHandler
    dh = DataHandler(m3c2_clustered)
    dh.load_las_files()
    df = dh.df
    clusters = np.unique(df["cluster_id"])
    collection = ChangeEventCollection()

    for cluster in clusters:
        cluster_df = df[df["cluster_id"] == cluster]
        event = ChangeEvent.from_cluster(cluster_df, cluster, m3c2_clustered, pc_folder, obj_folder)
        collection.add_event(event)

        # Save point cloud for the cluster
        dh_pc = DataHandler("")
        dh_pc.df = cluster_df
        dh_pc.save_as_las(event.cluster_point_cloud)

        # Save convex hull as an OBJ file
        hull_of_points_to_obj(cluster_df[["X", "Y", "Z"]].values, event.cluster_point_cloud_chull)

    collection.save_to_file(ce_file)
    return collection

def process_m3c2_file_into_change_events(m3c2_clustered):
        outfolder = os.path.dirname(m3c2_clustered)
        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)

        ce_filename = os.path.join(outfolder,"change_events.json")
        merged_ce_file = os.path.join(os.path.dirname(outfolder), "change_events.json")

        if not os.path.isfile(ce_filename):
            try:
                coll = process_m3c2_file(m3c2_clustered)
                # print("Processed change events:")
                # print(coll)
            except Exception as e:
                print("Error processing m3c2 file:", e)
                coll = ChangeEventCollection()
                coll.save_to_file(ce_filename)
        else:
            return True
            # coll = ChangeEventCollection.load_from_file(ce_filename)

        # Load existing merged change events file and attach new events or create a new one
        # print(ce_filename)
        # print(merged_ce_file)
        if os.path.isfile(merged_ce_file):
            # print("in merge")
            collection = ChangeEventCollection.load_from_file(merged_ce_file)
            # Suppose you have another JSON file with new events
            for event in coll.events:
                collection.add_event(event)
            # Save the updated collection
            collection.save_to_file(merged_ce_file)
        else:
            coll.save_to_file(merged_ce_file)


