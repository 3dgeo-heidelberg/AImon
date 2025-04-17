import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import jaccard_score, f1_score, accuracy_score, matthews_corrcoef, cohen_kappa_score
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns


######################### Random Forest Classifier #########################

def load_and_prepare_data_for_training(labelled_file):
    """
    Load and prepare the data for training.
    """
    labelled_coll = load_change_events(labelled_file)
    # Extract features from the labeled change events
    features_df = extract_features_for_random_forest(labelled_coll)
    #remove_undefined
    features_df = features_df[features_df['event_type'] != "undefined"]
    features_df = features_df.dropna()
    if features_df.empty:
        raise ValueError("No complete data available after dropping missing values.")
    features_df_without_id = features_df.drop(columns=['object_id',"event_type"])

    X = features_df_without_id.values
    y = features_df['event_type'].values
    return X,y, features_df_without_id.columns

def apply_trained_model_to_unlabelled_data(model, unlabelled_file, output_file):
    # Load the unlabelled data
    unlabelled_coll = load_change_events(unlabelled_file)
    # Extract features from the unlabelled data
    features_unlabelled_df = extract_features_for_random_forest(unlabelled_coll)
    # Remove rows with na values
    features_unlabelled_df = features_unlabelled_df.dropna()
    if features_unlabelled_df.empty:
        raise ValueError("No complete data available after dropping missing values.")
    #Prepare 
    features_unlabelled_df_without_id = features_unlabelled_df.drop(columns=["object_id","event_type"])
    X_unlabelled = features_unlabelled_df_without_id.values
    y_unlabelled_pred = model.predict(X_unlabelled)
    features_unlabelled_df['event_type'] = y_unlabelled_pred
    unlabelled_coll = map_label_to_unlabelled_collection(unlabelled_coll, features_unlabelled_df)
    save_events_to_json(unlabelled_coll, output_file)

def map_label_to_unlabelled_collection(unlabelled_coll, features_unlabelled_df):
    # Mapping from object_id to the chosen cluster label and saving of results
    label_mapping = dict(zip(features_unlabelled_df['object_id'], features_unlabelled_df['event_type']))

    # Update the original events by matching them by object_id.
    for event in unlabelled_coll:
        event = event.to_dict()
        obj_id = event.get('object_id')
        if obj_id in label_mapping:
            event['event_type'] = label_mapping[obj_id]
        else:
            # Mark events dropped due to missing features as None
            event['event_type'] = None
    return unlabelled_coll

def save_events_to_json(unlabelled_coll, output_file):    # Save the updated events to a new JSON file.
    with open(output_file, 'w') as f:
        json.dump(unlabelled_coll, f, indent=4)    


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, param_grid: dict, random_state: 42):
    """
    Train a RandomForest model using GridSearchCV to determine the best hyperparameters.
    """
    # Initialize the RandomForestClassifier with balanced class weighting
    rf = RandomForestClassifier(random_state=random_state, class_weight="balanced")
    
    # Set up and perform grid search with 3-fold cross-validation
    grid_search = GridSearchCV(estimator=rf,
                               param_grid=param_grid,
                               cv=3,
                               n_jobs=-1,
                               verbose=2)
    grid_search.fit(X_train, y_train)
    
    print("Best Hyperparameters:", grid_search.best_params_)
    return grid_search.best_estimator_, grid_search

# =============================================================================
# Model Evaluation and Visualization
# =============================================================================
def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray):
    """
    Plot the confusion matrix using a heatmap.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.show()

def plot_feature_importance(best_rf: RandomForestClassifier, features: pd.Index, X_train_shape: tuple):
    """
    Plot feature importances in descending order.
    """
    importances = best_rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances")
    plt.bar(range(X_train_shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train_shape[1]), features[indices], rotation=90)
    plt.xlim([-1, X_train_shape[1]])
    plt.tight_layout()
    plt.show()

def evaluate_model(best_rf: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray, labels: pd.DataFrame):
    """
    Evaluate the given model by printing the classification report, plotting the confusion matrix
    and feature importances, and printing additional performance metrics.
    """
    # Predict on the test set
    y_pred = best_rf.predict(X_test)
    
    # Print the classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot the confusion matrix and feature importance
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(best_rf, labels, X_test.shape)
    
    # Compute and print additional metrics
    jaccard = jaccard_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    mcc = matthews_corrcoef(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    
    print("Jaccard Score:", jaccard)
    print("Accuracy Score:", accuracy)
    print("F1 Score:", f1)
    print("Matthews Correlation Coefficient:", mcc)
    print("Cohen's Kappa:", kappa)


def save_model(model, file_path: str):
    """
    Save the trained model to disk using joblib.
    """
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

def load_model(file_path: str):
    """
    Load a machine learning model from disk using joblib.
    """
    model = joblib.load(file_path)
    print(f"Model loaded from {file_path}")
    return model

def load_change_events(change_events_file):
    """
    Load the change events from a JSON file.
    """
    if not os.path.exists(change_events_file):
        raise FileNotFoundError(f"{change_events_file} does not exist")
    with open(change_events_file, 'r') as f:
        events = json.load(f)
    return events

def extract_features_for_random_forest(events):
    """
    Convert the change events into a feature matrix.
    This example assumes each event contains an "object_id".
    The features include:
      - Basic event info (e.g., delta_t_hours, number_of_points)
      - Change magnitude statistics (mean, std, min, max, median, quant90, quant95, quant99)
      - Convex hull properties (hull surface area, volume, and ratio)
      - Geometric features from both epochs
    """
    rows = []
    for event in events:
        event = event.to_dict()
        row = {}
        # Basic features
        # row['delta_t_hours'] = event.get('delta_t_hours', np.nan)
        # row['number_of_points'] = event.get('number_of_points', np.nan)
        
        # Change magnitude statistics
        # change_mags = event.get('change_magnitudes', {})
        # for stat in ['mean', 'std', 'min', 'max', 'median', 'quant90', 'quant95', 'quant99']:
        #     row[f'change_{stat}'] = change_mags.get(stat, 0)

        # Change magnitude statistics
        change_mags = event.get('change_magnitudes', {})
        for stat in ['median']:
            row[f'change_{stat}'] = change_mags.get(stat, 0)

        # Convex hull properties
        convex_hull = event.get('convex_hull', {})
        # row['hull_surface_area'] = convex_hull.get('surface_area', 0)
        # row['hull_volume'] = convex_hull.get('volume', 0)
        row['hull_surf_vol_ratio'] = convex_hull.get('surface_area_to_volume_ratio', 0)
        
        # Geometric features from both epochs (if available)
        geom = event.get('geometric_features_both_epochs', {})
        for feat in ['sum_of_eigenvalues', 'omnivariance', 'eigentropy', 'anisotropy',
                     'planarity', 'linearity', 'surface_variation', 'sphericity', 'verticality']:
            row[f'geom_{feat}'] = geom.get(feat, 0)

        geo_epoch_1 = event.get('geometric_features_epoch_1', {})
        for feat in ['sum_of_eigenvalues', 'omnivariance', 'eigentropy', 'anisotropy',
                     'planarity', 'linearity', 'surface_variation', 'sphericity', 'verticality']:
            row[f'geo_epoch_1_{feat}'] = geo_epoch_1.get(feat, 0)

        geo_epoch_2 = event.get('geometric_features_epoch_2', {})
        for feat in ['sum_of_eigenvalues', 'omnivariance', 'eigentropy', 'anisotropy',
                     'planarity', 'linearity', 'surface_variation', 'sphericity', 'verticality']:
            row[f'geo_epoch_2_{feat}'] = geo_epoch_2.get(feat, 0)
        
        # Keep the unique identifier
        row['object_id'] = event.get('object_id')
        row["event_type"] = event.get('event_type', "undefined")
        
        rows.append(row)
    df = pd.DataFrame(rows)
    return df





####################### UMAP ###########################
def extract_features_umap(events):
    """
    Convert the change events into a feature matrix.
    This example assumes each event contains an "object_id".
    The features include:
      - Basic event info (e.g., delta_t_hours, number_of_points)
      - Change magnitude statistics (mean, std, min, max, median, quant90, quant95, quant99)
      - Convex hull properties (hull surface area, volume, and ratio)
      - Geometric features from both epochs
    """
    rows = []
    for event in events:
        event = event.to_dict()
        row = {}
        # Basic features
        # row['delta_t_hours'] = event.get('delta_t_hours', np.nan)
        # row['number_of_points'] = event.get('number_of_points', np.nan)
        
        # Change magnitude statistics
        change_mags = event.get('change_magnitudes', {})
        for stat in ['mean', 'std', 'min', 'max', 'median', 'quant90', 'quant95', 'quant99']:
            row[f'change_{stat}'] = change_mags.get(stat, 0)

        # Convex hull properties
        convex_hull = event.get('convex_hull', {})
        # row['hull_surface_area'] = convex_hull.get('surface_area', 0)
        # row['hull_volume'] = convex_hull.get('volume', 0)
        row['hull_surf_vol_ratio'] = convex_hull.get('surface_area_to_volume_ratio', 0)
        
        # Geometric features from both epochs (if available)
        geom = event.get('geometric_features_both_epochs', {})
        for feat in ['sum_of_eigenvalues', 'omnivariance', 'eigentropy', 'anisotropy',
                     'planarity', 'linearity', 'surface_variation', 'sphericity', 'verticality']:
            row[f'geom_{feat}'] = geom.get(feat, 0)

        geo_epoch_1 = event.get('geometric_features_epoch_1', {})
        for feat in ['sum_of_eigenvalues', 'omnivariance', 'eigentropy', 'anisotropy',
                     'planarity', 'linearity', 'surface_variation', 'sphericity', 'verticality']:
            row[f'geo_epoch_1_{feat}'] = geo_epoch_1.get(feat, 0)

        geo_epoch_2 = event.get('geometric_features_epoch_2', {})
        for feat in ['sum_of_eigenvalues', 'omnivariance', 'eigentropy', 'anisotropy',
                     'planarity', 'linearity', 'surface_variation', 'sphericity', 'verticality']:
            row[f'geo_epoch_2_{feat}'] = geo_epoch_2.get(feat, 0)
        
        # Keep the unique identifier
        row['object_id'] = event.get('object_id')
        row["event_type"] = event.get('event_type', "undefined")
        
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

def plot_umap_clusters(X_umap, labels):
    plt.figure(figsize=(5, 4))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        plt.scatter(X_umap[mask, 0], X_umap[mask, 1], label=f'Cluster {label}', s=1)
    plt.legend()
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.show()

def load_and_prepare_data_for_UMAP(change_events_file):
    """
    Load and prepare data for UMAP dimensionality reduction.
    """
    # Load the change events
    events = load_change_events(change_events_file)# Extract features into a DataFrame
    features_df = extract_features_umap(events)

    # Handle missing values by dropping them
    features_df = features_df.dropna()
    if features_df.empty:
        raise ValueError("No complete data available after dropping missing values.")
    # Separate the identifier from features
    features_df_without_id = features_df.drop(columns=["object_id","event_type"])

    # Standardize the feature set for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df_without_id.values)

    return features_df_without_id, features_df, X_scaled, events



############# Rule based classification ###############

def extract_features_all(events):
    """
    Convert the change events into a feature matrix.
    This example assumes each event contains an "object_id".
    The features include:
      - Basic event info (e.g., delta_t_hours, number_of_points)
      - Change magnitude statistics (mean, std, min, max, median, quant90, quant95, quant99)
      - Convex hull properties (hull surface area, volume, and ratio)
      - Geometric features from both epochs
    """
    rows = []
    for event in events:
        event = event.to_dict()
        row = {}
        # Basic features
        row['delta_t_hours'] = event.get('delta_t_hours', np.nan)
        row['number_of_points'] = event.get('number_of_points', np.nan)
        
        # Change magnitude statistics
        change_mags = event.get('change_magnitudes', {})
        for stat in ['mean', 'std', 'min', 'max', 'median', 'quant90', 'quant95', 'quant99']:
            row[f'change_{stat}'] = change_mags.get(stat, 0)

        # Convex hull properties
        convex_hull = event.get('convex_hull', {})
        row['hull_surface_area'] = convex_hull.get('surface_area', 0)
        row['hull_volume'] = convex_hull.get('volume', 0)
        row['hull_surf_vol_ratio'] = convex_hull.get('surface_area_to_volume_ratio', 0)
        
        # Geometric features from both epochs (if available)
        geom = event.get('geometric_features_both_epochs', {})
        for feat in ['sum_of_eigenvalues', 'omnivariance', 'eigentropy', 'anisotropy',
                     'planarity', 'linearity', 'surface_variation', 'sphericity', 'verticality']:
            row[f'geom_{feat}'] = geom.get(feat, 0)

        geo_epoch_1 = event.get('geometric_features_epoch_1', {})
        for feat in ['sum_of_eigenvalues', 'omnivariance', 'eigentropy', 'anisotropy',
                     'planarity', 'linearity', 'surface_variation', 'sphericity', 'verticality']:
            row[f'geo_epoch_1_{feat}'] = geo_epoch_1.get(feat, 0)

        geo_epoch_2 = event.get('geometric_features_epoch_2', {})
        for feat in ['sum_of_eigenvalues', 'omnivariance', 'eigentropy', 'anisotropy',
                     'planarity', 'linearity', 'surface_variation', 'sphericity', 'verticality']:
            row[f'geo_epoch_2_{feat}'] = geo_epoch_2.get(feat, 0)
        
        # Keep the unique identifier
        row['object_id'] = event.get('object_id')
        row["event_type"] = event.get('event_type', "undefined")
        
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

def classify_event(row, rules):
    """
    Classify an event based on a set of rule conditions provided in 'rules'.
    
    The rules dictionary should have the following structure:
    
      rules = {
          "label_name": {
              "feature1": {"min": value, "max": value},
              "feature2": {"min": value}
              ...
          },
          ... 
      }
    
    For each label, the event is considered a match if for every specified feature:
       - The feature value is not NaN.
       - If a "min" is specified, the value is >= the min.
       - If a "max" is specified, the value is <= the max.
       
    The function returns the first label (in iteration order) which all its conditions are met.
    If none match, "unclassified" is returned.
    """
    for label, conditions in rules.items():
        meets_conditions = True
        for feature, thresholds in conditions.items():
            value = row.get(feature, np.nan)
            if pd.isna(value):
                meets_conditions = False
                break
            if "min" in thresholds and value < thresholds["min"]:
                meets_conditions = False
                break
            if "max" in thresholds and value > thresholds["max"]:
                meets_conditions = False
                break
        if meets_conditions:
            return label
    return "unclassified"


def classify_rule_based(features_df, events, classification_rules):
    # Apply the generic classification function using the rules in classification_rules
    features_df['event_type'] = features_df.apply(lambda row: classify_event(row, classification_rules), axis=1)
    events = map_label_to_unlabelled_collection(events, features_df)
    return events, features_df

########################### Filter events based on rules ###########################
def filter_events(features_df,events, filter_rules):
    features_df = features_df[features_df.apply(lambda row: filter_event(row, filter_rules), axis=1)]
    events = map_label_to_unlabelled_collection(events, features_df)
    return events, features_df

def filter_event(row, rules):
    """
    Filter an event based on a set of rule conditions.
    
    If the event is already classified—that is, if the 'classification' key exists 
    and is not "unclassified"—then the row is kept as is.
    
    Otherwise, the function checks the rules (a dictionary structured as follows):
    
      rules = {
          "label_name": {
              "feature1": {"min": value, "max": value},
              "feature2": {"min": value},
              ...
          },
          ... 
      }
      
    For each label, the event meets the criteria if, for every specified feature:
       - The feature's value is not NaN.
       - If a "min" is specified, the value is >= min.
       - If a "max" is specified, the value is <= max.
       
    If any rule is satisfied, the function sets the event's classification to that label 
    and returns True (meaning the event should be kept).
    
    If no rules apply, the function returns False so that the row can be removed.
    
    Returns:
        bool: True if the event is classified (or already had a classification) and should 
              be kept, False otherwise.
    """
    # If the row already carries a valid classification, preserve it.
    if row.get('classification', 'unclassified') != 'unclassified':
        return True
    
    # Check each rule.
    for label, conditions in rules.items():
        meets_conditions = True
        for feature, thresholds in conditions.items():
            value = row.get(feature, np.nan)
            if pd.isna(value):
                meets_conditions = False
                break
            if "min" in thresholds and value < thresholds["min"]:
                meets_conditions = False
                break
            if "max" in thresholds and value > thresholds["max"]:
                meets_conditions = False
                break
        if meets_conditions:
            # Assign the new classification.
            row['classification'] = label
            return True
            
    # If no conditions are met, return False (event will be filtered out).
    return False