# AImon5.0
<img src="img/AImon_logo.png?raw=true" alt="logo" style="width:500px;"/>

<!--[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)-->

## Overview

AImon is a comprehensive tool designed for processing and projecting point cloud datasets. It facilitates the generation of range and color images, applies changes based on predefined events, and performs bi-temporal analysis to detect and visualize changes in spatial data.

## 🔨 Methods provided by AImon

- **Point Cloud Projection**: Generate range and color images from point cloud data.
- **Bi-Temporal Analysis**: Compare point clouds from different time frames to detect changes.
- **Change Event Management**: Convert clusters into change events.
- **Data Handling**: Efficiently split, append, and merge LAS/LAZ files.
- **Visualization**: Projected images and change events visualization.

## 🎮 Examples
|                                                                                                                                                                               |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Example 1: Rule based classification of change events <br> <a href="demo/classification_of_change_events_rule_based.ipynb"><img src="img/AImon_logo.png" width=500></a>                     |
| Example 2: Unsupervised classification of change events <br> <a href="demo/classification_of_change_events_unsupervised.ipynb"><img src="img/AImon_logo.png" width=500></a>                   |
| Example 3: Random forest classification of change events <br> <a href="demo/classification_of_change_events_using_random_forest_classifier.ipynb"><img src="img/AImon_logo.png" width=500></a> |
| Example 4: Rule based filtering of change events <br> <a href="demo/filtering_of_change_events_rule_based.ipynb"><img src="img/AImon_logo.png" width=500></a>                          |
<!--
| [![Example #](img/...)](link)          |
-->

## 💻 Installation with a Conda environment

To avoid negative interactions between installed packages and version conflicts, a conda environment should be created for each new project. Follow the three next steps:

1. Create a new environment and activate it.
```bash
conda create --name aimon python=3.11 -y
conda activate aimon

```

2. Clone this repository and navigate to the aimon folder
```bash
git clone https://github.com/3dgeo-heidelberg/aimon.git
cd aimon
```

3. Install the release version using pip
    1. Regular installation

    ```bash
    python -m pip install .
    ```

    2. Editable mode

    ```bash
    python -m pip install -v --editable .
    ```


# Key Functions
## main.py
Serves as the entry point for the m4dvap processing workflow. It orchestrates the execution of various processing stages, including configuration setup, bi-temporal analysis, and change detection.

Usage:

```
python main.py -c "config" -f "path/to/t1_file.las" "path/to/t2_file.las"
```
