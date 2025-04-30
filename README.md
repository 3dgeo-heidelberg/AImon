# Welcome to AImon5.0
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
## Using main.py file
Serves as the entry point for the AImon processing workflow. It orchestrates the execution of various processing stages, including configuration setup, bi-temporal analysis, and change detection.

Usage from the main repository folder:
```bash
python cd src/aimon/main.py -c "<path/to/config_file.json>" -f "<path/to/t1_point_cloud.las>" "<path/to/t2_point_cloud.las>"
```

## 🐍 Documentation of software usage
As a starting point, please have a look to the [Jupyter Notebooks](demo) available in the repository 


## 📑 Citation
Please cite AImon5.0 when using it in your research and reference the appropriate release version.

<!-- TODO: All releases of py4dgeo are listed on Zenodo where you will find the citation information including DOI. -->

```
article{AImon5.0,
author = {AImon5.0 Development Core Team}
title = {AImon5.0: tool for 3D point cloud processing and projection},
journal = {},
year = {2025},
number = {},
volume = {},
doi = {},
url = {https://github.com/3dgeo-heidelberg/AImon},
}
 ```

## 💟 Funding / Acknowledgements
TODO: Add funding and acknoledgment

## 🔔 Contact / Bugs / Feature Requests
You think you have found a bug or have specific request for a new feature? Please open a new issue in the online code repository on Github. Also for general questions please use the issue system.

Scientific requests can be directed to the [3DGeo Research Group Heidelberg](https://uni-heidelberg.de/3dgeo) and its respective members.

## 📜 License
See [LICENSE.md](LICENSE.md).


## 📚 Literature
* Paper 1
* Paper 2
* ...
