# Welcome to AImon<sup>5.0</sup>
<img src="img/AImon_logo.png?raw=true" alt="logo" style="width:500px;"/>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Research project definition
### Project title
Integrated, quality-assured real-time assistance system for risk management of infrastructures and natural hazards using autonomous 4D detection, 3D-metrology and artificial intelligence.

### Overall objective of the project
Our environment and the Earth's surface are constantly changing, and global warming and climate change are accelerating the pace and magnitude of these changes. As a result, geohazards - triggered by natural events or human activities - are becoming more frequent. For example, intense and prolonged rainfall is increasingly causing landslides and rockfalls that threaten local populations and critical infrastructure such as railways and roads, with serious economic consequences.

A key tool for integrated risk management is access to relevant 4D geospatial information - accurate 3D data with high temporal resolution - acquired through near real-time, permanent or on-demand monitoring. Permanently installed autonomous laser scanning (PLS) systems have shown great potential for monitoring hazard zones, producing billions of measurements daily. Early computational methods exist to analyze this data, but to make it operational, a new interface is needed to bridge application needs with 4D data collection and analysis.

This interface will connect stakeholder expertise with autonomous PLS systems and data archives using AI and 4D analysis. It will enable the operational use of PLS for risk monitoring - detecting and tracking relevant events such as slope activity in real time. For the first time, stakeholders will be able to use PLS for continuous hazard monitoring.

The goal of this project is to bridge the gap between research and practice. While key methods for multi-temporal analysis and subtopics such as uncertainty in change detection have been developed, this project focuses on refining and extending them for practical, application-oriented use. The [3DGEO](https://www.geog.uni-heidelberg.de/3dgeo/index_en.html) research group developed computer-based methods for automatic information extraction and visualization from 4D-PLS data. The study site is located in [Trier (Germany)](https://maps.app.goo.gl/1k6VpK1gXzoZ1TLJ9) (Fig 1), at the [Trierer Augenscheiner](https://maps.app.goo.gl/JLSZRwxY1ppR6zbr7) (Fig 2).

|                                                                                                                                                                                                                   |                                                                                                                                                                                                                         |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <br> <a href="demo/classification_of_change_events_rule_based.ipynb"><img src="img/Trier_map.png?raw=true" width=300></a>                                |<img src="img/Trier_pic.png?raw=true" alt="Trier Map"/> <br/> Study site of the AImon<sup>5.0</sup> project located in Trier in Germany (red dot). |



The methods are particularly suitable for operational use and adapted in order to deliver reliable and timely results. Automated information extraction represents a central interface between the PLS system in the field, the quality-assured change information, and the end users. Scientifically, we investigate and combine two complementary concepts that can integrate expert knowledge into automated data analysis: 

1. <u>Top-down approach via a knowledge- and rule-based classification of changes</u>: In that case, the users know exactly which events they want to find in the data streams and how these processes (e.g. rockfall) are defined in their sequence. A methodology and data management for fast and accurate searches must be developed and evaluated;

2. <u>Data-driven approach using AI</u>: Machine learning methods find relevant change events after a user-controlled training phase and present them to the experts for evaluation. The users do not know in advance how the events, possibly also overlaid processes, are represented in the data, but can evaluate relevant from non-relevant events and thus train an AI model.

For this second approach, research must be carried out into how the state-of-the-art point cloud-based deep learning models can be trained quickly and as automatically as possible in the background and how the hyperparameters can be optimized. For the visualization of the detected and classified changes, it must be determined - in coordination with end users and the PLS operator - which abstraction levels and visualization forms are best suited for certain tasks and also specified reaction times. In contrast to visualization in 2D and 3D (e.g. in GIS or dashboards), fundamental research must be carried out for the visualization of 4D processes in PLS data due to a lack of existing methods.


## 🔨 Methods developed as part of the AImon<sup>5.0</sup> project
The following methods were developed by the [3DGEO](https://www.geog.uni-heidelberg.de/3dgeo/index_en.html) research group.

<img src="img/change_event_model.png?raw=true" alt="logo" style="width:500px;"/>


- **Research target 1 - Hierarchical classification of detected change:**  Developing new methods and tools to automatically extract relevant change information from the two last point clouds. We analyse different types of changes in the terrain (e.g. rockfall events, movements or erosion processes) fully automatically by delimiting them in terms of time and space. Five different steps:
    - 1.1: Rule based change classification
    - 1.2: Hierarchical Analysis
    - 1.3: ML/DL Change Classification
    - 1.4: Derivation of adaptive workflows
    - 1.5: Continuous integration py4dgeo

- **Research target 2: Visualization of classified change events**: Development of new methodologies and tools for the visualization of the detected terrain changes from WP 5 and WP 6 for use by end users. Three different steps:
    - 2.1: Selection of relevant changes
    - 2.2: 2D GIS layer
    - 2.3: 3D objects

### Possible applications:
- **Point Cloud Projection**: Generate range and color images from point cloud data.
- **Bi-Temporal Analysis**: Compare point clouds from different time frames to detect changes.
- **Change Event Management**: Convert clusters into change events.
- **Data Handling**: Efficiently split, append, and merge LAS/LAZ files.
- **Visualization**: Projected images and change events visualization.

## 🎮 Examples
|                                                                                                                                                                                                                   |                                                                                                                                                                                                                         |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Example 1: Rule based classification of change events <br> <a href="demo/classification_of_change_events_rule_based.ipynb"><img src="img/classified_rule_based.png" width=500></a>                                | Example 2: Rule based filtering of change events <br><br> <a href="demo/filtering_of_change_events_rule_based.ipynb"><img src="img/filtered_rule_based.png" width=500></a>                                              |
| Example 3: Manually labelled dataset for random forest training <br> <a href="demo/classification_of_change_events_using_random_forest_classifier.ipynb"><img src="img/labelled_change_events.png" width=500></a> | Example 4: Random forest classification on prediction dataset <br> <a href="demo/classification_of_change_events_using_random_forest_classifier.ipynb"><img src="img/classified_using_random_forest.png" width=500></a> |
<!--
| Example 5: Unsupervised classification of change events <br> <a href="demo/classification_of_change_events_unsupervised.ipynb"><img src="img/AImon_logo.png" width=500></a>                                       |
-->

## 💻 Installation with a Conda environment

To avoid negative interactions between installed packages and version conflicts, a conda environment should be created for each new project. Follow the three next steps:

1. Create a new environment and activate it.
```bash
conda create --name aimon python=3.11 -y
conda activate aimon

```

2. Clone this repository and navigate to the main folder **aimon**
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
Serves as the entry point for the AImon<sup>5.0</sup> processing workflow. It orchestrates the execution of various processing stages, including configuration setup, bi-temporal analysis, and change detection.

Usage from the main repository folder:
```bash
python cd src/aimon/main.py -c "<path/to/config_file.json>" -f "<path/to/t1_point_cloud.las>" "<path/to/t2_point_cloud.las>"
```

## 🐍 Documentation of software usage
As a starting point, please have a look to the [Jupyter Notebooks](demo) available in the repository 


## 📑 Citation
Please cite AImon<sup>5.0</sup> when using it in your research and reference the appropriate release version.

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

I just add some text here to test stuff
## 🔔 Contact / Bugs / Feature Requests
You think you have found a bug or have specific request for a new feature? Please open a new issue in the online code repository on Github. Also for general questions please use the issue system.

Scientific requests can be directed to the [3DGeo Research Group Heidelberg](https://uni-heidelberg.de/3dgeo) and its respective members.

## 📜 License
See [LICENSE.md](LICENSE.md).


## 📚 Literature
* Paper 1
* Paper 2
* ...
