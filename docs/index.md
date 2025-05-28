<img src="img/3DGeo_Logo_300dpi.png" alt="3DGeo Logo" title="3DGeo Logo" width=150 style="float: right;">
<br>

# Welcome to AImon5.0
<a href="https://www.geog.uni-heidelberg.de/en/institute/geoinformatics/3dgeo-research-group/projects-of-the-3dgeo-research-group/aimon50">
  <img src="img/AImon_logo.png?raw=true" alt="AImon 5.0 logo" style="width:500px;" />
</a>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📑 Research project
<div style="text-align: justify">
Modern permanently installed laser scanners can deliver sub‐hourly point clouds, opening the door to early warning of surface deformations. Current workflows struggle to keep pace with such data volumes in near real-time. Vegetation and occlusions in forested or complex terrain further degrade ground‐point coverage, undermining the reliability of change estimates. As a result, there remains a critical need for efficient, robust processing strategies that can detect and quantify subtle surface shifts.
</div>

<!-- Real-time monitoring of gravitational mass movements for critical infrastructure risk management with AI-assisted 3D metrology -->

<!-- ### 🎯 Overall objective of the project -->
### Overall objective
<div style="text-align: justify">
Our environment and the Earth's surface are constantly changing, and global warming and climate change are accelerating the pace and magnitude of these changes. As a result, geohazards - triggered by natural events or human activities - are becoming more frequent. For example, intense and prolonged rainfall is increasingly causing landslides and rockfalls that threaten local populations and critical infrastructure such as railways and roads, with serious economic consequences.

A key tool for integrated risk management is access to relevant 4D geospatial information - accurate 3D data with high temporal resolution - acquired through near real-time, permanent or on-demand monitoring. Permanently installed autonomous laser scanning (PLS) systems have shown great potential for monitoring hazard zones, producing billions of measurements daily. Early computational methods exist to analyze this data, but to make it operational, a new interface is needed to bridge application needs with 4D data collection and analysis.

This interface will connect stakeholder expertise with autonomous PLS systems and data archives using AI and 4D analysis. It will enable the operational use of PLS for risk monitoring - detecting and tracking relevant events such as slope activity in real time. For the first time, stakeholders will be able to use PLS for continuous hazard monitoring.
</div>

<!-- ### 📌 Study site -->
### Study site
<div style="text-align: justify">
The goal of this project is to bridge the gap between research and practice. While key methods for multi-temporal analysis and subtopics such as uncertainty in change detection have been developed, this project focuses on refining and extending them for practical, application-oriented use. The 
<a href="https://www.geog.uni-heidelberg.de/en/3dgeo">3DGeo Research Group Heidelberg</a> developed computer-based methods for automatic information extraction and visualization from 4D-PLS data. The study site is located in <a href="https://maps.app.goo.gl/1k6VpK1gXzoZ1TLJ9">Trier, Germany</a> (Fig. 1), at the <a href="https://maps.app.goo.gl/JLSZRwxY1ppR6zbr7">Trierer Augenscheiner</a> (Fig. 2).
</div>


|                                                                                                                                                                                                                   |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|<img src="img/study_site.png?raw=true" alt="Trier Map"/> <br/> Study site of the AImon5.0 project located in Trier in Germany (red dot). |


<!-- ### 🧭 Approach Development -->
### Approach development
<div style="text-align: justify">
The methods are particularly suitable for operational use and adapted in order to deliver reliable and timely results. Automated information extraction represents a central interface between the PLS system in the field, the quality-assured change information, and the end users. Scientifically, we investigate and combine two complementary concepts that can integrate expert knowledge into automated data analysis:<br>

&nbsp;&nbsp;&nbsp;&nbsp; 1. <u>Top-down approach via a knowledge- and rule-based classification of changes</u>: In that case, the users know exactly which events they want to find in the data streams and how these processes (e.g. rockfall) are defined in their sequence. A methodology and data management for fast and accurate searches must be developed and evaluated;<br>
&nbsp;&nbsp;&nbsp;&nbsp; 2. <u>Data-driven approach using AI</u>: Machine learning methods find relevant change events after a user-controlled training phase and present them to the experts for evaluation. The users do not know in advance how the events, possibly also overlaid processes, are represented in the data, but can evaluate relevant from non-relevant events and thus train an AI model.<br>

For this second approach, research must be carried out into how the state-of-the-art point cloud-based deep learning models can be trained quickly and as automatically as possible in the background and how the hyperparameters can be optimized. For the visualization of the detected and classified changes, it must be determined - in coordination with end users and the PLS operator - which abstraction levels and visualization forms are best suited for certain tasks and also specified reaction times. In contrast to visualization in 2D and 3D (e.g. in GIS or dashboards), fundamental research must be carried out for the visualization of 4D processes in PLS data due to a lack of existing methods.
</div>


## ⚙️ Work packages
<div style="text-align: justify">
The following methods were developed by the 
<a href="https://www.geog.uni-heidelberg.de/en/3dgeo">3DGeo Research Group</a>.

A change event is characterized by different attributes, each representing a measurable dimension of the change. The diagram of Fig. 3 highlights the modular structure of change events, emphasizing how temporal and spatial metrics combine to define and categorize observed changes. A classifier assigns an event type to the change based on the characteristics, enabling semantic interpretation of what kind of event occurred (e.g., gravitational mass movement, change in vegetation, etc).<br>
<img src="img/change_event_model.png?raw=true" alt="logo" style="width:500px;"/>
</div>

**Research target 1 - Hierarchical classification of detected change:**  
<div style="text-align: justify">Developing new methods and tools to automatically extract relevant change information from the last two point clouds. We analyse different types of changes in the terrain (e.g. rockfall events, movements or erosion processes) fully automatically by delimiting them in terms of time and space. Five different steps:
</div>
- 1.1: Rule-based change classification
- 1.2: [Hierarchical analysis](https://doi.org/10.5445/IR/1000180377)
- 1.3: ML/DL change classification
- 1.4: Derivation of adaptive workflows
- 1.5: Continuous integration in [py4dgeo](https://github.com/3dgeo-heidelberg/py4dgeo/tree/main)

**Research target 2: Visualization of classified change events:**
<div style="text-align: justify">
Development of new methodologies and tools for the visualization of the detected terrain changes for use by end users. Three different steps:
</div>
- 2.1: Selection of relevant changes
- 2.2: 2D GIS layer
- 2.3: 3D objects

## 💡 Possible applications
- **Point Cloud Projection**: Generate range and color images from point cloud data.
- **Bi-Temporal Analysis**: Compare point clouds from different time frames to detect changes.
- **Change Event Management**: Convert clusters into change events.
- **Data Handling**: Efficiently split, append, and merge LAS/LAZ files.
- **Visualization**: Projected images and change events visualization.
<!-- </div> -->

## 🎮 Examples
|                                                                                                                                                                                                                   |                                                                                                                                                                                                                         |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Example 1: Main AImon5.0 monitoring pipeline <br>  <br> <a href="main.html"><img src="img/main.png" width=500></a> | Example 2: Adaptive monitoring <br> <a href="adaptive_monitoring.html"><img src="img/adaptive_monitoring.gif" width=500></a> |
| Example 3: Rule-based classification of change events <br> <a href="classification_of_change_events_rule_based.html"><img src="img/classified_rule_based.png" width=500></a> | Example 4: Rule-based filtering of change events <br> <a href="filtering_of_change_events_rule_based.html"><img src="img/filtered_rule_based.png" width=500></a>                                              |
| Example 5: Manually labelled dataset for random forest training <br> <a href="classification_of_change_events_using_random_forest_classifier.html"><img src="img/labelled_change_events.png" width=500></a> | Example 6: Random forest classification on prediction dataset <br> <a href="classification_of_change_events_using_random_forest_classifier.html"><img src="img/classified_using_random_forest.png" width=500></a> |






<!-- |                                                                                                                                                                                                                   |                                                                                                                                                                                                                         |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Example 1: Rule-based classification of change events <br> <a href="classification_of_change_events_rule_based.html"><img src="img/classified_rule_based.png" width=500></a>                                | Example 2: Rule-based filtering of change events <br> <a href="filtering_of_change_events_rule_based.html"><img src="img/filtered_rule_based.png" width=500></a>                                              |
| Example 3: Manually labelled dataset for random forest training <br> <a href="classification_of_change_events_using_random_forest_classifier.html"><img src="img/labelled_change_events.png" width=500></a> | Example 4: Random forest classification on prediction dataset <br> <a href="classification_of_change_events_using_random_forest_classifier.html"><img src="img/classified_using_random_forest.png" width=500></a> |
| Example 5: Adaptive monitoring <br> <a href="adaptive_monitoring.html"><img src="img/adaptive_monitoring.gif" width=500></a> | -->




## 🛠️ Key functions
<div style="text-align: justify">
<!-- ### Using main.py file -->
Serves as the entry point for the AImon5.0 processing workflow. It orchestrates the execution of various processing stages, including configuration setup, bi-temporal analysis, and change detection.
</div>


<!-- <h3> 📚 Publications </h3> -->
## 📚 Publications
<details>
<summary>Journal and conference</summary>
```
@article{Tabernig2025,
  author       = {Tabernig, Ronald and Albert, William and Weiser, Hannah and H{\"{o}}fle, Bernhard},
  year         = {2025},
  title        = {A hierarchical approach for near real-time 3D surface change analysis of permanent laser scanning point clouds},
  doi          = {10.5445/IR/1000180377},
  pagetotal    = {9}
}
```

```
@article{WeiserHoefle2025,
  author       = {Weiser, Hannah and H{\"{o}}fle, Bernhard},
  journal      = {EarthArXiv},
  year         = {2025},
  title        = {Advancing vegetation monitoring with virtual laser scanning of dynamic scenes (VLS-4D): Opportunities, implementations and future perspectives},
  doi          = {10.31223/X51Q5V},
}
```

```
@article{Albert2025,
  author       = {Albert, William and Weiser, Hannah and Tabernig, Ronald and H{\"{o}}fle, Bernhard},
  journal      = {ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
  year         = {2025},
  title        = {Wind during terrestrial laser scanning of trees: Simulation-based assessment of effects on point cloud features and leaf-wood classification},
  doi          = {},
  pagetotal    = {8}
}
```
</details>

<details>
<summary>Software</summary>
```
@software{Tabernig_VAPC_-_Voxel_2024,
  author = {Tabernig, Ronald and Albert, William and Weiser, Hannah and H{\"{o}}fle, Bernhard},
  license = {MIT},
  month = dec,
  title = {{VAPC - Voxel Analysis for Point Clouds}},
  url = {https://github.com/3dgeo-heidelberg/vapc},
  version = {0.0.1},
  year = {2024}
}
```
</details>

<details>
<summary>Abstract</summary>
```
@inproceedings{AlbertSummerSchool2024,
  author       = {Albert, William},
  title        = {Considering wind effects in LiDAR simulation-based machine learning for point cloud classification in forests},
  booktitle    = {Sensing Mountains. Innsbruck Summer School of Alpine Research 2024 – Close Range Sensing Techniques in Alpine Terrain},
  editor       = {Rutzinger, M. and Anders, K. and Eltner, A. and Gaevert, C. and Höfle, B. and Lindenbergh, R. and Mayr, A. and Nopens, S. and Oude Elberink, S. and Pirotti, F.},
  pages        = {19--22},
  year         = {2024},
  address      = {Innsbruck},
  publisher    = {innsbruck university press (IUP)},
  doi          = {10.15203/99106-137-3},
  note         = {Abstract},
  url          = {https://doi.org/10.15203/99106-137-3}
}
```

```
@inproceedings{TabernigSummerSchool2024,
  author       = {Tabernig, Ronald},
  title        = {Simulating laser scanning of dynamic virtual 3D scenes for improved 4D point cloud based topographic change analysis},
  booktitle    = {Sensing Mountains. Innsbruck Summer School of Alpine Research 2024 – Close Range Sensing Techniques in Alpine Terrain},
  editor       = {Rutzinger, M. and Anders, K. and Eltner, A. and Gaevert, C. and Höfle, B. and Lindenbergh, R. and Mayr, A. and Nopens, S. and Oude Elberink, S. and Pirotti, F.},
  pages        = {134--137},
  year         = {2024},
  address      = {Innsbruck},
  publisher    = {innsbruck university press (IUP)},
  doi          = {10.15203/99106-137-3},
  note         = {Abstract},
  url          = {https://doi.org/10.15203/99106-137-3}
}
```

```
@inproceedings{Tabernig2024,
  author       = {Tabernig, Ronald and Zahs, Vivien and Weiser, Hannah and H{\"{o}}fle, Bernhard},
  title        = {Simulating 4D scenes of rockfall and landslide activity for improved 3D point cloud-based change detection using machine learning},
  booktitle    = {EGU General Assembly 2024},
  year         = {2024},
  address      = {Vienna, Austria},
  month        = apr,
  note         = {EGU24-1613},
  doi          = {10.5194/egusphere-egu24-1613},
  url          = {https://doi.org/10.5194/egusphere-egu24-1613}
}
```

```
@inproceedings{Hofle2024,
  author       = {H{\"{o}}fle, B. and Tabernig, R. and Zahs, V. and Esmorís, A.M. and Winiwarter, L. and Weiser, H.},
  title        = {Machine-learning based 3D point cloud classification and multitemporal change analysis with simulated laser scanning data using open source scientific software},
  booktitle    = {EGU General Assembly 2024},
  volume       = {EGU24},
  pages        = {1--2},
  year         = {2024},
  doi          = {10.5194/egusphere-egu24-1261},
  note         = {Abstract},
  url          = {https://doi.org/10.5194/egusphere-egu24-1261}
}

```
</details>

## 📂 Credits
<!-- <h3> 🐍 Documentation of software usage </h3> -->
**Software usage**

As a starting point, please have a look to the Jupyter Notebooks available listed in the top left corner of the page.


<!-- <h3> 📑 Citation </h3> -->
**Citation**

Please cite AImon5.0 when using it in your research.

```
@software{AImon5.0,
author = {AImon5.0 Development Core Team},
title = {AImon5.0: tool for 3D point cloud processing and projection},
journal = {},
year = {2025},
number = {},
volume = {},
doi = {},
url = {https://github.com/3dgeo-heidelberg/AImon},
}
```

<!-- <h3> 💟 Funding / Acknowledgements </h3> -->
**Funding / acknowledgements**

This work is funded by the BMBF AIMon5.0: Project no. 02WDG1696 and by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation). The Federal Ministry of Education and Research (BMBF) is funding the AIMon5.0 project within the funding measure ’Digital GreenTech – Environmental Engineering meets Digitalisation’ as part of the ’Research for Sustainability (FONA) Strategy’.

<!-- <h3> 🔔 Contact / Bugs / Feature Requests </h3> -->
**Contact / bugs / feature requests**

<div style="text-align: justify">
You think you have found a bug or have specific request for a new feature? Please open a new issue in the online code repository on Github. Also for general questions please use the issue system.

Scientific requests can be directed to the 
<a href="https://www.geog.uni-heidelberg.de/en/3dgeo">3DGeo Research Group Heidelberg</a> and its respective members.
</div>


<!-- <h3> 📜 License </h3> -->
**License**

This is licensed under the [MIT license](LICENSE.md).


