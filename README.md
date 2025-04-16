# AImon5.0
## Overview

AImon is a comprehensive tool designed for processing and projecting point cloud datasets. It facilitates the generation of range and color images, applies changes based on predefined events, and performs bi-temporal analysis to detect and visualize changes in spatial data.

## Features

- **Point Cloud Projection**: Generate range and color images from point cloud data.
- **Bi-Temporal Analysis**: Compare point clouds from different time frames to detect changes.
- **Change Event Management**: Convert clusters into change events.
- **Data Handling**: Efficiently split, append, and merge LAS/LAZ files.
- **Visualization**: Projected images and change events visualization.

## Installation

### Prerequisites

- **Python 3.11+**
- **Conda** (for environment management)

```

## Creating Conda Environments
To avoid negative interactions between installed packages and version conflicts, you should create a conda environment for each new project. You do so by executing:
```bash
# First, create new environment
$ conda create --name aimon python=3.11

# Then activate the environment using:
$ conda activate aimon

```

Using AImon requires Python 3.11 or higher.
Clone and run this application:

```bash

# Clone this repository
$ git clone https://github.com/3dgeo-heidelberg/aimon.git

# Go into the repository
$ cd aimon

# Installing the release version using pip
$ python -m pip install .

#OR if editable needed
$ python -m pip install -v --editable .

```

# Key Functions
## main.py
Serves as the entry point for the m4dvap processing workflow. It orchestrates the execution of various processing stages, including configuration setup, bi-temporal analysis, and change detection.

Usage:

```
python main.py -c "config" -f "path/to/t1_file.las" "path/to/t2_file.las"
```
