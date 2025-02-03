# m4dvap: Modular 4-Dimensional Visualization and Analysis for Point clouds

## Overview

m4dvap is a comprehensive tool designed for processing and projecting point cloud datasets. It facilitates the generation of range and color images, applies changes based on predefined events, and performs bi-temporal analysis to detect and visualize changes in spatial data.

## Features

- **Point Cloud Projection**: Generate range and color images from point cloud data.
- **Bi-Temporal Analysis**: Compare point clouds from different time frames to detect changes.
- **Change Event Management**: Convert clusters into change events.
- **Data Handling**: Efficiently split, append, and merge LAS/LAZ files.
- **Visualization**: Projected images and change events visualization.

## Installation

### Prerequisites

- **Python 3.10+**
- **Conda** (for environment management)

### Setting Up the Conda Environment

1. **Clone the Repository**

```bash
$ git clone https://github.com/3dgeo-heidelberg/m4dvap.git
$ cd m4dvap

# Install a Conda Environment
$ conda env create -f environment.yml

# Activate environment 
$ conda activate m4dvap

```

# Key Functions
## main.py
Serves as the entry point for the m4dvap processing workflow. It orchestrates the execution of various processing stages, including configuration setup, bi-temporal analysis, and change detection.

Usage:

```
python main.py -c "config" -f "path/to/t1_file.las" "path/to/t2_file.las"
```
