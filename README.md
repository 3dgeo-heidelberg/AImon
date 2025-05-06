# AImon<sup>5.0</sup>

## 📖 About

The [3DGeo Research Group](https://www.geog.uni-heidelberg.de/en/3dgeo) focuses on advancing geohazard monitoring through the use of 4D geospatial data. With global climate change accelerating environmental risks like landslides and rockfalls, our work leverages autonomous laser scanning (PLS) systems to capture real-time 3D data, enabling proactive risk management. We develop AI-driven methods for continuous hazard monitoring, bridging the gap between research and practical, application-oriented solutions. Our goal is to refine and extend multi-temporal analysis techniques to enhance hazard detection and mitigate the impact on communities and infrastructure. The study site is located in Trier, Germany.

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

    ```
    python -m pip install .
    ```

    2. Editable mode

    ```
    python -m pip install -v --editable .
    ```

## 📜 License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)\
This is licensed under the MIT license.

## 📧 Contact

William Albert, albert@uni-heidelberg.de \
Ronald Tabernig, ronald.tabernig@uni-heidelberg.de \
[3DGeo Research Group](https://www.geog.uni-heidelberg.de/en/3dgeo), Institute of Geography, Heidelberg University