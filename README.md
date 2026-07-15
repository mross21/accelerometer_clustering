# Accelerometer Clustering

Code for clustering smartphone accelerometer readings on the unit sphere to identify predominant phone orientations over time. The method was developed for smartphone accelerometer data and is described in:

Ross, M. K., Tulabandhula, T., Bennett, C. C., Baek, E., Kim, D., Hussain, F., et al. *A Novel Approach to Clustering Accelerometer Data for Application in Passive Predictions of Changes in Depression Severity.* Sensors 23, 1585 (2023). DOI: 10.3390/s23031585.

## Overview

The main workflow:

1. Filters accelerometer samples by magnitude to keep points near the unit sphere.
2. Converts Cartesian accelerometer values `(x, y, z)` to spherical coordinates.
3. Fits a spherical kernel density estimate to each user/week subset.
4. Samples density across a grid of equidistant points on the sphere.
5. Identifies local density maxima as candidate cluster centers.
6. Assigns each accelerometer point to the nearest cluster center.
7. Saves clustered outputs and generates plots for inspection.

The repository also includes validation, downstream feature engineering, time-series plotting, baseline clustering methods for comparison, and a few helper scripts in `misc_functions/`.

## Repository Layout

- `accelerometer_clustering_pipeline.py`: main clustering pipeline that assigns orientation cluster labels to accelerometer readings.
- `analysis/cluster_method_validation.py`: validation script using labeled test data.
- `analysis/add_clusterVars_to_dataset.py`: creates downstream features from clustered accelerometer data.
- `plots/plot_acc_clusters_over_time.py`: visualizes how cluster/orientation preference changes over time.
- `clustering_method_comparison/`: alternative clustering baselines.
  - `spherical_kmeans_clustering.py`
  - `dbscan_clustering.py`
  - `GMM_clustering.py`
- `plots/vMF_plot_2D_XZ_byUserWeek.py`: 2D visualization of spherical KDE results.
- `plots/vMF_plot_3D_byUserWeek.py`: 3D visualization of spherical KDE results.
- `plots/equidistantPts_with_vMFdensities.py`: samples equidistant sphere points and evaluates densities.
- `accelerometer_utils.py`: shared preprocessing and geometry helpers.
- `io_utils.py`: shared output-path helpers.
- `config.py`: centralized path placeholders for all scripts.

## Configuration

Set all file and directory paths in `config.py` before running scripts. The defaults are example placeholders, not real paths.
The active pipeline and analysis scripts read parquet inputs and write parquet outputs.

## Data Inputs

The scripts expect accelerometer data with at least:

- `userID`
- `weekNumber`
- `x`, `y`, `z`

Some downstream scripts also expect session- or keypress-level parquet files with fields such as:

- `recordId`
- `sessionNumber`
- `keypressTimestampLocal`
- `sessionTimestampLocal`
- `date`
- `hour`

The exact file format depends on the script being run. Helper scripts in `misc_functions/` are standalone utilities and are not part of the main pipeline.

## Outputs

Depending on the script, outputs may include:

- clustered accelerometer files with added cluster labels
- `k_list.parquet` containing the estimated number of clusters per user/week
- per-user/per-week figures showing cluster assignments
- derived feature tables for downstream analysis
- time-series plots of orientation preference changes over the study period

Output paths are created through `io_utils.py` when a base directory is provided.

## Requirements

Python version used by the scripts:

- Python 3.7.4

Python packages:

- `pandas==1.2.0`
- `numpy==1.17.2`
- `scipy==1.3.1`
- `networkx==2.6.3`
- `matplotlib==3.5.3`
- `pyarrow==8.0.0`
- `scikit-learn==0.21.3`
- `spherecluster==0.1.7`
- `spherical_kde==0.1.0`

Install the package versions in [`requirements.txt`](./requirements.txt).

## Usage

Typical workflow:

1. Update `config.py` with your input and output paths.
2. Run the main clustering pipeline to estimate cluster counts and label accelerometer samples.
3. Use the resulting clustered files for downstream feature generation.
4. Run validation or comparison scripts if you want to evaluate alternative methods.
5. Generate plots to inspect orientation changes over time.

## Notes

- The code filters accelerometer points using a magnitude threshold of approximately `0.95 <= r <= 1.05` to focus on samples near the unit sphere.
- Some scripts skip Android data depending on the file format and preprocessing assumptions.
- Several scripts assume a specific folder structure and naming convention for user-level files.

## Citation

If you use this code or method, please cite:

Ross, M. K., Tulabandhula, T., Bennett, C. C., Baek, E., Kim, D., Hussain, F., et al. *A Novel Approach to Clustering Accelerometer Data for Application in Passive Predictions of Changes in Depression Severity.* Sensors 23, 1585 (2023). DOI: 10.3390/s23031585.
