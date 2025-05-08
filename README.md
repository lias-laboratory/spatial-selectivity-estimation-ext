# Bridging Machine Learning and Query Optimization: Feedback-Driven Selectivity Estimation for Spatial Filters

This repository contains code and resources related to our research on feedback-driven spatial selectivity estimation. The project focuses on leveraging optimizer feedback to improve the estimation of selectivity for multi-dimensional spatial predicates. Various Machine Learning models, including neural networks, tree-based models, and instance-based models, are explored to address this challenging task efficiently across different spatial filter types.

## Code Structure
The repository is organized as follows:

- analyse_results: Contains all code for generating figures, plots, and conducting statistical tests presented in our study
- intersect_filter: Implementation of our ML approach for intersect-type spatial selectivity estimation
- contain_filter: Implementation of our ML approach for containment-type spatial selectivity estimation
- distance_filter: Implementation of our ML approach for distance-based spatial selectivity estimation
- traditional_methods: Implementation of baseline approaches (RTree and Histogram-based estimation) used for comparison

## Additional Resources
To facilitate reproduction of our results without requiring lengthy retraining of models, we provide a [downloadable zip](https://www.lias-lab.fr/ftppublic/research/spatial-selectivity-estimation-ext) file containing:

- All 14 spatial datasets used in our experiments
- Pre-trained models for each filter type, including traditional approaches

# License
The work in this repository is licensed under the MIT License. Please refer to the [LICENSE](https://github.com/lias-laboratory/spatial-selectivity-estimation-ext/blob/main/LICENSE) file for more details.

## Contributors
1. Nadir GUERMOUDI (LIAS/University of Poitiers & LRIT/University of Tlemcen)
2. Houcine MATALLAH (LRIT/University of Tlemcen)
3. Amin MESMOUDI (LIAS/University of Poitiers)
4. Seif-Eddine BENKABOU (LIAS/University of Poitiers)
5. Allel HADJALI (LIAS/ISAE-ENSMA)
6. Ahmed-Youcef BENHALIMA (LIAS/ISAE-ENSMA)
