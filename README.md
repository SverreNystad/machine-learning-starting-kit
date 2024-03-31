# Machine Learning Starting Kit
Template for machine learning projects, featuring a diverse collection of ML models, AutoML solutions, and simple EDA tools for streamlined project development. Users only need to specify target features and add their data path in the Config to kickstart a wide array of machine learning tasks.

## Table of Contents

## How to Use
To use the project one must give the project access to the data. This can either be done by uploading the data in the data folder and specify the path to the data in the config script.
Or in the case the dataset is too large to be on the device, one can create a new `DataLoader` class in the `data_loader.py` script and configure the default data loader in the config script to use the new DataLoader class. After one have given access to the data one must specify the target feature in the config script. After this the project is ready to be used.

Start by running the different notebooks in the EDA folder to get a better understanding of the data. After this one can start running the different models in the models folder.

## Project Organization
Each folder in the project has a specific purpose and is organized as follows:
<details>
<summary><b>Click to expand</b></summary>

```bash
├── .github
│   └── workflows                  # Github actions for CI/CD
|
├── data
│   ├── external                   # Data from third party sources.
│   ├── processed                  # The final, feature-engineered data sets for modeling.
│   └── raw                        # The original, immutable data set.
|
├── docs                           # Design documents (or other project documentation)
│   └── sphinx_docs                # A default Sphinx project; see sphinx-doc.org for details
|
├── eda                            # Notebooks for exploratory data analysis and data visualization
|
├── models                         # Training and prediction scripts for different models, including AutoML solutions.
|
├── results
│   ├── figures                    # Generated graphics and figures to be used in reporting
│   ├── predictions                # Model predictions as CSV files 
│   └── reports                    # Generated analysis as HTML, PDF, LaTeX, etc.
|
├── src                            
│   ├── config.py                  # Configuration file for the project
│   ├── ml_service.py              # A class that contains all the functions needed to train and save predictions of the models
│   ├── data                       # Scripts to fetch training and testing data
│   │   └── data_loader.py         
│   ├── features                   # Scripts to preprocess raw data into better features for modeling
│   │   ├── feature_engineering.py 
│   │   └── post_processing.py     # Script to use domain knowledge to post process the predictions
│   └── visualization              # Scripts to create exploratory and results oriented visualizations
│       └── visualize.py           
|
├── test                           # Scripts to test the project
└── requirements.txt               # The requirements file for reproducing the analysis environment, e.g.,
                                   # generated with `pip freeze > requirements.txt`

```

</details>

## Resources
### Feature Selection resources
* https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing
* https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection
* https://scikit-learn.org/stable/modules/feature_extraction.html#feature-extraction