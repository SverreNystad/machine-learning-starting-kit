# Machine Learning Starting Kit
Template for machine learning projects, featuring a diverse collection of ML models, AutoML solutions, and simple EDA tools for streamlined project development. Users only need to specify target features and add their data path in the Config to kickstart a wide array of machine learning tasks.

## Table of Contents

## How to Use
To use the project one must give the project access to the data. This can either be done by uploading the data in the data folder and specify the path to the data in the config script.
Or in the case the dataset is too large to be on the device, one can create a new `DataLoader` class in the `data_loader.py` script and configure the default data loader in the config script to use the new DataLoader class. After one have given access to the data one must specify the target feature in the config script. After this the project is ready to be used.

Start by running the different notebooks in the EDA folder to get a better understanding of the data. After this one can start running the different models in the models folder.
