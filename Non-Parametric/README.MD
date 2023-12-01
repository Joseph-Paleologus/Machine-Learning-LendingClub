This folder contains code to implement non-parametric regression models, such as KNN, decision tree, bagging, random forest, etc.

The packages needed for this part of the analysis include `pandas` for data manipulation, `sklearn.model_selection` for model selection tools, `KNeighborsRegressor` and `GridSearchCV` from `sklearn` for regression modeling and hyperparameter optimization, `numpy` for numerical computing, StandardScaler for data normalization, `DecisionTreeRegressor` and `plot_tree` from `sklearn.tree` for decision tree modeling and visualization, `subplots` from `matplotlib` for plotting multiple figures, `GradientBoostingRegressor` and `RandomForestRegressor` from `sklearn.ensemble` for ensemble methods, `matplotlib.pyplot` for plotting graphs, and `warnings` to suppress warning messages. Please install those packages to your python environment prior to the implementation.

Please be aware that warnings are suppressed for cleaner output visualization using the warnings package.

First, import the dataset "Lending Club.xlsx" from the "Data Description" folder and implement the data cleaning process.

Then, execute each cell in the notebook sequence from top to bottom. Cells are organized to flow from data importation, preprocessing, model training, and visualization. Each cell has detailed comments to supplement understanding.

Notes:

The dataset used for this analysis is assumed to be present in the same directory as the notebook. If the dataset is located somewhere else, please update the directory accordingly.

For reproducibility, ensure that you set the `random_state` = 1.
