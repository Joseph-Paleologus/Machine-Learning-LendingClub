This folder contains codes to implement parametric regression models, such as PCR, PLS, variable selection, etc. (in Variable Selection, PCR & PLS.ipynb), and codes to implement regularization measures (in Regularization (with Lasso & Ridge & ElasticNet).ipynb).
                                                                                                                                                                                                           
The packages needed to implement Variable Selection and other linear parametric methods include `pandas` for data manipulation, `numpy` for numerical computing, `matplotlib.pyplot` for data visualization, `StandardScaler` from `sklearn.preprocessing` for feature scaling, `sklearn.model_selection` for tools like cross-validation, `sklearn.linear_model` for linear models, `Pipeline` for streamlining modeling workflows, `statsmodels.api` for statistical modeling, `PCA` for principal component analysis, `LinearRegression` for performing linear regression, `mean_squared_error` for evaluating the models, and `PLSRegression` and `PLSSVD` from `sklearn.cross_decomposition` for Partial Least Squares regression and Singular Value Decomposition, respectively. Additionally, the `time` module is used to track performance metrics, and `itertools` is included for efficient looping and iterations for variable selection processes.

Please install those packages to your python environment prior to the implementation.

First, import the dataset "Lending Club.xlsx" from the "Data Description" folder and implement the data cleaning process.

Then, execute each cell in the notebook sequence from top to bottom. Cells are organized to flow from data importation, preprocessing, model training, and visualization. Each cell has detailed comments to supplement understanding.

Notes:

The dataset used for this analysis is assumed to be present in the same directory as the notebook. If the dataset is located somewhere else, please update the directory accordingly.

For reproducibility, please make sure that you set the random_state = 1.
