This folder contains descriptive contents for the project.

First, import the "Lending Club.xlsx" dataset and run the "Data cleaning.ipynb" to replicate the result under this folder. Related codes can also be found in other ipynbs' under other folders.

From the data description section, we can see that there are two categorical variables, "Home" and "Employment length." To incorporate them into our regression model, we transform them into dummies. The 'Home' variable has three possible values: 'MORTGAGE,' 'OWN,' and 'RENT.' To avoid multicollinearity, we only keep 'MORTGAGE' and 'RENT" in our dataset. Similarly, for 'Employment length,' we dropped 'emp_length_<1 year' for the same reason. We also drop all the data with NAs' in any of the columns to ensure the robustness of our project.

Secondly, run the "Visual.ipynb" under this folder. This file provides visualizations of key features in the data set we use in this project. The results are stored as pictures in this folder. They are also presented in the "README.MD" under the master folder.

After finishing all the files under this folder, you can proceed to other folders.
