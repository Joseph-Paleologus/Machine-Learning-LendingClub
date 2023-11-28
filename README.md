# Predicting Loan Rate on P2P Platforms Using LendingClub Dataset


The emergence of Peer-to-Peer (P2P) lending has revolutionized the loan market by directly connecting borrowers with investors. A pivotal aspect of this market is the determination of appropriate loan rates. Loan rates are crucial as they reflect the risk associated with lending money. It also significantly influences the borrower’s ability to repay the loan and the investor’s return on investment. If the loan rate is too high, the debtors might default and fail to meet the repayment deadline. The creditor cannot get a satisfactory investment return if the loan rate is too low.

Our primary research objective is to use the LendingClub dataset to develop a predictive model for loan rates. This model aims to estimate the interest rate assigned to a loan based on various predictors. These predictors could include borrower-specific information such as credit score, income level, employment history, and loan-specific details like loan amount and loan purpose. 

![Image](https://miro.medium.com/v2/resize:fit:1176/format:webp/1*5o6x9IUFI4U7j0EeZuUraA.png)

## Data Description


| Variable Name     | Description                              | Data Type |
|-------------------|------------------------------------------|-----------|
| loan_amnt         | The total amount of the loan             | Numeric   |
| loan_term         | The term of the loan (in months)          | Numeric   |
| emp_length        | Employment length (in years)             | Categorical|
| home_ownership    | The home ownership status                | Categorical|
| income_source_verified| Indicator if income source was verified     | Binary   |
| income_verified   | Indicator if income was verified         | Binary    |
| income_thou       | Income in thousands                      | Numeric   |
| debt_income       | Debt-to-income ratio                     | Numeric   |
| delinq_2yrs       | The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years | Numeric   |
| credit_history_length | Length of the borrower's credit history | Numeric   |
| FICO              | FICO credit score                        | Numeric   |
| open_acc          | Number of open credit lines              | Numeric   |
| derogatory_recs   | Number of derogatory records             | Numeric   |
| revol_balance     | Total credit revolving balance           | Numeric   |
| revol_util        |The amount of credit the borrower is using relative to all available revolving credit.          | Numeric   |
| loan_rate         | Interest rate of the loan (Our Research target) | Numeric   |


## Motivation and Problem

The traditional method of determining loan rates often relies on somewhat arbitrary factors, and it involves a lengthy process that necessitates the involvement of trained staff. While potentially suitable for conventional financial institutions such as commercial banks where thoroughness is prioritized over speed, this approach may not align well with the needs of Peer-to-Peer (P2P) lending platforms that value prompt service and efficiency. In the context of P2P platforms, which operate in a more dynamic and fast-paced environment, the traditional loan rate determination process can be seen as cumbersome and time-consuming.

To address this challenge, integrating a machine-learning algorithm could significantly enhance the efficiency of the lending process for these platforms. Machine learning offers the ability to quickly, accurately, and objectively analyze vast amounts of data. By automating the analysis of borrower data, these algorithms can provide rapid and more objective assessments of credit risk, leading to quicker loan rate determinations. This streamlines the process and potentially reduces the likelihood of human error or bias in setting loan rates.

This approach has significant benefits compared to traditional methods. It can rapidly process large volumes of data, providing loan rates much quicker than manual methods. It could be vital in P2P platforms, where both lenders and borrowers value efficiency and timeliness.


## Data Transformation

From the data description section, we can see that there are two categorical variables, "Home" and "Employment length." To incorporate them into our regression model, we transform them into dummies. The 'Home' variable has three possible values: 'MORTGAGE,' 'OWN,' and 'RENT.'  To avoid multicollinearity, we only keep 'MORTGAGE' and 'RENT" in our dataset. Similarly, for 'Employment length,' we dropped 'emp_length_<1 year' for the same reason.

## Exploring Data 
<details>
  <summary>Click to expand</summary>

### Correlation Heatmap of the Variables.

![Image](https://raw.githubusercontent.com/Joseph-Paleologus/Machine-Learning-LendingClub/main/Data%20description/image.png)
From the map we can observe that:
1. Loan_term appears to have a moderate positive correlation with loan_rate, indicated by a coefficient value of 0.45. This suggests that as the term of the loan increases, the loan rate tends to be higher.

2. FICO score shows a negative correlation with loan_rate, with a coefficient of -0.41. This indicates that higher FICO scores, which suggest better creditworthiness, are associated with lower loan rates.

3. Revol_util (revolving line utilization rate) has a positive correlation with loan_rate at a value of 0.25, suggesting that higher utilization rates of available credit may lead to higher loan rates.

4. Debt_income (debt-to-income ratio) and Income_thou (income in thousands) have weaker positive correlations with loan_rate, with coefficients of 0.15 and 0.13, respectively, implying a less pronounced relationship with the loan rate.

Other variables, such as Loan_amt (loan amount), Income_source_verified, and Open_acc (number of open credit lines), show very weak correlations with loan_rate, as indicated by coefficients close to 0.

### Frequency Plot of FICO Score

![Image](https://raw.githubusercontent.com/Joseph-Paleologus/Machine-Learning-LendingClub/main/Data%20description/FICO.png)
The minimum FICO score for applicants who are granted the loans is 662, showing that there is a minimum FICO requirement for applicants to receive the loans.

The highest frequency occurs in the lower FICO score intervals, starting with the 660 range. This suggests that a larger number of individuals in the dataset have lower FICO scores.

As the FICO score increases, the frequency of individuals within those intervals decreases. This implies that fewer individuals have higher FICO scores.

The distribution appears to be right-skewed, meaning there is a longer tail towards the higher end of the FICO score spectrum. This skewness indicates that while most of the scores are concentrated on the lower end, there are still some individuals with very high credit scores, but they are less common.

### Frequency Plot of Loan Amount

![Image](https://raw.githubusercontent.com/Joseph-Paleologus/Machine-Learning-LendingClub/main/Data%20description/Loan_amt.png)
There is a relatively high frequency of loans in the lower loan amount ranges, starting from 0 up to around 15000. This suggests that most individuals in the dataset are taking out smaller loans.

The highest frequency of loans seems to occur in the 10000 to 15000 range, indicating that this loan amount range is the most common among the dataset's individuals.

As the loan amount increases beyond the peak range (15000), the frequency of loans gradually decreases, which means that larger loans are less common in this dataset.

The distribution may have multiple modes (peaks), as there seem to be several intervals with high frequencies, which suggests that certain loan amounts are particularly common, potentially due to common financing needs or lending products offered.

Overall, the distribution of loan amounts in this dataset is multimodal and shows that smaller loans are more frequently taken out than larger ones, with particular loan amounts appearing to be more popular than others. 

### Frequency Plot of Log Income

Because the data has a long-tail effect, we log-transform it.
![Image](https://raw.githubusercontent.com/Joseph-Paleologus/Machine-Learning-LendingClub/main/Data%20description/log_inc.png)
A peak in the distribution, which seems to occur around the 4.5 to 5.0 log income value, indicates that the majority of incomes in the dataset cluster around 31622 ($10 ^{4.5}$) to 100000 ($10 ^{5}$).

The bars at the lower and higher ends of the spectrum have lower counts, suggesting fewer individuals with very low or very high incomes, relative to the central peak.

The shape of the distribution could be somewhat bell-shaped with a peak in the middle, indicating that the underlying income distribution, before the log transformation, is right-skewed, which is typical for income data where a large number of people earn moderate incomes and fewer people earn extremely high incomes.

### Frequency Table of Categorical Variables
![Image](https://raw.githubusercontent.com/Joseph-Paleologus/Machine-Learning-LendingClub/main/Data%20description/categorical.png)
1. Empolyment Length: Generally, the number of loan applicants shows a decreasing pattern as employment length increases. However, the data does not distinguish employment length after 10 years, so we can observe there is a large amount of applicants with 10+ years working experience.

2. Home Ownership Status: The home ownership status graph might show a higher frequency for "MORTGAGE" and "RENT," as these are common living situations. The frequency for "OWN" might be lower, reflecting the smaller proportion of individuals who own their homes outright without a mortgage.

3. Income Source Verification and Income Verification: Those two graphs show that most applicants do not have their income source and income verified.

</details>

## Model Setups

All the codes in this project are executed under a Python 3.8+ environment. Our computing environment is MacBook Pro 2021 with M1 Max Chip and 32 GB memory. We expect less sophisticated computer systems will take more time to run the models.

### Parametric Models
Parametric models are the simplest and most intuitive way of modeling the loan rate. However, since some of the predictor variables show small correlations with the response, incorporating those predictors in our model will be problematic for an accurate, consistent prediction. Therefore, we introduce some methods for variable selection and dimensional reduction.

#### Forward Selection
Forward Selection is a stepwise approach in model building where predictors are added one at a time to a regression model, starting with the variable most strongly associated with the response variable. At each step, the variable that provides the greatest additional improvement to the model fit is included until no more significant variables are found, or a specified criterion is met. For our dataset with many predictors, this method could be computationally more convenient than traditional subset selection methods. 

#### Backward Selection
Backward Selection is also a stepwise approach in modeling where all potential predictors are initially included, and then the least significant variables are removed one by one. This process continues until only variables that contribute significantly to the model's predictive power remain, ensuring a more parsimonious and potentially more interpretable model.

#### PCR
PCR is useful for variable selection as it reduces the dimensionality of the data by selecting a few principal components, which are linear combinations of the original variables, thus simplifying the model without significantly losing information. It combines Principal Component Analysis (PCA) and regression and is useful for handling multicollinearity in regression models. 

#### PLS
Like Principal Component Regression (PCR), PLS reduces the dimensionality of data but focuses more on predicting a dependent variable by finding the multidimensional direction in the space of the independent variables that explains the maximum multidimensional variance of the dependent variable. It not only reduces the number of variables but also maintains the ones most relevant for predicting the dependent variable.

### Regularization
To reduce the probability of overfitting, we introduce regularization methods:

#### Ridge Regression
Ridge regression is a technique used to analyze multiple regression data that suffer from overfitting. It reduces the standard error by adding a bias parameter in the estimates of the regression and shrinking large parameters to avoid overfitting issues.

#### Lasso (Least Absolute Shrinkage and Selection Operator) Regression
Lasso regression is a type of linear regression that uses shrinkage: it applies a penalty to the absolute size of the regression coefficients. This method not only helps in variable selection by shrinking less important coefficients to zero but also improves the prediction accuracy and interpretability of the statistical model.




