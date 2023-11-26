# Machine-Learning-LendingClub
Using ML methods to predict P2P loan rate


## Predicting Loan Rate on P2P Platforms Using LendingClub Dataset


The emergence of Peer-to-Peer (P2P) lending has revolutionized the loan market by directly connecting borrowers with investors. A pivotal aspect of this market is the determination of appropriate loan rates. Loan rates are crucial as they reflect the risk associated with lending money. It also significantly influences the borrower’s ability to repay the loan and the investor’s return on investment. If the loan rate is too high, the debtors might default and fail to meet the repayment deadline. The creditor cannot get a satisfactory investment return if the loan rate is too low.

Our primary research objective is to use the LendingClub dataset to develop a predictive model for loan rates. This model aims to estimate the interest rate assigned to a loan based on various predictors. These predictors could include borrower-specific information such as credit score, income level, employment history, and loan-specific details like loan amount and loan purpose. 

![Image](https://miro.medium.com/v2/resize:fit:1176/format:webp/1*5o6x9IUFI4U7j0EeZuUraA.png)

### Data Description


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


### Motivation and Problem

The traditional method of determining loan rates often relies on somewhat arbitrary factors, and it involves a lengthy process that necessitates the involvement of trained staff. While potentially suitable for conventional financial institutions such as commercial banks where thoroughness is prioritized over speed, this approach may not align well with the needs of Peer-to-Peer (P2P) lending platforms that value prompt service and efficiency. In the context of P2P platforms, which operate in a more dynamic and fast-paced environment, the traditional loan rate determination process can be seen as cumbersome and time-consuming.

To address this challenge, integrating a machine-learning algorithm could significantly enhance the efficiency of the lending process for these platforms. Machine learning offers the ability to quickly, accurately, and objectively analyze vast amounts of data. By automating the analysis of borrower data, these algorithms can provide rapid and more objective assessments of credit risk, leading to quicker loan rate determinations. This streamlines the process and potentially reduces the likelihood of human error or bias in setting loan rates.

This approach has significant benefits compared to traditional methods. It can rapidly process large volumes of data, providing loan rates much quicker than manual methods. It could be vital in P2P platforms, where both lenders and borrowers value efficiency and timeliness.


### Data Transformation

From the data description section, we can see that there are two categorical variables, "Home" and "Employment length." To incorporate them into our regression model, we transform them into dummies. The 'Home' variable has three possible values: 'MORTGAGE,' 'OWN,' and 'RENT.'  To avoid multicollinearity, we only keep 'MORTGAGE' and 'RENT" in our dataset. Similarly, for 'Employment length,' we dropped 'emp_length_<1 year' for the same reason.

### Exploring Data 

#### Correlation heatmap of the variables.

![Image](https://raw.githubusercontent.com/Joseph-Paleologus/Machine-Learning-LendingClub/main/Data%20description/image.png)https://raw.githubusercontent.com/Joseph-Paleologus/Machine-Learning-LendingClub/main/Data%20description/image.png)
