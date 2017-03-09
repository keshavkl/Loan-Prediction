# Loan-Prediction
Predicting whether a loan will be approved or not

Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form.
These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. 
To automate this process, they have given a problem to identify the customers segments, those are eligible for loan amount so that they can specifically target these customers. 

https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/

Evaluation Metric is accuracy i.e. percentage of loan approval we can correctly predict.

I have built 2 models,
1. initial model with minimal data manipulation and no new variables created.
  Following this, I did some data munging and exploratory analysis to understand variable importance and create new variables.
2. The final model which ranked me at 16th position on the leaderboard with accuracy of 0.805.
  I have used gradient boosting with ensemble learning to optimize the model.
