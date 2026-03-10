Credit Risk Default Prediction

Overview:
Credit risk is one of the most consequential problems in banking. A model that misjudges default likelihood does not just affect a spreadsheet. It affects real lending decisions and real people. This project builds a predictive model to estimate the probability of customer loan default while walking through data science workflow from raw data to model evaluation and business interpretation.

The goal is not only to build a model that performs well, but also to understand why it performs well. The analysis focuses on identifying the key drivers of default risk and translating model outputs into insights that could support real credit decision environments.

Dataset:
Source: Credit Risk Dataset (Kaggle)

Size: 32,581 loan applications

Target variable: loan_status (0 = repaid, 1 = defaulted)

Default rate: 21.8% of borrowers defaulted

Variables include: borrower age, income, employment length, home ownership, loan amount, interest rate, loan intent, loan grade, and credit history

Methodology:
 1. Data Cleaning
    Removed unrealistic values such as ages above 90 and employment lengths above 60 years
    Filled 9.6 percent missing interest rate values with the median rate
    Retained 28,465 clean records after preprocessing

 2. Exploratory Data Analysis
 Key findings from the exploratory analysis:
    Younger borrowers between ages 20 and 35 account for the majority of both applications and defaults
    Debt consolidation loans have the highest default rate at 28 percent, followed by medical loans at 26 percent
    Defaulters tend to pay significantly higher interest rates on average
    Loan amount as a percentage of income clearly separates defaulters from non defaulters

3. Feature Engineering:
Three new features were created to capture relationships not expressed directly in the raw data:
    income_to_loan measures how many times income covers the loan amount
    interest_burden captures interest rate relative to income to reflect repayment pressure
    stable_employment is a binary indicator for borrowers with three or more years of employment

4. Handling Class Imbalance:
SMOTE (Synthetic Minority Oversampling Technique) was applied to the training set to balance the 78 to 22 default split. This helps the models learn meaningful patterns associated with default rather than simply predicting the majority class.

5. Models Trained
Three classification models were trained and compared:
   Logistic Regression
   Decision Tree
   Random Forest with 100 estimators

Results: 
Model	ROC AUC	Accuracy	Defaults Caught	False Alarms:
   Logistic Regression	0.855	84.9%	595	218
   Decision Tree	0.845	89.0%	951	341
   Random Forest	0.935	93.4%	896	33

The Random Forest model achieved the strongest overall performance with a ROC AUC of 0.935 while maintaining a very    low false alarm rate. This represents a strong balance between identifying borrowers likely to default and avoiding unnecessary rejection of creditworthy applicants.

Key Findings:
Loan percent income is the strongest predictor of default. Borrowers whose loan exceeds roughly 40 percent of their annual income show significantly higher risk. This aligns with affordability checks used in real lending.

Interest rate acts as both a signal and a driver of risk. Lenders charge higher rates to riskier borrowers, but higher rates also increase repayment pressure. The model captures this relationship.

Loan intent carries meaningful risk information. Debt consolidation loans show much higher default rates than other loan purposes, suggesting loan purpose is a valuable predictor.

Age and loan size interact in meaningful ways. Younger borrowers taking large loans form one of the highest risk segments, while older borrowers taking smaller loans default at much lower rates.

Random Forest produced the strongest predictive performance, although Logistic Regression remains important in practice because simpler models are easier to explain in regulated lending environments.

How to Run:
Clone the repository
git clone https://github.com/tmudau/credit-risk-prediction.git

Install dependencies:
pip install -r requirements.txt

Run the notebook
notebooks/credit-risk-analysis.ipynb
Run the notebook

notebooks/credit_risk_analysis.ipynb
