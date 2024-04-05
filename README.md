[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/LkCf-P6F)
ConfigWithYourProjectName
==============================
# DSEI210-S24-Final-Project
Final Project for DSEI210-S23 

## Home Credit - Credit Risk Model Stability

The project aims to address the challenge of predicting loan default risk for clients with limited credit history. By leveraging machine learning techniques, we seek to develop a robust and stable model that consumer finance providers can use to make more informed lending decisions. 

Link to the Kaggle Competition: https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability 

# Problem Understanding 


## Credit Risk

#### What is Credit Risk?
Credit risk refers to the possibility of a loss resulting from a borrower's failure to repay a loan or meet contractual obligations. Traditionally, it means the borrower does not make the promised loan payments, leading to a loss for the lender. This risk is a critical factor in the financial industry, especially for banks and lenders, as it directly impacts their profitability and sustainability.

#### Credit Risk in Loans
In the context of loans, credit risk assessment is crucial. Lenders, such as banks and consumer finance companies, evaluate the likelihood that a borrower may default on loan obligations. The assessment involves analyzing the borrower's financial history, current financial status, and other socio-economic factors. A high credit risk implies a higher probability of default, which could result in financial losses for the lender due to unpaid loan principal and interest.

#### Stakeholders
The primary stakeholders in credit risk assessment include:
- Lenders: Financial institutions like banks, credit unions, and consumer finance companies that provide loans to individuals and businesses.
- Borrowers: Individuals or entities seeking loans for various purposes, such as purchasing a home, car, or financing a business.
- Regulators: Government and regulatory bodies that oversee the lending industry to ensure fair practices and financial stability.
- Investors: Entities or individuals investing in financial products that involve credit risk, such as bonds and securities.

## Credit Risk Models

#### State-of-the-art Credit Risk Predictors
The landscape of credit risk prediction has evolved significantly with the advent of big data and machine learning technologies. State-of-the-art models often combine traditional financial indicators with advanced data analytics techniques. Some of the prominent methods include:
- Machine Learning Models: Algorithms such as random forests, gradient boosting machines (GBMs), and neural networks have been successful in predicting credit risk by learning complex patterns in data.
- Deep Learning Models: More complex models, like deep neural networks, have been explored for their ability to process vast amounts of unstructured data, providing insights from data points traditional models might miss.
- Alternative Data Sources: Modern credit risk models incorporate non-traditional data, such as mobile phone usage, social media activity, and even psychometric tests, to gauge a borrower's creditworthiness, especially useful for those with little to no traditional credit history

#### Previously Successful Features
In the context of credit risk prediction, certain features have been found to be particularly indicative of a borrower's likelihood to default. These include:

- Credit History: A borrower's past behavior with credit, including payment history, credit utilization, and age of credit accounts.
- Income Level and Stability: Regular and verifiable income sources are strong indicators of a borrower's ability to repay loans.
- Debt-to-Income Ratio: This measures how much of a borrower's income is going towards servicing existing debt, indicating the capacity to take on and repay new debt.
- Employment Status: Employment stability and the nature of a borrower's job can influence their risk profile.
- Socio-Economic Factors: Factors such as education level, marital status, and homeownership can also play a role in credit risk assessment.
  
## Metrics

1. **Gini Score Calculation for each WEEK**:
Accuracy Gini Score Calculation: used to quantify the accuracy of a model's predictions. It uses the AUC (Area under the ROC curve), and do a transformation
so the results are between 0 and 1. A Gini score close to 1 indicates perfect predictive accuracy, whereas a score closer to 0 indicates that the model's
predictions are no better than random chance. In our context, a Gini score is calculated for predictions corresponding to each WEEK_NUM, 
which means we are assessing how well the model predicts outcomes on a week-by-week basis.

   The Gini coefficient (\(Gini\)) is calculated for each week using the formula:
   $$ 
  \text{Gini} = 2 \times \text{AUC} - 1
  $$
   where $AUC$ is the Area Under the Receiver Operating Characteristic (ROC) Curve, providing a measure of the model's predictive accuracy for each week.

2. **Linear Regression Model through Gini Scores by Week**:
Linear Regression on Weekly Gini Scores: After calculating the Gini scores for each week, these scores are plotted over time (i.e., over the weeks).
A linear regression model is then fit through these weekly Gini scores. The linear model would generally take the form y = mx + b, where y represents
the Gini score, x represents the week number, m is the slope of the line (which indicates the rate of change of Gini scores over time), and b is the
y-intercept.

   A linear regression model is fitted through the Gini scores over the weeks, represented by the equation:
   ```math
   y = ax + b
   ```
   where:
   - $y$ is the Gini score,
   - $x$ is the week number,
   - $a$ is the slope of the regression line, indicating the rate of change in Gini score over time,
   - $b$ is the y-intercept.

3. **Calculation of Falling Rate**:
Falling Rate Calculation: The falling_rate is calculated from the slope of this linear regression line (m). This metric (falling_rate = -m)
essentially measures how quickly the Gini score decreases over time. If the model's predictive ability is dropping off over time, 
the slope would be negative, and thus the falling_rate (being the negative of a negative value) would be positive. A positive falling_rate indicates
a decline in predictive performance, and this metric is used to penalize models that lose their predictive ability.

   The falling rate is calculated using the slope (\(a\)) from the linear regression model. It is defined as:

   $falling\ rate = \min(0, a)$

   This step penalizes models that show a decline in predictive ability over time by considering the rate at which the Gini score decreases.

4. **Variability of Predictions**:
Penalty for model variability: Standard Deviation of Residuals: The residuals of a regression model are the differences between the observed values 
and the values predicted by the model. By calculating the standard deviation of these residuals, we're essentially measuring the variability in the
model's predictions from week to week. High variability (a large standard deviation of residuals) indicates that the model's performance is inconsistent.
This variability is undesirable, and therefore, a penalty is applied to models that exhibit high variability in their predictions. 
It would reduce the model's overall evaluation score.
   The variability in the model's predictions is quantified by calculating the standard deviation of the residuals from the linear regression model. The residuals are the differences between the observed Gini scores and the scores predicted by the model. The formula for calculating the standard deviation of residuals is:

   $std\ residuals = \sqrt{\frac{1}{N - 2} \sum{(y_i - (ax_i + b))^2}}$

   where $N$ is the number of weeks, $y_i$ are the observed Gini scores, and $ax_i + b$ are the predicted Gini scores by the linear regression model.

5. **Final Metric Calculation**:
   The final stability metric combines the mean Gini score, the falling rate penalty, and the penalty for variability in predictions:

   $stability\ metric = mean(Gini) + 88.0 \times \min(0, a) - 0.5 \times std\ residuals$

   This formula assesses the overall stability and consistency of the predictive model over time.

In summary, the evaluation method is designed to assess not just how accurate the model's predictions are at a single point in time,
but how consistently it performs over time, with penalties for declining accuracy and for inconsistency in prediction quality from week to week.
This approach ensures that selected models are both accurate and reliable over the duration of their application.



## Stability in Predictors

#### Impact of Stability in Credit Risk Prediction
The competition's emphasis on stability in credit risk prediction underscores the importance of maintaining reliable and consistent models over time. Here's how stability impacts credit risk prediction:

- Reliability: Stable models provide consistent predictions over time, offering lenders confidence in their decision-making processes. This reliability ensures that loan approval decisions are based on accurate and trustworthy assessments of credit risk.

- Risk Management: A stable credit risk model helps lenders effectively manage their risk exposure by identifying potential defaults early and accurately. By consistently evaluating borrowers' creditworthiness, lenders can mitigate losses and maintain a healthy loan portfolio.

- Financial Inclusion: Stability in credit risk prediction promotes financial inclusion by reducing the likelihood of unfair denials or approvals based on unstable model performance. Consistent and reliable models enable lenders to make more informed decisions, potentially increasing access to credit for underserved populations.

- Regulatory Compliance: Stable models ensure compliance with regulatory requirements, which often mandate the use of robust and stable credit risk assessment methods. By adhering to regulatory standards, lenders can avoid penalties and maintain their reputation in the industry.

- Business Continuity: Stability in credit risk prediction is crucial for the long-term sustainability of lending institutions. Unstable models can lead to financial losses, reputational damage, and even business failure. By prioritizing stability, lenders can safeguard their operations and maintain competitiveness in the market.

- Customer Trust: Consistent and fair credit risk assessment processes build trust with borrowers, enhancing customer satisfaction and loyalty. Borrowers are more likely to engage with lenders who demonstrate stability and reliability in their lending practices.


#### How to include stability into our model?
- Loss function includes regularization terms
- Ensembling?? 
- Reducing the number of features
- Reducing the model complexity (for instance, the number of leaves in a tree-based method)

## Data
- We see target unbalance.. consider this. How our metrics are affected by FP, TN, ... what should we optimize? 

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third-party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. The naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results-oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
