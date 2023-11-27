# Prediction-with-Multiple-linear-Regression
Implementation of multiple linear regression model.

# Description
* Multiple linear regression algorithm is used for regression problems.
* Dataset is having continous data and multiple independent variable, so multiple linear regression algorithm used for building model.
* 50 start ups dataset used for model building.

# Data set format
* CSV (Comma Separated Values) format.
* Attributes can be integer or real values.
* Responses can be integer, real or categorical.

# Overview
The primary goal is predict profit of start up based on R&D spend, administration, marketing spend and state.

# liabrary 
* pandas, numpy, matplotlib,seaborn,sklearn,joblib used in project

# Methodology
1. ## Machine learning life cycle:
   - followed indistry standard practice of machine learning life cycle steps.
2. ## Preprocessing and EDA:
   - implement necessary transformation, preprocessing of dataset.
   - conduct exploratory data analysis on dataset.
3. ## Visualization:
   - visualised data using visualisation library like matplotlib, seaborn.
4. ## Algorithm:
   - scikit library use for linear regression.
5. ## model validation:
   - model validate with r2_score, RMSE.
6. ## save model:
   - joblib library used to dump model.
   - model is saved in .ipynb formate as 50_startups_multiple_regression_model.
# EDA:
- No any null values in dataset.
- profit having strong correlation with R&D Spend - 97% and then marketing spend- 74%.
- No outliers in dataset.
- california state profit shows high increase trend wrt to R&D spend.
- california state profit shows high increase trend with marketing spend.
- highest profitable state is new york - 1.93M.
- R&D spend in new york - 35.1% of total spend.
- highest marketing spend is in florida state.
  
# Multiple-linear-regression-model:
- multiple number of model build using different independent variable.
- state categorial data converted into numeric using get dummies.
- model with r2_score - 90% is found highest accuracy, so it is load and saved using joblib.
  
