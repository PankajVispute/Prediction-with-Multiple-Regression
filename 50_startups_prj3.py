#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Importing python liabraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


# In[4]:


## ETL - Extact, tranform and load the data
data=pd.read_csv(r"C:\Users\pv11379\OneDrive - Deere & Co\Desktop\Personal folder\Data science\Database/50_Startups.csv")


# In[5]:


## copy data into another dataframe
df=data.copy()


# # Read the data

# In[8]:


## shape of data
print("Total rows in data is- ", df.shape[0])
print("Total columns in data is- ", df.shape[1])


# In[11]:


df.index


# In[12]:


df.head(3)## top 3 rows of data with header


# In[13]:


df.tail(3)## bottom 3 rows of data with header


# In[15]:


df.columns ## columns in data


# In[36]:


## dataype of data
df.dtypes


# In[37]:


df.info()


# In[16]:


df.nunique() ## number of unique values in data


# In[18]:


df['State'].unique() ## unique state names are


# In[19]:


## statistical analysis of data
df.describe()


# In[20]:


## any null values in data
df.isna().sum()


# In[21]:


df.isna().any()


# In[ ]:


# inference - No any null values in data.


# In[29]:


## correlation in data
df_corr=df.corr()
df_corr


# In[32]:


sns.heatmap(df_corr,annot=True,cmap='rainbow')
plt.title("Correlation chart")
plt.show()


# In[ ]:


## insights -
1. R&d with profit having strong correlation.So if we spend more on r&d then more will be profit in business.
2. marketing with profit also having good correlation.So we can increase profit by spending more on marketing.
3. No correlation between administration and profit. 


# # Any outliers in data

# In[50]:


## check any outliers in data
col_cont= df.columns[df.dtypes!='O']
print(col_cont)


# In[56]:


for i in col_cont:
    plt.figure()
    plt.title("Box plot")
    sns.boxplot(x=df[i])


# In[ ]:


## insights - No any outliers in continous columns of data.


# # EDA - Exploratory Data analysis

# In[57]:


df.head(2)


# In[58]:


df['State'].value_counts()


# In[ ]:


## Insights - Equally number of times data collected from different state.


# In[86]:


px.scatter(df,x='R&D Spend',y='Profit',size='Profit',title='Profit vs R&D Spend',color='State',trendline='ols')


# In[ ]:


## Inference - 
1. R&D spend and Profit is hightly correlate with each other.
2. More spend in R&D will give more profit.
3. california state profit shows slightly high increase trend wrt to R&D spend.


# In[88]:


px.scatter(df,x='Marketing Spend',y='Profit',size='Profit',title='Profit vs Marketing Spend',color='State',trendline='ols')


# In[ ]:


## Inference - 
1. Marketing spend and Profit is having good correlation.
2. More spend in Marketing will give more profit.
3. california state profit shows slightly high increase trend with marketing spend.


# In[89]:


px.scatter(df,x='Administration',y='Profit',size='Profit',title='Profit vs Administration',color='State',trendline='ols')


# In[ ]:


## Insights - No any noticable impact on profit after spending on administration. So we dont need to spend more on 
administration.


# In[78]:


## Highest profitable state
px.histogram(df,x='State',y='Profit',color='State',title='State wise profit')


# In[ ]:


## Insights - 
1. highest profitable state os new york - 1.93M.
2. Florida state having 1.90M profitability.
3. California state having 1.76M profitability.


# In[120]:


px.pie(df,names='State',values='R&D Spend',title='State Wise R&D Spend')


# In[ ]:


## Insights - R&D spend in new york (12,95,316 Rs) - 35.1% of total spend and florida(12,91,584 Rs) - 35% of total spend 
is nearly equal and highest than california(10,99,180 Rs) - 29.8% of total spend.


# In[116]:


sns.barplot(df,x='State',y='Marketing Spend',estimator='sum',order=['Florida','New York','California'],palette='Paired')
plt.title("State Wise Marketing Spend")
plt.show()


# In[ ]:


## Insights - 
1. highest marketing spend is in florida state and then new york state.
2. New york state having sligthly more profit than florida even after slightly less marketing spend but R&D spend is more.


# In[ ]:


## Final insights of data - 
1. profit will increase when we increase R&D Spend and Marketing spend in state.
2. New york will give more profit if we spend more in R&D, Marketing.
3. Do not spend more on administration, it will not increase profitability.


# # Linear Regression Machine learning Model Building

# In[121]:


## import machine learning library
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# # feature Engineering 

# In[122]:


df.head(2)


# In[132]:


## define x= independent varibale, y= dependent variable
x=df[['R&D Spend','Administration','Marketing Spend']]
y=df['Profit']


# In[133]:


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=1234) ## split x&y into 80:20 ratio


# In[134]:


x_train.shape,y_train.shape ## training dataset shape


# In[135]:


x_test.shape,y_test.shape ## testing dataset shape


# In[136]:


model=LinearRegression()


# In[137]:


model_fit=model.fit(x_train,y_train) ## fit into model


# In[144]:


## intercept of model
print(round(model_fit.intercept_,0))


# In[154]:


## coef of model
print(np.around(model_fit.coef_,2))


# In[162]:


y_pred=model_fit.predict(x_test)


# In[164]:


df_y_pred=pd.DataFrame(y_pred,columns=['profit_pred']) ## convert into dataframe.
df_y_pred['profit_actual']=y_test.values ## add actual profit column into pred dataframe.


# In[166]:


df_y_pred


# In[167]:


## check model strenght
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from math import sqrt


# In[169]:


r2_score(df_y_pred['profit_pred'],df_y_pred['profit_actual']) ## r2_score of model


# In[170]:


mean_absolute_error(df_y_pred['profit_pred'],df_y_pred['profit_actual']) ## MAS of model


# In[172]:


sqrt(mean_squared_error(df_y_pred['profit_pred'],df_y_pred['profit_actual'])) ## RMSE of model


# # check another best model if any-1

# In[185]:


df1=pd.get_dummies(df['State']) ## creat dummy dataframe on state
final_data=pd.concat([df,df1],axis=1) ## convcatenate both dataframe
final_data.columns ## columns in final_data


# In[186]:


x1=final_data[['R&D Spend', 'Administration', 'Marketing Spend','California', 'Florida', 'New York']]
y1=final_data['Profit']


# In[187]:


x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,train_size=0.8,random_state=1234) ## split x&y into 80:20 ratio


# In[188]:


x1_train.shape,y1_train.shape ## training dataset shape


# In[189]:


x1_test.shape,y1_test.shape ## testing dataset shape


# In[190]:


model1=LinearRegression()


# In[191]:


model1_fit=model1.fit(x1_train,y1_train) ## fit into model


# In[192]:


## intercept of model
print(round(model1_fit.intercept_,0))


# In[193]:


## coef of model
print(np.around(model1_fit.coef_,2))


# In[194]:


y1_pred=model1_fit.predict(x1_test)


# In[195]:


df_y1_pred=pd.DataFrame(y1_pred,columns=['profit_pred']) ## convert into dataframe.
df_y1_pred['profit_actual']=y1_test.values ## add actual profit column into pred dataframe.


# In[196]:


df_y1_pred


# In[197]:


r2_score(df_y1_pred['profit_pred'],df_y1_pred['profit_actual']) ## r2_score of model


# In[198]:


mean_absolute_error(df_y1_pred['profit_pred'],df_y1_pred['profit_actual']) ## MAS of model


# In[199]:


sqrt(mean_squared_error(df_y1_pred['profit_pred'],df_y1_pred['profit_actual'])) ## RMSE of model


# # check another best model if any-2

# In[200]:


x2=final_data[['R&D Spend', 'Marketing Spend','California', 'Florida', 'New York']]
y2=final_data['Profit']


# In[201]:


x2_train,x2_test,y2_train,y2_test=train_test_split(x2,y2,train_size=0.8,random_state=1234) ## split x&y into 80:20 ratio


# In[202]:


model2=LinearRegression()


# In[203]:


model2_fit=model2.fit(x2_train,y2_train) ## fit into model


# In[204]:


y2_pred=model2_fit.predict(x2_test)


# In[205]:


df_y2_pred=pd.DataFrame(y2_pred,columns=['profit_pred']) ## convert into dataframe.
df_y2_pred['profit_actual']=y2_test.values ## add actual profit column into pred dataframe.


# In[206]:


df_y2_pred


# In[207]:


r2_score(df_y2_pred['profit_pred'],df_y2_pred['profit_actual']) ## r2_score of model


# In[208]:


mean_absolute_error(df_y2_pred['profit_pred'],df_y2_pred['profit_actual']) ## MAS of model


# In[209]:


sqrt(mean_squared_error(df_y2_pred['profit_pred'],df_y2_pred['profit_actual'])) ## RMSE of model


# # check another best model if any-3

# In[210]:


x3=df[['R&D Spend','Marketing Spend']]
y3=df['Profit']


# In[211]:


x3_train,x3_test,y3_train,y3_test=train_test_split(x3,y3,train_size=0.8,random_state=1234) ## split x&y into 80:20 ratio


# In[212]:


model3=LinearRegression()


# In[213]:


model3_fit=model3.fit(x3_train,y3_train) ## fit into model


# In[214]:


y3_pred=model3_fit.predict(x3_test)


# In[215]:


df_y3_pred=pd.DataFrame(y3_pred,columns=['profit_pred']) ## convert into dataframe.
df_y3_pred['profit_actual']=y3_test.values ## add actual profit column into pred dataframe.


# In[216]:


df_y3_pred


# In[217]:


r2_score(df_y3_pred['profit_pred'],df_y3_pred['profit_actual']) ## r2_score of model


# In[218]:


mean_absolute_error(df_y3_pred['profit_pred'],df_y3_pred['profit_actual'])


# In[219]:


sqrt(mean_squared_error(df_y3_pred['profit_pred'],df_y3_pred['profit_actual']))


# In[ ]:


## r2 score of first model is more and closed to 1, so consider as strong model.
## so we are finalised model_fit linear regression model and save it using library.


# In[220]:


## save the model
import joblib


# In[221]:


filename='50_startup_multiple_regression_model.sav' ## file name of model
joblib.dump(model_fit,filename) ## model dump into above filename


# In[222]:


## load the model
load_model=joblib.load(filename)
print(load_model)


# In[225]:


y_var_predict=load_model.predict(x_test)


# In[226]:


y_var_predict


# In[ ]:




