#!/usr/bin/env python
# coding: utf-8

# # Bike Sharing Assignment

# ## Problem Statement

# A bike-sharing system is a service in which bikes are made available for shared use to individuals on a short term basis for a price or free. Many bike share systems allow people to borrow a bike from a "dock" which is usually computer-controlled wherein the user enters the payment information, and the system unlocks it. This bike can then be returned to another dock belonging to the same system.
# 
# 
# A US bike-sharing provider BoomBikes has recently suffered considerable dips in their revenues due to the ongoing Corona pandemic. The company is finding it very difficult to sustain in the current market scenario. So, it has decided to come up with a mindful business plan to be able to accelerate its revenue as soon as the ongoing lockdown comes to an end, and the economy restores to a healthy state. 
# 
# 
# In such an attempt, BoomBikes aspires to understand the demand for shared bikes among the people after this ongoing quarantine situation ends across the nation due to Covid-19. They have planned this to prepare themselves to cater to the people's needs once the situation gets better all around and stand out from other service providers and make huge profits.
# 
# 
# They have contracted a consulting company to understand the factors on which the demand for these shared bikes depends. Specifically, they want to understand the factors affecting the demand for these shared bikes in the American market. The company wants to know:
# 
# Which variables are significant in predicting the demand for shared bikes.
# How well those variables describe the bike demands
# Based on various meteorological surveys and people's styles, the service provider firm has gathered a large dataset on daily bike demands across the American market based on some factors. 
# 
# 

# ## Business Goal

# You are required to model the demand for shared bikes with the available independent variables. It will be used by the management to understand how exactly the demands vary with different features. They can accordingly manipulate the business strategy to meet the demand levels and meet the customer's expectations. Further, the model will be a good way for management to understand the demand dynamics of a new market. 

# ## Reading and Understanding the Data 

# In[2]:


# Supress Warnings

import warnings
warnings.filterwarnings('ignore')


# In[3]:


#importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


# reading data
bike = pd.read_csv("day.csv")


# In[6]:


bike.head()


# In[7]:


bike.info()


# In[8]:


bike.describe()


# Since the difference between mean and median is not significant , we can conclude that data has no outliers

# In[9]:


bike.shape


# ### Checking for any null values and duplicates

# In[10]:


round(100*(bike.isnull().sum()/len(bike)), 2).sort_values(ascending=False)


# In[11]:


round((bike.isnull().sum(axis=1)/len(bike))*100,2).sort_values(ascending=False)


# In[12]:


bike_dup = bike.copy()

bike_dup.drop_duplicates(subset=None, inplace=True)


# In[13]:


bike_dup.shape


# In[14]:


bike.shape


# Since the shape of the both bike_dup and bike is same, we can concude that there are not duplicates

# ### Data preparation 

# In[17]:


# Converting some numeric values to categorical data
import calendar
bike['mnth'] = bike['mnth'].apply(lambda x: calendar.month_abbr[x])


# In[19]:


bike.season = bike.season.map({1: 'Spring',2:'Summer',3:'Fall',4:'Winter'})


# In[20]:


bike.weathersit = bike.weathersit.map({1:'Clear',2:'Mist & Cloudy', 
                                             3:'Light Snow & Rain',4:'Heavy Snow & Rain'})


# In[21]:


bike.head()


# #### Creating Dummy variables 

# In[22]:


dummy = bike[['season','mnth','weekday','weathersit']]


# In[23]:


dummy = pd.get_dummies(dummy,drop_first=True )


# In[27]:


bike = pd.concat([dummy,bike],axis = 1)


# In[28]:


bike.head()


# #### Dropping columns for which the dummy variables are created 

# In[29]:


bike.drop(['season', 'mnth', 'weekday','weathersit'], axis = 1, inplace = True)


# In[30]:


bike.head()


# In[32]:


bike.shape


# ### Importing scikit learn libraries and splitting the data into train and test 

# In[33]:


import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score

from math import sqrt


# In[35]:


# Dropping irrelevant columns
bike.drop(['instant','dteday','casual','registered'],axis = 1,inplace = True)


# In[36]:


bike.head()


# In[37]:


train, test = train_test_split(bike, train_size = 0.7, test_size = 0.3, random_state = 100)


# ### Scaling the data 

# In[38]:


scaler = MinMaxScaler()


# In[39]:


num_vars = ['cnt','hum','windspeed','temp','atemp']

train[num_vars] = scaler.fit_transform(train[num_vars])


# In[40]:


train.head()


# In[41]:


train.describe()


# ### Visualizing the data 

# In[44]:


plt.figure(figsize = (20, 10))
sns.heatmap(train.corr(), annot = True)
plt.show()


# In[45]:


sns.pairplot(data=bike,vars=['cnt', 'temp', 'atemp', 'hum','windspeed'])
plt.show()


# ### Building the model

# In[48]:


X_train = train
y_train = train.pop('cnt')


# In[49]:


# Fit a regression line through the training data using statsmodels
lm = LinearRegression()
lm.fit(X_train, y_train)

#Running RFE
rfe = RFE(lm, 13)             
rfe = rfe.fit(X_train, y_train)


# In[50]:


list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[51]:


col = X_train.columns[rfe.support_]
col


# In[52]:


X_train.columns[~rfe.support_]


# In[53]:


X_train_rfe = X_train[col]


# In[54]:


X_train_rfe = sm.add_constant(X_train_rfe)


# In[55]:


lm = sm.OLS(y_train,X_train_rfe).fit()


# In[56]:


lm.params


# In[57]:


print(lm.summary())


# In[58]:


# dropping mnth_Ja since it has p > 0.05
X_train_new = X_train_rfe.drop(["mnth_Jan"], axis = 1)


# ### Rebuilding the model 

# In[59]:


X_train_lm = sm.add_constant(X_train_new)


# In[60]:


lm = sm.OLS(y_train,X_train_lm).fit()


# In[61]:


lm.summary()


# ### Checking VIF for multicollinearity 

# In[62]:


vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[63]:


X_train_new = X_train_new.drop(['const'], axis=1)


# Calculating VIF again after dropping variable with VIF greater than 5

# In[64]:


vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[65]:


#dropping hum from the model
X_train_new = X_train_new.drop(['hum'], axis=1)


# In[66]:


X_train_lm = sm.add_constant(X_train_new)


# In[67]:


lm = sm.OLS(y_train,X_train_lm).fit()  


# In[68]:


print(lm.summary())


# In[69]:


# Calculating the VIF again 
vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# ### Residual Analysis of the train data 

# In[70]:


y_train_cnt = lm.predict(X_train_lm)


# In[74]:


fig = plt.figure()
sns.distplot((y_train - y_train_cnt), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                 
plt.xlabel('Errors', fontsize = 20)  


# ### Making Predictions Using the Final Model 

# In[75]:


# Applying the scaling on the test sets
num_vars = ['cnt','hum','windspeed','temp','atemp']

test[num_vars] = scaler.transform(test[num_vars])


# In[76]:


test.describe()


# In[77]:


# Dividing into X_test and y_test

X_test = test
y_test = test.pop('cnt')


# In[78]:


X_test = sm.add_constant(X_test)


# ### Predicting using values used by the final model 

# In[79]:


test_col = X_train_lm.columns
X_test=X_test[test_col[1:]]

X_test = sm.add_constant(X_test)

X_test.info()


# In[80]:


y_pred = lm.predict(X_test)


# ### Evaluating the model 

# In[81]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[82]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
mse


# In[83]:


fig = plt.figure()
plt.scatter(y_test, y_pred)
fig.suptitle('y_test vs y_pred', fontsize = 20)             
plt.xlabel('y_test', fontsize = 20)                      
plt.ylabel('y_pred', fontsize = 20)  


# In[84]:


param = pd.DataFrame(lm.params)
param.insert(0,'Variables',param.index)
param.rename(columns = {0:'Coefficient value'},inplace = True)
param['index'] = list(range(0,12))
param.set_index('index',inplace = True)
param.sort_values(by = 'Coefficient value',ascending = False,inplace = True)
param


# Final Model Equation:
# cnt = 0.199648 + 0.491508 X temp + 0.233482 X yr + 0.083084 X seasonWinter - 0.066942 X season Spring + 0.083084 X season_Winter -0.052418 X mnth_Jul + 0.076686 X mnth_Sep -0.285155 X weathersit_Light Snow & Rain -0.081558 X weathersit_Mist & Cloudy -0.098013 X holiday -0.147977X windspeed

# All the positive coefficients like temp,season_Summer indicate that an increase in these values will lead to an increase in the value of cnt.
# All the negative coefficients like indicate that an increase in these values will lead to an increase in the value of cnt.

# Temp is the most significant with the largest coefficient.
# Followed by weathersit_Light Snow & Rain.
# Bike rentals is more for the month of september
# The rentals reduce during holidays

# This indicates that the bike rentals is majorly affected by temperature,season and month

# In[ ]:




