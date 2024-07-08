#!/usr/bin/env python
# coding: utf-8

# In[8]:


# BNB/BTC Time series exploration 


# In[9]:


# Step 1 import all the extensions


# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
from math import sqrt
from sklearn.metrics import r2_score , mean_absolute_error , mean_absolute_percentage_error , mean_squared_error
import warnings
warnings.filterwarnings('ignore')


# In[11]:


# change the data directory to the homework folder


# In[36]:


os.chdir('C:\\Users\Boris\Desktop\Homework')
df = pd.read_csv('BNB-BTC-Test.csv')
df.head()


# In[13]:


# Exploratory Data Analysis


# In[16]:


df.shape


# In[17]:


# Check for nulls (although redundant)


# In[18]:


df.isnull().sum()


# In[19]:


#Check for dublicates 


# In[21]:


df.duplicated().sum()


# In[30]:


df.info()


# In[24]:


# Need to convert Day to time datetime


# In[28]:


df['Day'] = pd.to_datetime(df['Day'])


# In[31]:


df.info()


# In[32]:


#Looks good now lets looks at all data 


# In[49]:


df.describe(include='all')



# In[40]:


# Day describes date BTC + BNB Vol are daily volumes, BTC+BNB Close are the closing prices for a given day.
# Diff is % difference from previous close and ex is the BNB/BTC exchange Rate


# In[42]:


sns.pairplot(data=df);
#Lets take a look at the relationships of different variables. As we can see there already a postiive relationsahip between BTC  and BNB volume


# In[58]:


#We Are specifically interested in the exchange rate however, so lets plot it
# First lets set the day as index 
# I has issued separating the variable here. In the interest of time I created a different table


# In[89]:


os.chdir('C:\\Users\Boris\Desktop\Homework')
df = pd.read_csv('BNBEX.csv')
df.head()


# In[90]:


df['Date'] = pd.to_datetime(df['Date'])


# In[91]:


df.set_index('Date',inplace = True)


# In[92]:


df.plot(figsize = (10,5))
plt.title('BNB-BTC Exchange Time series ')
plt.show()


# In[93]:


#Lets see the distribution
sns.distplot(df)
plt.title('Distribution of BNB/BTC')
plt.show()


# In[94]:


# We can that the data is roughly normally distributed although it is scewed. IF we had more data it would most likely
#look more normally distributed
#Now lets check for autocorrelation
fig , (ax1,ax2) = plt.subplots(nrows = 2 ,ncols = 1,sharex = False , sharey = False , figsize = (10,5))
ax1 = plot_pacf(df , lags = 5 , ax = ax1)
ax2 = plot_acf(df , lags = 5 , ax = ax2)
plt.show()


# In[66]:


#Now we need to check if the data is stationary in order to exclude the effect of seasonality.
# For this we use the Dickley-Fuller test. 
#If the p val is less that or = to 0.05 there is evidence against the null hypthesis and the series is stanionary.


# In[95]:


# Define the test
def adf_check(time_series):
    result = adfuller(time_series , autolag = 'AIC')
    label = pd.Series(result[0:4], index=['Test Statistic','p-value','Number of Lags Used','Number of Observations Used'])
    for key,value in result[4].items():
        label['Critical Value (%s)'%key] = value
    print(label)
    if result[1] <= 0.05:
        print('Strong evidence against the null hypothesis, hence REJECT null hypothesis and the series is Stationary')
    else:
        print ('Weak evidence against the null hypothesis, hence ACCEPT null hypothesis and the series is Not Stationary ')


# In[96]:


adf_check(df)


# In[97]:


#Hmm this means there is seasonality present. What if we resample the dataset with first difference to make it stationary

df1 = df.diff().dropna()
print('Count of weekly First Difference',df1.shape[0])
df1.head()
adf_check(df1)


# In[98]:


#Right now that we can assume the data is not affected by seasonality. 
#Lets see how our data looks stationary vs non stationary

fig , (ax1,ax2) = plt.subplots(nrows = 2 ,ncols = 1,sharex = False , sharey = False , figsize = (10,5))
ax1 = autocorrelation_plot(df , ax = ax1)
ax1.set_title('Non-Stationary Data')
ax2 = autocorrelation_plot(df1 , ax = ax2)
ax2.set_title('Stationary Data')
plt.subplots_adjust(hspace = 0.5)
plt.show()


# In[99]:


#We need to fit the model for the p and q values using Auto-arima function.
# For this model we will AIC - Auaike Information Criterion
model = auto_arima(df, m = 52, d = 1 ,seasonal = False , max_order = 8 , test = 'adf' , trace = True)


# In[100]:


model.summary()


# In[101]:


# Now lets fit Arima with the Best value  we got from the auto_arima.

model = ARIMA(df, order = (0,1,1))
result = model.fit()
result.summary()


# In[102]:


# Now lets plot the model
result.plot_diagnostics()
plt.subplots_adjust(hspace = 0.5)
plt.show()


# In[103]:


# Now try predicting the value with the new data
predictions = result.predict(typ = 'levels')


# In[86]:


print('Evaluation Result for whole data : ','\n')
print('R2 Score for whole data : {0:.2f} %'.format(100*r2_score(df['Value'],predictions)),'\n')
print('Mean Squared Error : ',mean_squared_error(df['Value'],predictions),'\n')
print('Mean Absolute Error : ',mean_absolute_error(df['Value'],predictions),'\n')
print('Root Mean Squared Error : ',sqrt(mean_squared_error(df['Value'],predictions)),'\n')
print('Mean Absolute Percentage Error : {0:.2f} %'.format(100*mean_absolute_percentage_error(df['Value'],predictions)))


# In[87]:


# Lets display the resulting data 


# In[104]:


Final_data = pd.concat([df,df1,predictions],axis=1)
Final_data.columns = ['BNB-BTC Exchange','First Difference','Predicted Exchange']
Final_data.head()


# In[105]:


# Now lets test the model with training data and test data.

size = int(len(df)*0.80)
train , test = df[0:size]['Value'] , df[size:(len(df))]['Value']
print('Counts of Train Data : ',train.shape[0])
print('Counts of Train Data : ',test.shape[0])


# In[106]:


# We will create the list of training values in train_values as well as predictions. 
# This list we will then later append after the prediction
#We will then use the best fit values with Arima in the train_values and predict the best value.
# We will then print a predictions list


# In[107]:


train_values = [x for x in train]
prediction = []
print('Printing Predictied vs Expected Values....')
print('\n')
for t in range(len(test)):
    model = ARIMA(train_values , order = (0,1,1))
    model_fit = model.fit()
    output = model_fit.forecast()
    pred_out = output[0]
    prediction.append(float(pred_out))
    test_in = test[t]
    train_values.append(test_in)
    print('Predicted = %f , Actual = %f' % (pred_out , test_in))


# In[108]:


# Now lets evaluate the model with standart metrics like: 
# r square, 
# mean square, 
# mean absolute error, 
# mean absolute percentage error
# Although I understand that due to limited amount of data the values are doing to be off, I will continue the test anyway


# In[109]:


print('Evaluation Result for Test data : ','\n')
print('R2 Score for Test data : {0:.2f} %'.format(100*r2_score(test,prediction)),'\n')
print('Mean Squared Error : ',mean_squared_error(test,prediction),'\n')
print('Mean Absolute Error : ',mean_absolute_error(test,prediction),'\n')
print('Root Mean Squared Error : ',sqrt(mean_squared_error(test,prediction)),'\n')
print('Mean Absolute Percentage Error : {0:.2f} %'.format(100*mean_absolute_percentage_error(test,prediction)))


# In[110]:


# As we can see our model is very erroneous. 
# According to r2 our data only explain 69% of the data
# MAPE is not zero but is very close
# MSE forecast is poor. Closer to 0 means fewer unaccounted data and better predictor
# MAE shows that our predictions are not too far off as the average between value and predictor is close to zero
# RMSE likewise shows that our prediction error is low


# In[111]:


# Now lets replace test data with prediction data and plot them
predictions_df = pd.Series(prediction, index = test.index)


# In[113]:


plt.rcParams['figure.figsize'] = (12,6)
fig, ax = plt.subplots()
ax.set(title='Foreign Exchange Rate Prediction, Euro to USD', xlabel='Date', ylabel='Foreign Exchange Rate')
ax.plot(df, 'o', label='Actual')
ax.plot(predictions_df, 'r', label='forecast')
legend = ax.legend(loc='upper left')
legend.get_frame().set_facecolor('w')


# In[ ]:


# As we can see we have built a good prediction model for the data despite its poor availability.
# This is by no means full-proof model and doesnt account for the many exogenous factors as well the underlying covariance
# of the two currencies.

