# -*- coding: utf-8 -*-
"""
Created on Sat May  2 23:19:15 2020
@author: revanth
"""

#linear algebra
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 25)

#Plotting
import seaborn as sns
import matplotlib.pyplot as plt

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler

#For Multivariate Linear Regression
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

#K-Fold cross validation
from sklearn.model_selection import cross_val_score

df = pd.read_csv('Bias_correction_ucl.csv', index_col = 'Date', parse_dates=True)

df.reset_index(inplace=True)
df['Year']  = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day']   = df['Date'].dt.day

#There are 25 stations, dividing data into respective stations
station = [[] for _ in range(26)]

for i in range(1,26):
    station[i] = df.loc[df['station'] == i]

#All the columns are float64 and since the std. deviaiton is less and the total no. of
#missing values are les, filling all the nan values with the respective column avg.
for i in range(1,26):
    station[i] = station[i].fillna(station[i].mean())
    #station[i].set_index('Date', inplace=True)    

targets  = ['Next_Tmax','Next_Tmin']
features = ['Present_Tmax','Present_Tmin','LDAPS_RHmax','LDAPS_RHmin','LDAPS_Tmax_lapse',
            'LDAPS_Tmin_lapse','LDAPS_WS','LDAPS_LH','LDAPS_CC1','LDAPS_CC2','LDAPS_CC3',
            'LDAPS_CC4','LDAPS_PPT1','LDAPS_PPT2','LDAPS_PPT3','LDAPS_PPT4']

#let's predict the Next_Tmax and Next_Tmin for 2017 at station-1 
train = station[1].loc[station[1]['Year']!=2017]  
valid = station[1].loc[station[1]['Year']==2017]

train = train.drop(['Date','station','lat','lon','DEM','Slope','Year','Month','Day'], axis=1)
valid = valid.drop(['Date','station','lat','lon','DEM','Slope','Year','Month','Day'], axis=1)

X_train = train.iloc[:, :-2].values
Y_train_Tmax = train.iloc[:, [-2]].values
Y_train_Tmin = train.iloc[:, [-1]].values

X_test = valid.iloc[:, :-2].values
Y_test_Tmax = valid.iloc[:, [-2]].values
Y_test_Tmin = valid.iloc[:, [-1]].values

#Normalizing the data
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)


'''
Multiple Linear Regression
'''

regressor_max = LinearRegression()
regressor_max.fit(X_train, Y_train_Tmax)

regressor_min = LinearRegression()
regressor_min.fit(X_train, Y_train_Tmin)

Y_pred_Tmax  = regressor_max.predict(X_test)
Y_pred_Tmin  = regressor_min.predict(X_test)
#Y_pred_train = regressor.predict(X_train)

#K-fold cross validation
accuracies_max = cross_val_score(estimator = regressor_max, X = X_train, y = Y_train_Tmax, cv = 10)
acc_mean_max = accuracies_max.mean()
acc_std_max  = accuracies_max.std()


accuracies_min = cross_val_score(estimator = regressor_min, X = X_train, y = Y_train_Tmin, cv = 10)
acc_mean_min = accuracies_min.mean()
acc_std_min  = accuracies_min.std()

kfold = {'acc_mean_max':acc_mean_max, 'acc_std_max':acc_std_max,
         'acc_mean_min':acc_mean_min, 'acc_std_min':acc_std_min}



#Backelimination for Tmax
X_train_mul = np.append(arr = np.ones((X_train.shape[0], 1)).astype(int), values = X_train, axis = 1)
sl = 0.05
arr_max = []
get_arr_max = []
for i in range(0, X_train.shape[1]):
    arr_max.append(i)

def backElimination_Tmax(X, arr, get_arr):
    
    X_opt = X[:, arr]
    regressor_OLS = sm.OLS(endog = Y_train_Tmax, exog = X_opt).fit()
    results_summary = regressor_OLS.summary()

    results_as_html = results_summary.tables[1].as_html()
    dfs = pd.read_html(results_as_html, header=0, index_col=0)[0]

    p_value =  dfs.iloc[:, 3]
    p_value_arr = p_value.tolist()
    pop_item = False
    maxpos  = p_value_arr.index(max(p_value_arr))
    if(p_value_arr[maxpos]>sl):
        get_arr.append(arr[maxpos])
        arr.pop(maxpos)
        pop_item = True
    
    if(pop_item): return backElimination_Tmax(X, arr, get_arr)
    else: 
        get_arr.sort()
        return results_summary      
      
summary_max = backElimination_Tmax(X_train_mul, arr_max, get_arr_max)

acc_max = round(regressor_max.score(X_train, Y_train_Tmax)*100, 2)


#Backelimination for Tmin
X_train_mul = np.append(arr = np.ones((X_train.shape[0], 1)).astype(int), values = X_train, axis = 1)
sl = 0.05
arr_min = []
get_arr_min = []
for i in range(0, X_train.shape[1]):
    arr_min.append(i)

def backElimination_Tmin(X, arr, get_arr):
    
    X_opt = X[:, arr]
    regressor_OLS = sm.OLS(endog = Y_train_Tmin, exog = X_opt).fit()
    results_summary = regressor_OLS.summary()

    results_as_html = results_summary.tables[1].as_html()
    dfs = pd.read_html(results_as_html, header=0, index_col=0)[0]

    p_value =  dfs.iloc[:, 3]
    p_value_arr = p_value.tolist()
    pop_item = False
    maxpos  = p_value_arr.index(max(p_value_arr))
    if(p_value_arr[maxpos]>sl):
        get_arr.append(arr[maxpos])
        arr.pop(maxpos)
        pop_item = True
    
    if(pop_item): return backElimination_Tmax(X, arr, get_arr)
    else: 
        get_arr.sort()
        return results_summary      
      
summary_min = backElimination_Tmin(X_train_mul, arr_min, get_arr_min)

acc_min = round(regressor_min.score(X_train, Y_train_Tmin)*100, 2)

#Accuracies for Tmax and Tmin
acc = {'Tmax accuarcy':acc_max, 'Tmin_accuracy':acc_min}

Y_pred_Tmax = pd.DataFrame(Y_pred_Tmax)
Y_pred_Tmin = pd.DataFrame(Y_pred_Tmin)

Y_pred_Tmax.columns = ['Predicted Next_Tmax']
Y_pred_Tmin.columns = ['Predicted Next_Tmin']

Y_pred = pd.concat([Y_pred_Tmax, Y_pred_Tmin], axis=1, join='inner')
test = valid
test.index = [i for i in range(0,len(test))]

#plotting the results
nrows = 2
ncols = 1
fig, (ax1,ax2) = plt.subplots(nrows=nrows, ncols=ncols)

ax1.plot(test['Next_Tmax'], label='True value')
ax1.plot(Y_pred['Predicted Next_Tmax'], label='Predicted')
ax1.title.set_text('Tmax - True v/s Predicted')
ax1.set_xticks([])
ax1.tick_params(axis='y', labelsize=5)
ax1.legend(loc='upper right')

ax2.plot(test['Next_Tmin'], label='True value')
ax2.plot(Y_pred['Predicted Next_Tmin'], label='Predicted')
ax2.title.set_text('Tmin - True v/s Predicted')
ax2.set_xticks([])
ax2.tick_params(axis='y', labelsize=5)
ax2.legend(loc='upper right')

#This Multiple Linear Regression model has an accuracy of 80% for predicting Next_Tmax and
#86% accuracy for predicting Next_Tmin at station-1
#This model can be applied at other stations also and the accuracies can be observed
