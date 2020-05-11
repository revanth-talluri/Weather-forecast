# -*- coding: utf-8 -*-
"""
Created on Sun May  3 15:35:17 2020
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

#Multiple Linear Regression
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

#K-Fold cross validation
from sklearn.model_selection import cross_val_score


def get_train_valid(station):
    
    train = station.loc[station['Year']!=2017] 
    valid = station.loc[station['Year']==2017]
    
    train = train.drop(['Date','station','lat','lon','DEM','Slope','Year','Month','Day'], axis=1)
    valid = valid.drop(['Date','station','lat','lon','DEM','Slope','Year','Month','Day'], axis=1)
    
    return train, valid

def scale(X_train, X_test):
    
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)
    
    return X_train, X_test

def backElimination(X, Y_train, arr, get_arr):
    
    X_opt = X[:, arr]
    regressor_OLS = sm.OLS(endog = Y_train, exog = X_opt).fit()
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
    
    if(pop_item): return backElimination(X, Y_train, arr, get_arr)
    else: 
        get_arr.sort()        
        return results_summary

def plot(Y_pred_Tmax, Y_pred_Tmin, valid, iter_no):
    
    Y_pred_Tmax = pd.DataFrame(Y_pred_Tmax)
    Y_pred_Tmin = pd.DataFrame(Y_pred_Tmin)
    
    Y_pred_Tmax.columns = ['Predicted Next_Tmax']
    Y_pred_Tmin.columns = ['Predicted Next_Tmin']
    
    Y_pred = pd.concat([Y_pred_Tmax, Y_pred_Tmin], axis=1, join='inner')
    test = valid
    test.index = [k for k in range(0,len(test))]    
    
    #plotting the results
    nrows = 2
    ncols = 1
    fig, (ax1,ax2) = plt.subplots(nrows=nrows, ncols=ncols)
    fig.suptitle('Station {}'.format(iter_no))
    
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
    
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])   

def build_model(X_train, Y_train, X_test, sl):
    
    regressor = LinearRegression()
    regressor.fit(X_train, Y_train)
    
    Y_pred  = regressor.predict(X_test)
    
    accuracies = cross_val_score(estimator = regressor, X = X_train, y = Y_train, cv = 10)
    acc_mean = accuracies.mean()
    acc_std  = accuracies.std()
    
    kfold = {'acc_mean':acc_mean, 'acc_std':acc_std}
    
    X_train_mul = np.append(arr = np.ones((X_train.shape[0], 1)).astype(int), values = X_train, axis = 1)
    arr = []
    get_arr = []
    for i in range(0, X_train.shape[1]):
        arr.append(i)
        
    summary = backElimination(X_train_mul, Y_train, arr, get_arr)

    acc = round(regressor.score(X_train, Y_train)*100, 2)
    
    return summary, acc, Y_pred    

def run(station, sl, iter_no):
    
    train, valid = get_train_valid(station)
    
    X_train = train.iloc[:, :-2].values
    Y_train_Tmax = train.iloc[:, [-2]].values
    Y_train_Tmin = train.iloc[:, [-1]].values
    
    X_test = valid.iloc[:, :-2].values
    Y_test_Tmax = valid.iloc[:, [-2]].values
    Y_test_Tmin = valid.iloc[:, [-1]].values
    
    scaled_X_trian, scaled_X_test = scale(X_train, X_test) #Returns the scaled values of X_train and X_test
    
    summary_max, acc_max, Y_pred_Tmax = build_model(scaled_X_trian, Y_train_Tmax, scaled_X_test, sl)
    summary_min, acc_min, Y_pred_Tmin = build_model(scaled_X_trian, Y_train_Tmin, scaled_X_test, sl)
    
    summ = [summary_max, summary_min]
    a  = [acc_max, acc_min]
    
    #Plotting True v's Predicted values
    plot(Y_pred_Tmax, Y_pred_Tmin, valid, iter_no)
    
    return summ, a

if __name__ == '__main__':
    
    data_df = pd.read_csv('Bias_correction_ucl.csv', index_col = 'Date', parse_dates=True)
    
    data_df.reset_index(inplace=True)
    data_df['Year']  = data_df['Date'].dt.year
    data_df['Month'] = data_df['Date'].dt.month
    data_df['Day']   = data_df['Date'].dt.day
    
    #There are 25 stations, dividing data into respective stations
    station_info = [[] for _ in range(26)]
    
    for i in range(1,26):
        station_info[i] = data_df.loc[data_df['station'] == i]    
    
    #All the columns are float64 and since the std. deviaiton is less and the total no. of
    #missing values are les, filling all the nan values with the respective column avg.
    for i in range(1,26):
        station_info[i] = station_info[i].fillna(station_info[i].mean())
        #station[i].set_index('Date', inplace=True)
        
    targets  = ['Next_Tmax','Next_Tmin']
    features = ['Present_Tmax','Present_Tmin','LDAPS_RHmax','LDAPS_RHmin','LDAPS_Tmax_lapse',
                'LDAPS_Tmin_lapse','LDAPS_WS','LDAPS_LH','LDAPS_CC1','LDAPS_CC2','LDAPS_CC3',
                'LDAPS_CC4','LDAPS_PPT1','LDAPS_PPT2','LDAPS_PPT3','LDAPS_PPT4']
    
    acc = [i for i in range(0,len(station_info))] #To store accuracy values
    summary = [j for j in range(0,len(station_info))] #To store summary of the model
    sl = 0.05
    
    for iter_no in range(1,len(station_info)):  
        
        summary[iter_no], acc[iter_no] = run(station_info[iter_no], sl, iter_no)
    
    #Converting summary list to DataFrame object     
    summary[0] = [0,0]
    summary_df = pd.DataFrame(summary, columns=['Next_Tmax','Next_Tmin'])
    summary_df['Station'] = summary_df.index
    
    cols = summary_df.columns.to_list()
    cols = cols[-1:] + cols[:-1]
    
    summary_df = summary_df[cols]
    summary_df.drop(summary_df.head(1).index, inplace=True)
    summary_df.index = range(0,len(summary_df))
    
    #Converting accuracy list to DataFrame object  
    acc[0] = [0,0]
    acc_df = pd.DataFrame(acc, columns=['Next_Tmax','Next_Tmin'])
    acc_df['Station'] = acc_df.index
    
    cols = acc_df.columns.to_list()
    cols = cols[-1:] + cols[:-1]
    
    acc_df = acc_df[cols]
    acc_df.drop(acc_df.head(1).index, inplace=True)
    acc_df.index = range(0,len(acc_df))
    
    #Accuracy of predictions
    print('Accuracy at individual station is: ')
    print(acc_df)
    print() #h-space
    
    print('Avg accuracy for Next_Tmax is: {}'.format(acc_df['Next_Tmax'].mean()))
    print('Avg accuracy for Next_Tmin is: {}'.format(acc_df['Next_Tmin'].mean()))
    