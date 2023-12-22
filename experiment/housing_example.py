# -*- coding: utf-8 -*-
"""
Regression example applied on housing dataset. See other example on Kaggle:
https://www.kaggle.com/code/yasserh/housing-price-prediction-best-ml-algorithms
    
File used to plot convergence w.r.t to number of samples using a single process.

Mehdi Anhichem.
University of Liverpool
14/11/2023
"""

#==============================================================================
# IMPORT REQUIRED MODULES
#==============================================================================
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
# Plotting options
plt.rcParams["font.size"] = "20"
plt.rc('text', usetex=True)
plt.rcParams['font.family'] = 'serif'
plt.rcParams["mathtext.fontset"] = 'cm'
plt.rcParams['hatch.linewidth'] = 1.0
plt.rcParams["legend.frameon"] = 'True'
plt.rcParams["legend.fancybox"] = 'True'
plt.rcParams["figure.autolayout"] = 'True'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['pdf.fonttype'] = 42

# Split options
from sklearn.model_selection import train_test_split

# Folder locations
barkla_path = '/users/anhichem/sharedscratch/programming'
local_path = 'C:/Users/anhic/Documents/00-LIVERPOOL_UNI/10-Programming/01-Project'

working_path = local_path # To change according to working platform

import sys
sys.path.append(working_path + '/MCDropout_MPI/')
import bnn_mcdropout

#==============================================================================
# DATA IMPORTATION AND PROCESSING
#==============================================================================
os.chdir(working_path + '/MCDropout_MPI/')
#Importing the dataset
df = pd.read_csv('./data/Housing.csv')

target = 'price'
features = [i for i in df.columns if i not in [target]]

#Checking number of unique rows in each feature
nu = df[features].nunique().sort_values()
nf = []; cf = []; nnf = 0; ncf = 0; #numerical & categorical features
for i in range(df[features].shape[1]):
    if nu.values[i]<=16:cf.append(nu.index[i])
    else: nf.append(nu.index[i])

print('\n\033[1mInference:\033[0m The Datset has {} numerical & {} categorical features.'.format(len(nf),len(cf)))

#Check for empty elements
nvc = pd.DataFrame(df.isnull().sum().sort_values(), columns=['Total Null Values'])
nvc['Percentage'] = round(nvc['Total Null Values']/df.shape[0],3)*100

#Converting categorical Columns to Numeric
df_num = df.copy()
ecc = nvc[nvc['Percentage']!=0].index.values
fcc = [i for i in cf if i not in ecc]
#One-Hot Binary Encoding
oh=True
dm=True
for i in fcc:
    #print(i)
    if df_num[i].nunique()==2:
        if oh==True: print("\033[1mOne-Hot Encoding on features:\033[0m")
        print(i);oh=False
        df_num[i]=pd.get_dummies(df_num[i], drop_first=True, prefix=str(i))
    if (df_num[i].nunique()>2 and df_num[i].nunique()<17):
        if dm==True: print("\n\033[1mDummy Encoding on features:\033[0m")
        print(i);dm=False
        df_num = pd.concat([df_num.drop([i], axis=1), pd.DataFrame(pd.get_dummies(df_num[i], drop_first=True, prefix=str(i)))],axis=1)
# df_num = df_num.astype('float64')

#Splitting the data intro training & testing sets
X = df_num.drop([target],axis=1)
Y = df_num[target]
X_train_full, X_valid, y_train_full, y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.875, test_size=0.125, random_state=123)
X_train, y_train, X_valid, y_valid, X_test, y_test = X_train.values, y_train.values.reshape(-1,1), X_valid.values, y_valid.values.reshape(-1,1), X_test.values, y_test.values.reshape(-1,1)

print('Original set  ---> ',X.shape,Y.shape,
      '\nTraining set  ---> ',X_train.shape,y_train.shape,
      '\nValidation set  ---> ',X_valid.shape,y_valid.shape,
      '\nTesting set   ---> ', X_test.shape,'', y_test.shape)

#==============================================================================
# TRAIN BNN
#==============================================================================

# Train model
model_test = bnn_mcdropout.BNN_MCDropout(X_train = X_train, y_train = y_train, normalisation = True, name = 'model_test',
                                         X_valid = X_valid, y_valid = y_valid,
                                         hidden_layers = [512, 512, 512],
                                         dropout = 0.05, reg_length= 1.0, tau = 1.0,
                                         loss_fct = 'mean_squared_error')
print(model_test.model.summary())

history = model_test.fit(n_epochs = 1000)

# Save model
model_test.model.save(working_path+'/MCDropout_MPI/data/model_housing.h5')

#==============================================================================
# TEST CONVERGENCE W.R.T MC SAMPLES
#==============================================================================
# This section study the convergence of the BNN estimate depending on the number of MC samples

def test_cv(model, X, test_list):
    mean_cv_list, std_cv_list, pred_cv_time = [], [], []
    for n in test_list:
        start_time = time.time()
        mean_cv, std_cv = model.predict(X, n_mc_samples = n)
        pred_time = time.time() - start_time
        mean_cv_list.append(mean_cv[0])
        std_cv_list.append(std_cv[0])
        pred_cv_time.append(pred_time)
    return mean_cv_list, std_cv_list, pred_cv_time

mc_samples = [1, 3, 6, 10, 30, 60, 100, 300, 600, 1000, 3000, 6000, 10000]
mean_conv, std_conv, time_conv = test_cv(model_test, X_test, mc_samples)

fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
plt.plot(mc_samples, mean_conv, color='green', marker='o')
plt.axhline(y = 1.01*mean_conv[-1], color = 'k', linestyle = '--')
plt.axhline(y = 0.99*mean_conv[-1], color = 'k', linestyle = '--')
plt.xscale('log')
plt.xlabel(r'Monte Carlo samples [-]', fontsize=23)
plt.ylabel(r'Mean estimate ($\mu_{MCD}$) [-]', fontsize=23)
plt.legend(['Monte Carlo estimate', '$\pm$ 1\% of value with 10,000 samples'], loc = 4)
plt.grid(visible = True, linestyle = '--')
plt.show()

fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
plt.plot(mc_samples, std_conv, color='green', marker='o')
plt.axhline(y = 1.01*std_conv[-1], color = 'k', linestyle = '--')
plt.axhline(y = 0.99*std_conv[-1], color = 'k', linestyle = '--')
plt.xscale('log')
plt.xlabel(r'Monte Carlo samples [-]', fontsize=23)
plt.ylabel('Standard deviation estimate ($\sigma_{MCD}) [-]$', fontsize=23)
plt.legend(['Monte Carlo estimate', '$\pm$ 1\% of value with 10,000 samples'], loc = 4)
plt.grid(visible = True, linestyle = '--')
plt.show()  

#==============================================================================
# MSE PREDICTION ON TEST SET
#==============================================================================
# Compute model's MSE on test set

start_time = time.time()
mean_test, std_test = model_test.predict(X_test, n_mc_samples = 1000)
pred_time = time.time() - start_time

mse_test = np.mean((mean_test - y_test)**2)
print('MSE on test set = {}'.format(mse_test))
print('Prediction time on test set = {}'.format(pred_time))