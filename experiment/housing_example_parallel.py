"""
Regression example applied on housing dataset. See other example on Kaggle:
https://www.kaggle.com/code/yasserh/housing-price-prediction-best-ml-algorithms
    
File used to study speed-up based on MPI.

Mehdi Anhichem.
University of Liverpool
15/11/2023
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

from sklearn.model_selection import train_test_split

# Folder locations
barkla_path = '/users/anhichem/sharedscratch/programming'
local_path = 'C:/Users/anhic/Documents/00-LIVERPOOL_UNI/10-Programming/01-Project'

working_path = local_path # To change according to working platform

import sys
sys.path.append(working_path + '/MCDropout_MPI/')
import bnn_mcdropout

# Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#==============================================================================
# MPI
#==============================================================================
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
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
        df_num[i]=pd.get_dummies(df_num[i], drop_first=True, prefix=str(i))
    if (df_num[i].nunique()>2 and df_num[i].nunique()<17):
        df_num = pd.concat([df_num.drop([i], axis=1), pd.DataFrame(pd.get_dummies(df_num[i], drop_first=True, prefix=str(i)))],axis=1)

#Splitting the data intro training & testing sets
X = df_num.drop([target],axis=1)
Y = df_num[target]
X_train_full, X_valid, y_train_full, y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.875, test_size=0.125, random_state=123)
X_train, y_train, X_valid, y_valid, X_test, y_test = X_train.values, y_train.values.reshape(-1,1), X_valid.values, y_valid.values.reshape(-1,1), X_test.values, y_test.values.reshape(-1,1)

# if rank == 0:
#     print('Original set  ---> ',X.shape,Y.shape,
#         '\nTraining set  ---> ',X_train.shape,y_train.shape,
#         '\nValidation set  ---> ',X_valid.shape,y_valid.shape,
#         '\nTesting set   ---> ', X_test.shape,'', y_test.shape)

#==============================================================================
# LOAD BNN
#==============================================================================

# Train model
model_test = bnn_mcdropout.BNN_MCDropout(X_train = X_train, y_train = y_train, normalisation = True, name = 'model_test',
                                         X_valid = X_valid, y_valid = y_valid,
                                         hidden_layers = [512, 512, 512],
                                         dropout = 0.05, reg_length= 1.0, tau = 1.0,
                                         loss_fct = 'mean_squared_error')

if rank == 0:
    print(model_test.model.summary())

model_test.load(working_path+'/MCDropout_MPI/data/model_housing.h5')

# # Uncomment below for quick comparison with 1000 samples
# start_time = time.time()
# mean_temp, std_temp = model_test.predict_parallel(X_test, comm, rank, size, n_mc_samples = 1000)
# pred_time_parallel_temp = time.time() - start_time
# if rank == 0:
#     print(mean_temp, pred_time_parallel_temp)
# if rank == 0:
#     start_time = time.time()
#     mean_temp, std_temp = model_test.predict(X_test, n_mc_samples = 1000)
#     pred_time_temp = time.time() - start_time
#     print(mean_temp, pred_time_temp)

#==============================================================================
# PREDICTION TIME STUDY ON TEST SET
#==============================================================================

# Comment below not to run comparison with various number of samples
mc_samples = [10, 30, 60, 100, 300, 600, 1000, 3000, 6000, 10000]
if rank == 0:
    pred_time_serial, pred_time_parallel = [], []
for n in mc_samples:
    if rank == 0:
        start_time = time.time()
        mean_temp, std_temp = model_test.predict(X_test, n_mc_samples = n)
        pred_time_serial_temp = time.time() - start_time
        pred_time_serial.append(pred_time_serial_temp)

     
    start_time = time.time()
    mean_temp, std_temp = model_test.predict_parallel(X_test, comm, rank, size, n_mc_samples = n)
    pred_time_parallel_temp = time.time() - start_time
    if rank == 0:  
        pred_time_parallel.append(pred_time_parallel_temp)
if rank == 0:
    df_pred_time = pd.DataFrame({'mc_samples':mc_samples, 'pred_time_serial': pred_time_serial, 'pred_time_parallel': pred_time_parallel} )
    df_pred_time.to_csv(working_path + '/MCDropout_MPI/results/prediction_time_study_n{}.csv'.format(size), index=False)

