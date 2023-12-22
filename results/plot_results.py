# -*- coding: utf-8 -*-
"""  
File used to plot speed-up analysis.

Mehdi Anhichem.
University of Liverpool
21/12/2023
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

# Folder locations
barkla_path = '/users/anhichem/sharedscratch/programming'
local_path = 'C:/Users/anhic/Documents/00-LIVERPOOL_UNI/10-Programming/01-Project'

working_path = local_path # To change according to working platform

#==============================================================================
# READ RESULT FILES
#==============================================================================
os.chdir(working_path + '/MCDropout_MPI/')
#Importing the dataset
prediction_time_n2 = pd.read_csv('./results/prediction_time_study_n2.csv')
prediction_time_n4 = pd.read_csv('./results/prediction_time_study_n4.csv')
prediction_time_n6 = pd.read_csv('./results/prediction_time_study_n6.csv')
prediction_time_n8 = pd.read_csv('./results/prediction_time_study_n8.csv')

mc_samples = prediction_time_n2['mc_samples'].values
pred_time_serial = prediction_time_n2['pred_time_serial'].values
pred_time_n2 = prediction_time_n2['pred_time_parallel'].values
pred_time_n4 = prediction_time_n4['pred_time_parallel'].values
pred_time_n6 = prediction_time_n6['pred_time_parallel'].values
pred_time_n8 = prediction_time_n8['pred_time_parallel'].values

#==============================================================================
# PLOT SPEED-UP FIGURE
#==============================================================================
fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
plt.plot(mc_samples, pred_time_serial, marker='o')
plt.plot(mc_samples, pred_time_n2, marker='o')
plt.plot(mc_samples, pred_time_n4, marker='o')
plt.plot(mc_samples, pred_time_n6, marker='o')
plt.plot(mc_samples, pred_time_n8, marker='o')
plt.xscale('log')
plt.xlabel(r'Monte Carlo samples [-]', fontsize=23)
plt.ylabel(r'Prediction time [s]', fontsize=23)
plt.legend(['Serial','2 Cores','4 Cores','6 Cores','8 Cores'])
plt.grid(visible = True, linestyle = '--')
plt.savefig(working_path + '/MCDropout_MPI/results/prediction_time_study.png')

fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
# plt.plot(mc_samples, pred_time_serial, marker='o')
plt.plot(mc_samples, pred_time_serial/pred_time_n2, marker='o')
plt.plot(mc_samples, pred_time_serial/pred_time_n4, marker='o')
plt.plot(mc_samples, pred_time_serial/pred_time_n6, marker='o')
plt.plot(mc_samples, pred_time_serial/pred_time_n8, marker='o')
plt.xscale('log')
plt.xlabel(r'Monte Carlo samples [-]', fontsize=23)
plt.ylabel(r'Speed-up [-]', fontsize=23)
plt.legend(['2 Cores','4 Cores','6 Cores','8 Cores'])
plt.grid(visible = True, linestyle = '--')
plt.savefig(working_path + '/MCDropout_MPI/results/speedup_study.png')