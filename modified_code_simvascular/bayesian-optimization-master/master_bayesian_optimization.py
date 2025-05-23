# -*- coding: utf-8 -*-
"""
Main code for carrying out Bayesian optimization (BO) to estimate Surface Evolver model parametsrs from
wing imaginal disc tissue cross section data

Created on Tue Jul  6 23:35:16 2021

Originally created by: Nilay Kumar
@author: Nilay Kumar
email: nkumar4@nd.edu
Multicellular Systems Engineering Lab (MSELab)
Department of Chemical and Biomolecular Engineering
Institution: University of Notre Dame

Modified: 2025-03-14
Modified by: Stephen Cini
@author: Stephen Cini
email: scini@nd.edu
Dowling Lab
Department of Chemical and Biomolecular Engineering
Institution: University of Notre Dame
"""

# Adding dependencies folder to the path. Dependencies stores all the classes used in bayesian optimization BO
import sys
sys.path.append("/home/nkumar4/Desktop/")

# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
# import spatial_efd
import math 
# import signac
import numpy as np
import os.path
import os
import torch
import gpytorch
import subprocess
import gc
# import similaritymeasures
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from smt.sampling_methods import LHS
# Importing helper libraries for bayesian optimization
from dependencies.data_preprocessing_class import DataPreprocessing
from dependencies.gaussian_process_regression_class import GaussianProcessRegression
from dependencies.acquisition_functions_class import AcqisitionFunctions
from dependencies.geometry_writer import GeometryWriter
from dependencies.feature_extractor_4 import FeatureExtractor

"""
User derived inputs
"""
# Number of parameters that need to be estimated during BO 
num_parameters_LHS = 5
# List containing parameter indices that need to be estimated using the BO framework
LHS_parameter_index = [0, 1, 2, 3, 4]
# Path to the target simulation data for which the parameters need to be estimated
# It should be a .txt file containing time and pressure/flow data for the target simulation
simulation_data = 'input_data/target_simulation.txt'
# simulation_data_type = 1: data belongs to an experimental simulation
# simulation_data_type = 2: synthetic data from a simvascular onedsolver output
simulation_data_type = 1
# Total number of parameter sets sampled during calculation of acquisition function
num_samples_af = 100000
# Total number of iterations for the BO process
n_iterations = 3
# Number of samples from the total samples that will constitute the training data
split_size = 149
# Total number of data points that are used for training the GP model
num_samples = 150
# A sample parameter set where all the parameters except the ones varied are kept similar to one during the LHS sampling for generating training data.
paraminputs_stable = [0.5, 0.01, 0.01, 0.01, 0.01]
# A parameter to define the tradeoff between exploration and exploitation during BO
exploration_param_val = 0.05
# Total number of iterations for training the GP model
num_iteration_gpr = 5000
# Selecting the type of optimizer used for training of GP model 
# 1: Adam Optimizer 2: LBFGS 
optimizer_type = 1
# Name of the file generated as a simvascular onedsolver output
# NOTE: Make sure to change the next line if changing this
sv_filename = 'simvascular_output'
# Path containing the simvascular installation and solver filename to run it
sv_path = "/path/to/simvascular/onedsolver"

"""
STEP 1:
Load the input and output data generated by the simvascular model for building a GPR model.
The input data should consist of a [num_samples x 5] and the output data contains the error metrics
should be of size [num_samples x 1]
"""
# Checking if data exists
doesDataFileExist = os.path.isfile("input_data/master_feature_output.npy")
# Loading datafiles if they exist
# Else fetching and preparing data from the workspace
if doesDataFileExist:
	master_parameter_input_n = np.load('input_data/master_parameter_input_n.npy')
	master_feature_output = np.load('input_data/master_feature_output.npy')

"""
STEP 2: 
Input data preprocessing - Preparing inputs and outputs for the GPR model
1. Input data: Selects the parameters sampled in LHS from total 5 parameters of the simvascular model. 
   The resulting input data to the GP model should be of size [num_samples x 5]
2. Output data: Target simulation data for which parameter estimation has to be carried out is loaded first.
   A suitable error metric (e.g., RMSE) is used to evaluate the error between the sampled parameters and target simulation.
   Lastly, a negative of the error metric is taken that constitutes the output training data for the GP model.
"""
# Loading in the data processing class
dataPreprocess = DataPreprocessing(master_parameter_input_n, master_feature_output, num_samples)
# Converting the input parameters to logscale
master_parameter_input_log = dataPreprocess.input_log_transform()
# Calling in the function to separate out the desired parameters
data_x = dataPreprocess.input_parameter_selection(num_parameters_LHS, LHS_parameter_index, master_parameter_input_log)
# Storing mean and standard deviation of input training data for later use
data_x_mean = np.mean(data_x, axis=0)
data_x_variance = np.std(data_x, axis=0)
# Normalizing data
data_x = StandardScaler().fit_transform(data_x)
# Calculating the minimum and maximum for input parameters for the purpose of sampling the parameters using LHS later
max_data_x = np.amax(data_x, axis=0) 
min_data_x = np.amin(data_x, axis=0) 

# Reading in experimental data as time-pressure/flow data
if simulation_data_type == 1:
	if type(simulation_data) is str:
		# Reading in data as two separate lists of time and pressure/flow values
		if os.stat(simulation_data).st_size != 0:
			time_data = []
			value_data = []
			with open(simulation_data) as f:
				for line in f:
					data = line.split()
					time_data.append(float(data[0]))
					value_data.append(float(data[1]))
		else:
			time_data = []
			value_data = []

	target_time = np.array(time_data)
	target_values = np.array(value_data)

# Synthetic data from simvascular onedsolver
elif simulation_data_type == 2:
	# Load synthetic data from a predefined format
	target_time, target_values = load_synthetic_data(simulation_data)

# Calculating the error metric (e.g., RMSE) between the target simulation and the simulation data present in the master_feature_output data
error_simulation_target_data = np.zeros(num_samples)
for i in range(num_samples):
	simulated_values = master_feature_output[i, :]
	error_simulation_target_data[i] = np.sqrt(np.mean((target_values - simulated_values) ** 2))

# Taking a negative of the error metric to generate the input data for the GP model
data_y = (np.reshape(error_simulation_target_data, (num_samples, 1))) * (-1)
print(np.shape(data_y))
