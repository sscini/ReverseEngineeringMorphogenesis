# -*- coding: utf-8 -*-
"""
Code Summary: The code is used to generate the data for hessian analysis of parameters.
The parametsrs in  a user defined list are perturbed two at a time by increasing and decreasing them of
the 70% of the value. For each parameter combination Surface Evolver is un and the frechet distance is
calculated from teh base simulation case.The results are stored and later used for calculating hessian.

The inputs in the particular code were used to generate Figure 3E within teh manuscript.
Refer hessian_analysis.ipynb for the analysis part

Created on Tue Jul  6 23:35:16 2021

@author: Nilay Kumar
email: nkumar4@nd.edu
Multicellular Systems Engineering Lab (MSELab)

Updates started Tue Jan 27 2026
Updated by Stephen Cini
email: scini@nd.edu
Dowling Lab

Department of Chemical and Biomolecular Engineering
Institution: University of Notre Dame
"""

# Adding dependencies folder to the path. Dependencies stores all teh classes used in bayesian optimization BO
import sys

sys.path.append(
    "/Users/scini/Documents/GitHub/ReverseEngineeringMorphogenesis/multiobjective_update_StephenCini2026"
)

# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import spatial_efd
import math
import time
import signac
import numpy as np
import os.path
import os
import torch
import gpytorch
import subprocess
import gc
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from smt.sampling_methods import LHS

# Importing helper libraries for bayesian optimization
from dependencies.data_preprocessing_class import DataPreprocessing
from dependencies.gaussian_process_regression_class import GaussianProcessRegression
from dependencies.acquisition_functions_class import AcqisitionFunctions
from dependencies.backends import get_backend
from dependencies.scorers import score_frechet_distance
from dependencies.workflow_artifacts import archive_artifact, save_contour_overlay_plot
from parameter_samplin import SampleParameters

"""
User Inputs
"""
# Number of parameters varied in LHS
n_param_model = 7
# Indices of the parameters varied during the LHS
param_index_model = [17, 18, 19, 28, 29, 30, 33]
# Parametsr of the mdoel around which hessian is being computed
param_inputs_stable = [
    0,
    3.162277660168379e-08,
    0,
    0,
    3.162277660168379e-08,
    0,
    0,
    1e-05,
    0,
    0,
    0,
    0.1,
    0.1,
    10,
    0.1,
    0.1,
    0.1,
    0.01,
    3.9810717055349695e-05,
    0.01,
    0.001,
    0.001,
    1,
    1,
    0.6,
    0.6,
    0.6,
    0.6,
    0.2,
    0.1,
    3,
    0.6,
    1.8,
    0.002511886431509582,
    0.001,
]
# Defining the set system pressure
param_pressure = 0.001
# Filename for saving SE file
se_filename = 'wingDisc'
BACKEND = get_backend("SurfaceEvolver", se_filename=se_filename, num_harmonics_efd=20)

"""
STEP 1: Reading in the target shape data (point around which sensitivity has to be calculated)
"""
# Reading the vertices output file from a sample SE simulation output with known parameters
target_features = BACKEND.extract_features('input_data/vertices_target_SE.txt')
coeffs_exp_basal = target_features["efd"]["basal_normalized_coefficients"]
coeffs_exp_apical = target_features["efd"]["apical_normalized_coefficients"]
# Obtaining normalized x and y coordinates for the bsaal surface of the tissue
xt_exp_apical, yt_exp_apical = spatial_efd.inverse_transform(
    coeffs_exp_apical, harmonic=20
)
exp_data_apical = np.zeros((300, 2))
exp_data_apical[:, 0] = xt_exp_apical
exp_data_apical[:, 1] = yt_exp_apical

xt_exp_basal, yt_exp_basal = spatial_efd.inverse_transform(
    coeffs_exp_basal, harmonic=20
)
exp_data_basal = np.zeros((300, 2))
exp_data_basal[:, 0] = xt_exp_basal
exp_data_basal[:, 1] = yt_exp_basal

"""
STEP 2: Hessian analysis
Utilized for computing the upper diagonal components of the Hessian matrix, the process involves a systematic iteration 
through the parameters.The initial iteration encompasses all parameters. Subsequently, an inner loop navigates through parameters following 
the one in the main loop, which are pertinent to Hessian computation.Within each parameter combination, adjustments 
are made to the parameters, both collectively and individually, to facilitate Hessian calculations. The specific parameter
variations are detailed in the parameter_sampling function, further elucidated within the hessian_analysis.ipynb file 
situated within the designated folder.
"""

# initializing the error array for storing the frechet errors
error_target_sampled_apical = []
error_target_sampled_basal = []
# Initializing the array for storing the parameter values
param_sampled = np.zeros((300, 35))
curvature_basal_master = np.zeros((300, 129))
tissue_edge_length_master = np.zeros((300, 390))

# Counter for iterations
k = 0
# Iterating over all theparametsr sin the list
for i in range(n_param_model - 1):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # Iterating over all teh parametsrs after the parameter in main loop
    for j in range(i + 1, n_param_model):
        # For parametsr i and j, each of them is increased and decreased by 70% and SE is run
        for hess_ctr in range(4):
            """
            STEP 2B : Defining parameters for the sensitivity analysis
            """
            paraminputsMaster = param_inputs_stable.copy()
            # Setting the param of interest iteratively as the lower and upper bounds
            paraminputs = SampleParameters(
                paraminputsMaster, param_index_model, i, j, hess_ctr
            )
            # Saving the parameter values
            param_sampled[k, :] = paraminputs

            """
			STEP 2B :  Running surface evolver and extracting shape information
			"""
            # Writing geometry file
            parsed_result = BACKEND.execute_candidate(paraminputs, param_pressure)
            efd_coeff_sampled_basal = parsed_result.features["efd"][
                "basal_normalized_coefficients"
            ]
            efd_coeff_sampled_apical = parsed_result.features["efd"][
                "apical_normalized_coefficients"
            ]
            xt_sampled_apical, yt_sampled_apical = spatial_efd.inverse_transform(
                efd_coeff_sampled_apical, harmonic=20
            )
            xt_sampled_basal, yt_sampled_basal = spatial_efd.inverse_transform(
                efd_coeff_sampled_basal, harmonic=20
            )
            # Local curvature along the exterior pouch surface (129 points in total)
            curvature_basal_sampled = parsed_result.features["curvature"]
            curvature_basal_sampled_reshaped = np.reshape(
                curvature_basal_sampled, (1, 129)
            )
            curvature_basal_master[k, :] = curvature_basal_sampled_reshaped
            # Lengths of all the edges used to make the SE model (390 in total)
            tissue_edge_length_sampled = parsed_result.features["edge_length"]
            tissue_edge_length_sampled_reshaped = np.reshape(
                tissue_edge_length_sampled, (1, 390)
            )
            tissue_edge_length_master[k, :] = tissue_edge_length_sampled_reshaped
            # MOving the counter forward
            k = k + 1

            """
			STEP 3C: Plotting and saving the sampled and target shape
			"""
            # Defining filename for plot showing overlap between the sampled shape and the target shape

            contour_plot_folder = "/Users/scini/Library/CloudStorage/GoogleDrive-scini@nd.edu/Shared drives/Stephen Cini Research/Projects/eMB/ucr_data/contour_evolution_plots"
            run_time_folder_contour= os.path.join(contour_plot_folder, "hrun_" + timestamp)
            os.makedirs(run_time_folder_contour, exist_ok=True)

            filename_shape_plot_apical = (
                str(i)
                + "_"
                + str(j)
                + "_"
                + str(hess_ctr)
                + "apical_sapled_target_xy_plot.png"
            )
            filename_shape_plot_basal = (
                str(i)
                + "_"
                + str(j)
                + "_"
                + str(hess_ctr)
                + "basal_sapled_target_xy_plot.png"
            )
            apical_plot_path= os.path.join(run_time_folder_contour, filename_shape_plot_apical)
            basal_plot_path= os.path.join(run_time_folder_contour, filename_shape_plot_basal)   
            save_contour_overlay_plot(exp_data_apical, parsed_result.contours["apical_normalized"], apical_plot_path, sampled_style="line")
            save_contour_overlay_plot(exp_data_basal, parsed_result.contours["basal_normalized"], basal_plot_path, sampled_style="line")

            """
			STEP 3D: Caculating frechet error
			"""
            sampled_data_apical = np.zeros((300, 2))
            sampled_data_basal = np.zeros((300, 2))
            sampled_data_apical[:, 0] = xt_sampled_apical
            sampled_data_apical[:, 1] = yt_sampled_apical
            sampled_data_basal[:, 0] = xt_sampled_basal
            sampled_data_basal[:, 1] = yt_sampled_basal
            # Computing the Frechet distance for the apical surface
            error_target_sampled_step_apical = score_frechet_distance(
                exp_data_apical,
                sampled_data_apical,
                objective_name="frechet_distance_apical",
            ).value
            error_target_sampled_step_basal = score_frechet_distance(
                exp_data_basal,
                sampled_data_basal,
                objective_name="frechet_distance_basal",
            ).value
            # Computing Frechet distance for the basal surface
            error_target_sampled_apical.append(error_target_sampled_step_apical)
            error_target_sampled_basal.append(error_target_sampled_step_basal)

            # Saving vertices
            archive_artifact(
                parsed_result.artifacts.get("vertices_path"),
                "vertices_"
                + str(i)
                + "_"
                + str(j)
                + "_"
                + str(hess_ctr)
                + ".txt",
            )
            BACKEND.cleanup_generated_files()
            # Removing variables that are not necessary for memory issues
            del xt_sampled_apical
            del yt_sampled_apical
            del xt_sampled_basal
            del yt_sampled_basal
            gc.collect()


"""Saving important arrays
"""
np.save(
    'output_data_files/error_target_sampled_apical.npy', error_target_sampled_apical
)
np.save('output_data_files/error_target_sampled_basal.npy', error_target_sampled_basal)
np.save('output_data_files/curvature_basal_master.npy', curvature_basal_master)
np.save('output_data_files/tissue_edge_length_master.npy', tissue_edge_length_master)
np.save('output_data_files/param_sampled.npy', param_sampled)
