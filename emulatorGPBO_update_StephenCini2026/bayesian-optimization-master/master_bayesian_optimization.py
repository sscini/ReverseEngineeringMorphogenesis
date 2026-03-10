# -*- coding: utf-8 -*-
"""
Main code for carrying out Bayesian optimization (BO) to estimate Surface Evolver
model parameters from wing imaginal disc tissue cross section data.
"""

import gc
import os
import time
from dataclasses import dataclass, field

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from smt.sampling_methods import LHS

from dependencies.acquisition_functions_class import AcqisitionFunctions
from dependencies.backends import get_backend
from dependencies.data_preprocessing_class import DataPreprocessing
from dependencies.gaussian_process_regression_class import GaussianProcessRegression


@dataclass
class BOConfig:
    method: str = "SurfaceEvolver"
    num_parameters_lhs: int = 7
    lhs_parameter_index: list = field(
        default_factory=lambda: [17, 18, 19, 28, 29, 30, 33]
    )
    geometry_data: str = "input_data/case1_pzornai_ctrl.txt"
    geometry_data_type: int = 1
    num_samples_af: int = 100000
    n_iterations: int = 100
    split_size: int = 149
    num_samples: int = 150
    paraminputs_stable: list = field(
        default_factory=lambda: [
            0,
            0.0001,
            0,
            0,
            0,
            0,
            0,
            0.001,
            0,
            0,
            0,
            0.1,
            0.1,
            10,
            0.1,
            0.1,
            0.1,
            0.1,
            10,
            0.0001,
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
            0.001,
            0.001,
        ]
    )
    param_pressure: float = 0.001
    num_harmonics_efd: int = 20
    exploration_param_val: float = 0.05
    num_iteration_gpr: int = 5000
    optimizer_type: int = 1
    se_filename: str = "wingDisc"
    input_parameter_path: str = "input_data/master_parameter_input_n.npy"
    input_feature_path: str = "input_data/master_feature_output.npy"
    output_data_dir: str = "output_data"
    contour_plot_dir: str | None = None
    error_plot_dir: str | None = None
    timestamp: str = field(default_factory=lambda: time.strftime("%Y%m%d-%H%M"))


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, ard_num_dims):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def _build_exact_gp_model(num_parameters_lhs):
    class ConfiguredExactGPModel(ExactGPModel):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood, num_parameters_lhs)

    return ConfiguredExactGPModel


def load_training_data(config):
    if not os.path.isfile(config.input_feature_path):
        raise FileNotFoundError(
            f"Expected training feature data at '{config.input_feature_path}'."
        )
    master_parameter_input_n = np.load(config.input_parameter_path)
    master_feature_output = np.load(config.input_feature_path)
    return master_parameter_input_n, master_feature_output


def prepare_bo_data(config, master_parameter_input_n, master_feature_output, backend):
    data_preprocess = DataPreprocessing(
        master_parameter_input_n, master_feature_output, config.num_samples
    )
    master_parameter_input_log = data_preprocess.input_log_transform()
    data_x = data_preprocess.input_parameter_selection(
        config.num_parameters_lhs,
        config.lhs_parameter_index,
        master_parameter_input_log,
    )
    data_x_mean = np.mean(data_x, axis=0)
    data_x_variance = np.std(data_x, axis=0)
    data_x = StandardScaler().fit_transform(data_x)
    max_data_x = np.amax(data_x, axis=0)
    min_data_x = np.amin(data_x, axis=0)

    target_type = "experimental" if config.geometry_data_type == 1 else "surface_evolver"
    target = backend.load_target(config.geometry_data, target_type)

    error_simulation_target_data = np.zeros(config.num_samples)
    for i in range(config.num_samples):
        temp = master_feature_output[i, :]
        temp2 = np.reshape(temp, (config.num_harmonics_efd, 4))
        sim_data = backend._coefficients_to_contour(temp2)
        error_simulation_target_data[i] = backend.evaluate_training_data(
            target.contour, sim_data
        )

    data_y = np.reshape(error_simulation_target_data, (config.num_samples, 1)) * (-1)
    return {
        "data_x": data_x,
        "data_y": data_y,
        "data_x_mean": data_x_mean,
        "data_x_variance": data_x_variance,
        "max_data_x": max_data_x,
        "min_data_x": min_data_x,
        "target": target,
    }


def run_bayesian_optimization(config=None, backend=None):
    config = config or BOConfig()
    backend = backend or get_backend(
        config.method,
        se_filename=config.se_filename,
        num_harmonics_efd=config.num_harmonics_efd,
    )

    master_parameter_input_n, master_feature_output = load_training_data(config)
    prepared = prepare_bo_data(
        config, master_parameter_input_n, master_feature_output, backend
    )

    train_x = prepared["data_x"][: config.split_size, :]
    train_y = prepared["data_y"][: config.split_size, :]
    test_x = prepared["data_x"][config.split_size : config.num_samples, :]
    test_y = prepared["data_y"][config.split_size : config.num_samples, :]

    xlimits = np.array(
        [
            [prepared["min_data_x"][i], prepared["max_data_x"][i]]
            for i in range(config.num_parameters_lhs)
        ]
    )
    sampling = LHS(xlimits=xlimits)
    x = sampling(config.num_samples_af)

    error_target_sampled = []
    iter_counter = []
    param_sampled = np.zeros((config.n_iterations, config.num_parameters_lhs))

    for i in range(config.n_iterations):
        gpr = GaussianProcessRegression(train_x, train_y, test_x, test_y)
        model_cls = _build_exact_gp_model(config.num_parameters_lhs)
        model, likelihood, train_x_t, train_y_t, _, _ = gpr.GP_model_definition(
            model_cls, 0, 1
        )
        model, likelihood, _ = gpr.GP_training(
            train_x_t,
            train_y_t,
            model,
            likelihood,
            config.optimizer_type,
            config.num_iteration_gpr,
        )

        af = AcqisitionFunctions(x, train_x, train_y)
        ei, _, _ = af.expected_improvement(
            model, likelihood, config.exploration_param_val
        )

        x_sampled_index = np.argmax(ei)
        x_sampled_logscale_standardized = x[x_sampled_index, :]
        x_sampled = np.exp(
            np.add(
                np.multiply(
                    x_sampled_logscale_standardized, prepared["data_x_variance"]
                ),
                prepared["data_x_mean"],
            )
        )

        model_parameters = backend.sample_to_model_parameters(
            x_sampled, config.paraminputs_stable, config.lhs_parameter_index
        )
        evaluation = backend.evaluate(
            model_parameters,
            prepared["target"],
            config.param_pressure,
        )
        backend.save_artifacts(
            i,
            evaluation,
            {
                "vertices_dir": config.output_data_dir,
                "contour_dir": _runtime_output_dir(
                    config.contour_plot_dir, f"brun_{config.timestamp}"
                ),
            },
        )

        y_sampled = np.reshape(evaluation.objective_value, (1, 1)) * (-1)
        train_x = np.vstack(
            (train_x, np.reshape(x_sampled_logscale_standardized, (1, config.num_parameters_lhs)))
        )
        train_y = np.vstack((train_y, y_sampled))

        error_target_sampled.append(evaluation.objective_value)
        iter_counter.append(i + 1)
        param_sampled[i, :] = np.asarray(x_sampled).reshape(-1)

        save_error_plot(
            config,
            prepared["data_y"],
            error_target_sampled,
            i,
            config.num_samples,
        )

        del gpr
        del model
        del likelihood
        gc.collect()

    os.makedirs(config.output_data_dir, exist_ok=True)
    np.save(os.path.join(config.output_data_dir, "error_target_sampled.npy"), error_target_sampled)
    np.save(os.path.join(config.output_data_dir, "param_sampled.npy"), param_sampled)
    np.save(os.path.join(config.output_data_dir, "train_x.npy"), train_x)
    np.save(os.path.join(config.output_data_dir, "train_y.npy"), train_y)

    return {
        "error_target_sampled": error_target_sampled,
        "iter_counter": iter_counter,
        "param_sampled": param_sampled,
        "train_x": train_x,
        "train_y": train_y,
    }


def _runtime_output_dir(base_dir, run_name):
    if not base_dir:
        return None
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_error_plot(config, data_y, error_target_sampled, iteration, num_samples):
    if not config.error_plot_dir:
        return
    run_time_folder_error = _runtime_output_dir(
        config.error_plot_dir, f"run_{config.timestamp}"
    )
    filename_error_iteration = f"{iteration}error_evolution.png"
    error_train = data_y * -1
    error_train_reshaped = np.reshape(error_train, (num_samples,))
    error_train_sorted = np.flip(np.sort(error_train_reshaped))
    index_trained = np.linspace(1, num_samples, num_samples)
    index_sampled = np.linspace(
        num_samples + 1, num_samples + iteration + 1, iteration + 1
    )
    plt.scatter(index_trained, error_train_sorted, color="black")
    plt.scatter(index_sampled, error_target_sampled, color="red")
    plt.ylabel("Index")
    plt.xlabel("Error between target and backend shape")
    plt.savefig(os.path.join(run_time_folder_error, filename_error_iteration))
    plt.close()


def main():
    run_bayesian_optimization()


if __name__ == "__main__":
    main()
