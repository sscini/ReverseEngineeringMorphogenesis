import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


if "similaritymeasures" not in sys.modules:
    similaritymeasures = types.ModuleType("similaritymeasures")
    similaritymeasures.frechet_dist = lambda a, b: float(
        np.linalg.norm(np.asarray(a) - np.asarray(b))
    )
    sys.modules["similaritymeasures"] = similaritymeasures

if "spatial_efd" not in sys.modules:
    spatial_efd = types.ModuleType("spatial_efd")

    def calculate_efd(x, y, harmonic):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        coeffs = np.zeros((harmonic, 4))
        if len(x) == 0:
            return coeffs
        coeffs[:, 0] = np.linspace(x.min(), x.max(), harmonic)
        coeffs[:, 1] = np.linspace(y.min(), y.max(), harmonic)
        coeffs[:, 2] = coeffs[:, 0]
        coeffs[:, 3] = coeffs[:, 1]
        return coeffs

    def normalize_efd(coeffs, size_invariant=True):
        return coeffs, 0.0

    def inverse_transform(coeffs, harmonic):
        coeffs = np.asarray(coeffs)
        x = coeffs[:, 0]
        y = coeffs[:, 1]
        return x, y

    spatial_efd.CalculateEFD = calculate_efd
    spatial_efd.normalize_efd = normalize_efd
    spatial_efd.inverse_transform = inverse_transform
    sys.modules["spatial_efd"] = spatial_efd

if "gpytorch" not in sys.modules:
    gpytorch = types.ModuleType("gpytorch")

    class ExactGP:
        def __init__(self, *args, **kwargs):
            pass

    class ConstantMean:
        def __call__(self, x):
            return x

    class RBFKernel:
        def __init__(self, ard_num_dims=None):
            self.ard_num_dims = ard_num_dims

    class ScaleKernel:
        def __init__(self, base_kernel):
            self.base_kernel = base_kernel

    class MultivariateNormal:
        def __init__(self, mean, covariance):
            self.mean = mean
            self.covariance = covariance

    class GaussianLikelihood:
        def __call__(self, value):
            return value

    gpytorch.models = types.SimpleNamespace(ExactGP=ExactGP)
    gpytorch.means = types.SimpleNamespace(ConstantMean=ConstantMean)
    gpytorch.kernels = types.SimpleNamespace(
        ScaleKernel=ScaleKernel, RBFKernel=RBFKernel
    )
    gpytorch.distributions = types.SimpleNamespace(
        MultivariateNormal=MultivariateNormal
    )
    gpytorch.likelihoods = types.SimpleNamespace(GaussianLikelihood=GaussianLikelihood)
    sys.modules["gpytorch"] = gpytorch

if "torch" not in sys.modules:
    torch = types.ModuleType("torch")
    torch.from_numpy = lambda array: array
    sys.modules["torch"] = torch

if "smt" not in sys.modules:
    smt = types.ModuleType("smt")
    sampling_methods = types.ModuleType("smt.sampling_methods")

    class LHS:
        def __init__(self, xlimits):
            self.xlimits = np.asarray(xlimits)

        def __call__(self, num_samples):
            return np.tile(self.xlimits[:, 0], (num_samples, 1))

    sampling_methods.LHS = LHS
    smt.sampling_methods = sampling_methods
    sys.modules["smt"] = smt
    sys.modules["smt.sampling_methods"] = sampling_methods


def load_module(relative_path, module_name):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bo_module():
    return load_module(
        "bayesian-optimization-master/master_bayesian_optimization.py",
        "master_bayesian_optimization",
    )
