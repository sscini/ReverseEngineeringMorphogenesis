import numpy as np

from dependencies.workflow_types import TargetData


class DummyBackend:
    def load_target(self, target_source, target_type):
        return TargetData(contours={"primary": np.ones((20, 2))})

    def contour_from_coefficients(self, coefficients):
        contour = np.zeros((coefficients.shape[0], 2))
        contour[:, 0] = coefficients[:, 0]
        contour[:, 1] = coefficients[:, 1]
        return contour


def test_prepare_bo_data_uses_public_backend_contract(bo_module):
    config = bo_module.BOConfig(num_samples=2)
    backend = DummyBackend()
    master_parameter_input_n = np.ones((2, 35))
    master_feature_output = np.ones((2, config.num_harmonics_efd * 4))

    prepared = bo_module.prepare_bo_data(
        config, master_parameter_input_n, master_feature_output, backend
    )

    assert prepared["data_x"].shape == (2, config.num_parameters_lhs)
    assert prepared["data_y"].shape == (2, 1)
    assert prepared["target"].primary_contour.shape == (20, 2)
