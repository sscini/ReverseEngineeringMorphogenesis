import numpy as np


class DummyBackend:
    def __init__(self):
        self.calls = []

    def load_target(self, target_source, target_type):
        return type("Target", (), {"contour": np.array([[0.0, 0.0], [1.0, 1.0]])})()

    def _coefficients_to_contour(self, coefficients):
        contour = np.zeros((coefficients.shape[0], 2))
        contour[:, 0] = coefficients[:, 0]
        contour[:, 1] = coefficients[:, 1]
        return contour

    def evaluate_training_data(self, target_contour, sim_data):
        return 0.25

    def sample_to_model_parameters(self, x_sampled, base_parameters, sampled_indices):
        params = list(base_parameters)
        for idx, sampled_index in enumerate(sampled_indices):
            params[sampled_index] = np.asarray(x_sampled).reshape(-1)[idx]
        return params

    def evaluate(self, model_parameters, target, param_pressure):
        self.calls.append((model_parameters, param_pressure))
        return type(
            "Evaluation",
            (),
            {
                "objective_value": 1.5,
                "contours": {
                    "primary": np.array([[0.0, 0.0], [1.0, 1.0]]),
                    "target": target.contour,
                },
                "features": {},
                "artifacts": {},
            },
        )()

    def save_artifacts(self, iteration, evaluation, output_config):
        return None


class FakeGPR:
    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    def GP_model_definition(self, model_cls, pc_index, model_initialize_check):
        return object(), object(), None, None, None, None

    def GP_training(
        self, train_x_t, train_y_t, model, likelihood, optimizer_select, training_iter
    ):
        return model, likelihood, None


class FakeAF:
    def __init__(self, x, train_x, train_y):
        self.x = x

    def expected_improvement(self, model, likelihood, exploration_param_val):
        return np.array([0.1, 0.9]), None, None


def test_bo_loop_uses_backend_contract(bo_module, monkeypatch, tmp_path):
    backend = DummyBackend()
    config = bo_module.BOConfig(
        n_iterations=1,
        num_samples=2,
        split_size=1,
        num_samples_af=2,
        num_iteration_gpr=1,
        output_data_dir=str(tmp_path / "output"),
    )

    monkeypatch.setattr(
        bo_module,
        "load_training_data",
        lambda config: (
            np.ones((2, 35)),
            np.ones((2, config.num_harmonics_efd * 4)),
        ),
    )
    monkeypatch.setattr(bo_module, "GaussianProcessRegression", FakeGPR)
    monkeypatch.setattr(bo_module, "AcqisitionFunctions", FakeAF)

    results = bo_module.run_bayesian_optimization(config=config, backend=backend)

    assert len(backend.calls) == 1
    assert results["train_x"].shape[0] == 2
    assert results["train_y"].shape[0] == 2
    assert results["error_target_sampled"] == [1.5]
