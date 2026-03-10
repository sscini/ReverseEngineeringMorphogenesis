import numpy as np

from dependencies.backends.surface_evolver_backend import BackendEvaluation, SurfaceEvolverBackend


def test_save_error_plot_and_backend_artifacts(bo_module, tmp_path):
    config = bo_module.BOConfig(error_plot_dir=str(tmp_path / "error_plots"))
    bo_module.save_error_plot(config, np.array([[1.0], [2.0]]), [0.5], 0, 2)

    output_dir = tmp_path / "error_plots" / f"run_{config.timestamp}"
    assert (output_dir / "0error_evolution.png").exists()

    backend = SurfaceEvolverBackend()
    evaluation = BackendEvaluation(
        objective_value=0.1,
        contours={
            "target": np.array([[0.0, 0.0], [1.0, 1.0]]),
            "primary": np.array([[0.0, 0.0], [1.0, 1.0]]),
        },
        features={},
        artifacts={},
    )
    backend.save_artifacts(
        1, evaluation, {"contour_dir": str(tmp_path / "contours"), "vertices_dir": None}
    )

    assert (tmp_path / "contours" / "1_sampled_target_xy_plot.png").exists()
