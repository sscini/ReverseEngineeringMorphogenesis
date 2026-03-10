import numpy as np

from dependencies.backends.surface_evolver_backend import SurfaceEvolverBackend, TargetData


def test_backend_evaluate_and_save_artifacts(tmp_path, monkeypatch):
    backend = SurfaceEvolverBackend()
    target = TargetData(contour=np.array([[0.0, 0.0], [1.0, 1.0]]))
    vertices_path = tmp_path / "vertices.txt"
    vertices_path.write_text("dummy\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(backend, "write_geometry", lambda *args, **kwargs: None)
    monkeypatch.setattr(backend, "run_simulation", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        backend,
        "extract_features",
        lambda path: {
            "efd": {},
            "edge_length": np.array([1.0, 2.0]),
            "curvature": np.array([0.1, 0.2]),
            "contours": {
                "basal_raw": np.array([[0.0, 0.0], [1.0, 1.0]]),
                "basal_normalized": np.array([[0.0, 0.0], [1.0, 1.0]]),
                "apical_raw": np.array([[0.0, 1.0], [1.0, 2.0]]),
                "apical_normalized": np.array([[0.0, 1.0], [1.0, 2.0]]),
            },
        },
    )
    monkeypatch.setattr(
        "dependencies.backends.surface_evolver_backend.similaritymeasures.frechet_dist",
        lambda target_contour, sampled_contour: 3.5,
    )

    evaluation = backend.evaluate([1, 2, 3], target, param_pressure=0.001)

    assert evaluation.objective_value == 3.5
    assert "primary" in evaluation.contours
    assert "edge_length" in evaluation.features

    backend.save_artifacts(
        0,
        evaluation,
        {
            "vertices_dir": str(tmp_path / "archive"),
            "contour_dir": str(tmp_path / "plots"),
        },
    )

    assert (tmp_path / "archive" / "vertices_0.txt").exists()
    assert (tmp_path / "plots" / "0_sampled_target_xy_plot.png").exists()
    assert not vertices_path.exists()
