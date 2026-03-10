import numpy as np

from dependencies.backends.surface_evolver_backend import SurfaceEvolverBackend
from dependencies.workflow_types import ParsedResult, TargetData


def test_backend_execute_candidate_and_cleanup(tmp_path, monkeypatch):
    backend = SurfaceEvolverBackend()
    vertices_path = tmp_path / "vertices.txt"
    vertices_path.write_text("dummy\n")
    energylog_path = tmp_path / "energylog.txt"
    energylog_path.write_text("energy\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(backend, "write_geometry", lambda *args, **kwargs: None)
    monkeypatch.setattr(backend, "run_simulation", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        backend,
        "parse_output",
        lambda path: ParsedResult(
            contours={
                "primary": np.array([[0.0, 0.0], [1.0, 1.0]]),
                "basal_raw": np.array([[0.0, 0.0], [1.0, 1.0]]),
                "basal_normalized": np.array([[0.0, 0.0], [1.0, 1.0]]),
                "apical_raw": np.array([[0.0, 1.0], [1.0, 2.0]]),
                "apical_normalized": np.array([[0.0, 1.0], [1.0, 2.0]]),
            },
            features={"edge_length": np.array([1.0, 2.0]), "curvature": np.array([0.1, 0.2])},
            artifacts={"vertices_path": str(vertices_path)},
        ),
    )

    parsed_result = backend.execute_candidate([1, 2, 3], param_pressure=0.001)

    assert "primary" in parsed_result.contours
    assert "edge_length" in parsed_result.features
    assert parsed_result.artifacts["vertices_path"] == str(vertices_path)

    backend.cleanup_generated_files(run_dir=tmp_path)
    assert not vertices_path.exists()
    assert not energylog_path.exists()
