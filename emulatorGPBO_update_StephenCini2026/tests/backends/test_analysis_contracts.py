import numpy as np

from dependencies.scorers import score_frechet_distance
from dependencies.workflow_artifacts import archive_artifact, save_contour_overlay_plot
from dependencies.workflow_types import ParsedResult


def test_save_error_plot_and_workflow_artifacts(bo_module, tmp_path):
    config = bo_module.BOConfig(error_plot_dir=str(tmp_path / "error_plots"))
    bo_module.save_error_plot(config, np.array([[1.0], [2.0]]), [0.5], 0, 2)

    output_dir = tmp_path / "error_plots" / f"run_{config.timestamp}"
    assert (output_dir / "0error_evolution.png").exists()

    parsed_result = ParsedResult(
        contours={
            "primary": np.array([[0.0, 0.0], [1.0, 1.0]]),
        },
        features={},
        artifacts={},
    )
    save_contour_overlay_plot(
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        parsed_result.primary_contour,
        str(tmp_path / "contours" / "1_sampled_target_xy_plot.png"),
    )

    assert (tmp_path / "contours" / "1_sampled_target_xy_plot.png").exists()

    artifact_source = tmp_path / "vertices.txt"
    artifact_source.write_text("dummy\n")
    archive_artifact(str(artifact_source), str(tmp_path / "archive" / "vertices_1.txt"))
    assert (tmp_path / "archive" / "vertices_1.txt").exists()

    objective_result = score_frechet_distance(
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array([[0.0, 0.0], [1.0, 1.0]]),
    )
    assert objective_result.name == "frechet_distance"
