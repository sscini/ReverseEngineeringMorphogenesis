import numpy as np

from dependencies.scorers import score_frechet_distance, score_parsed_result_frechet
from dependencies.workflow_types import ParsedResult, TargetData


def test_target_and_parsed_result_primary_contour_properties():
    target = TargetData(contours={"primary": np.array([[0.0, 0.0], [1.0, 1.0]])})
    parsed_result = ParsedResult(
        contours={"primary": np.array([[1.0, 1.0], [2.0, 2.0]])},
        features={},
    )

    assert target.primary_contour.shape == (2, 2)
    assert parsed_result.primary_contour.shape == (2, 2)


def test_frechet_scoring_seam_uses_typed_inputs():
    target = TargetData(contours={"primary": np.array([[0.0, 0.0], [1.0, 1.0]])})
    parsed_result = ParsedResult(
        contours={"primary": np.array([[0.0, 0.0], [1.0, 1.0]])},
        features={},
    )

    direct = score_frechet_distance(target.primary_contour, parsed_result.primary_contour)
    typed = score_parsed_result_frechet(target, parsed_result)

    assert direct.name == "frechet_distance"
    assert typed.name == "frechet_distance"
    assert direct.value == typed.value
