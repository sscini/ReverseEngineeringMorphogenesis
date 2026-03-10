from __future__ import annotations

import similaritymeasures

from dependencies.workflow_types import ObjectiveResult


def score_frechet_distance(
    target_contour,
    candidate_contour,
    objective_name="frechet_distance",
    metadata=None,
):
    return ObjectiveResult(
        name=objective_name,
        value=float(similaritymeasures.frechet_dist(target_contour, candidate_contour)),
        metadata=metadata or {},
    )


def score_parsed_result_frechet(
    target,
    parsed_result,
    target_contour_key="primary",
    parsed_contour_key="primary",
    objective_name="frechet_distance",
):
    return score_frechet_distance(
        target.contours[target_contour_key],
        parsed_result.contours[parsed_contour_key],
        objective_name=objective_name,
        metadata={
            "target_contour_key": target_contour_key,
            "parsed_contour_key": parsed_contour_key,
        },
    )
