from pathlib import Path

import pandas as pd

from dependencies.feature_extractor_4 import FeatureExtractor


def test_feature_extractor_returns_named_efd_components():
    fixture_path = (
        Path(__file__).resolve().parent / "fixtures" / "vertices_minimal.txt"
    )
    edge_data = pd.DataFrame(
        {
            "eid": list(range(1, 13)),
            "v1": [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5],
            "v2": [2, 3, 4, 1, 6, 7, 8, 5, 6, 7, 8, 1],
        }
    )
    extractor = FeatureExtractor(
        str(fixture_path), edge_data, n_squamous=1, n_columnar=2, n_cuboidal=1
    )

    components = extractor.extract_tissue_efd_components(2)

    assert set(components) == {
        "basal_coefficients",
        "basal_normalized_coefficients",
        "basal_rotation",
        "apical_coefficients",
        "apical_normalized_coefficients",
        "apical_rotation",
    }
    assert components["basal_coefficients"].shape == (2, 4)
    assert components["apical_coefficients"].shape == (2, 4)
