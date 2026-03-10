from dependencies.backends import get_backend


def test_sample_to_model_parameters_maps_selected_indices():
    backend = get_backend("SurfaceEvolver")
    base_parameters = list(range(35))
    sampled_indices = [17, 18, 19, 28, 29, 30, 33]
    sampled_values = [101, 102, 103, 104, 105, 106, 107]

    translated = backend.sample_to_model_parameters(
        sampled_values, base_parameters, sampled_indices
    )

    for idx, expected in zip(sampled_indices, sampled_values):
        assert translated[idx] == expected
    assert translated[0] == 0
    assert translated[34] == 34
