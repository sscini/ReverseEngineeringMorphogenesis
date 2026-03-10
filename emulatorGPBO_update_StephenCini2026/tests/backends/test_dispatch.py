import pytest

from dependencies.backends import get_backend
from dependencies.backends.surface_evolver_backend import SurfaceEvolverBackend


def test_get_surface_evolver_backend():
    backend = get_backend("SurfaceEvolver")
    assert isinstance(backend, SurfaceEvolverBackend)


def test_unsupported_backend():
    with pytest.raises(ValueError, match="Unsupported method"):
        get_backend("UnsupportedBackend")
