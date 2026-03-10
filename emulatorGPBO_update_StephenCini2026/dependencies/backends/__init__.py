from dependencies.backends.surface_evolver_backend import SurfaceEvolverBackend


def get_backend(method="SurfaceEvolver", **kwargs):
    if method == "SurfaceEvolver":
        return SurfaceEvolverBackend(**kwargs)
    raise ValueError(f"Unsupported method '{method}'. Supported methods: SurfaceEvolver")
