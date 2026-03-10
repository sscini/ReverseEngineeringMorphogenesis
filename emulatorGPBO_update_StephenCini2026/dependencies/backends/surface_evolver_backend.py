from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
import spatial_efd

from dependencies.feature_extractor_4 import FeatureExtractor
from dependencies.geometry_writer import GeometryWriter
from dependencies.workflow_types import ParsedResult, TargetData


class SurfaceEvolverBackend:
    def __init__(
        self,
        se_filename="wingDisc",
        edge_data_path="input_data/log_edges.xlsx",
        num_harmonics_efd=20,
        runner=None,
        evolver_command=None,
        cleanup_files=None,
    ):
        self.se_filename = se_filename
        self.edge_data_path = edge_data_path
        self.num_harmonics_efd = num_harmonics_efd
        self.runner = runner or self._default_runner
        self.evolver_command = evolver_command
        self.cleanup_files = cleanup_files or [
            "vertices.txt",
            "energylog.txt",
            "specificenergylog.txt",
        ]

    def _default_runner(self, command, cwd=None):
        try:
            subprocess.run(command, check=True, cwd=cwd)
        except FileNotFoundError as exc:
            executable = command if isinstance(command, str) else command[0]
            raise FileNotFoundError(
                "Surface Evolver executable not found. "
                f"Tried '{executable}'. Install Surface Evolver and ensure it is on PATH, "
                "or set EVOLVER_BIN to the full executable path."
            ) from exc

    def build_command(self, se_input_filename=None):
        if self.evolver_command is not None:
            if isinstance(self.evolver_command, str):
                return self.evolver_command
            return list(self.evolver_command)
        evolver_bin = os.environ.get("EVOLVER_BIN", "evolver")
        if os.path.sep in evolver_bin:
            if not os.path.exists(evolver_bin):
                raise FileNotFoundError(
                    f"EVOLVER_BIN points to missing path: '{evolver_bin}'"
                )
        elif shutil.which(evolver_bin) is None:
            raise FileNotFoundError(
                f"Surface Evolver executable '{evolver_bin}' not found on PATH. "
                "Set EVOLVER_BIN to the full executable path."
            )
        input_name = se_input_filename or f"{self.se_filename}.fe"
        return [evolver_bin, input_name]

    def run_simulation(self, cwd=None, se_input_filename=None):
        command = self.build_command(se_input_filename=se_input_filename)
        self.runner(command, cwd=cwd)

    def load_target(self, target_source, target_type):
        if target_type == "experimental":
            contour = self._load_xy_contour(target_source)
            coeff = spatial_efd.CalculateEFD(
                contour[:, 0], contour[:, 1], self.num_harmonics_efd
            )
            coeff, rotation = spatial_efd.normalize_efd(coeff, size_invariant=True)
            primary_contour = self.contour_from_coefficients(coeff)
            return TargetData(
                contours={"primary": primary_contour, "raw": contour},
                metadata={
                    "source": target_source,
                    "target_type": target_type,
                    "coefficients": coeff,
                    "rotation": rotation,
                },
            )

        if target_type == "surface_evolver":
            parsed_result = self.parse_output(target_source)
            return TargetData(
                contours={
                    "primary": parsed_result.contours["primary"],
                    "basal_raw": parsed_result.contours["basal_raw"],
                    "basal_normalized": parsed_result.contours["basal_normalized"],
                    "apical_raw": parsed_result.contours["apical_raw"],
                    "apical_normalized": parsed_result.contours["apical_normalized"],
                },
                metadata={
                    "source": target_source,
                    "target_type": target_type,
                    "parsed_result": parsed_result,
                },
            )

        raise ValueError(
            f"Unsupported target_type '{target_type}'. Supported target types: experimental, surface_evolver"
        )

    def sample_to_model_parameters(self, x_sampled, base_parameters, sampled_indices):
        sampled_values = np.asarray(x_sampled).reshape(-1)
        paraminputs = list(base_parameters)
        for idx, sampled_index in enumerate(sampled_indices):
            paraminputs[sampled_index] = sampled_values[idx]
        return paraminputs

    def write_geometry(self, model_parameters, param_pressure, output_file_name=None):
        GeometryWriter(
            model_parameters,
            param_pressure,
            output_file_name or self.se_filename,
        )

    def contour_from_coefficients(self, coefficients):
        xt, yt = spatial_efd.inverse_transform(
            coefficients, harmonic=self.num_harmonics_efd
        )
        contour = np.zeros((len(xt), 2))
        contour[:, 0] = xt
        contour[:, 1] = yt
        return contour

    def extract_features(self, vertices_path):
        extractor = FeatureExtractor(vertices_path, self.edge_data_path)
        efd = extractor.extract_tissue_efd_components(self.num_harmonics_efd)
        basal_raw = self.contour_from_coefficients(efd["basal_coefficients"])
        basal_normalized = self.contour_from_coefficients(
            efd["basal_normalized_coefficients"]
        )
        apical_raw = self.contour_from_coefficients(efd["apical_coefficients"])
        apical_normalized = self.contour_from_coefficients(
            efd["apical_normalized_coefficients"]
        )
        return {
            "efd": efd,
            "edge_length": extractor.edge_length(),
            "curvature": extractor.tissue_local_curvature(),
            "contours": {
                "basal_raw": basal_raw,
                "basal_normalized": basal_normalized,
                "apical_raw": apical_raw,
                "apical_normalized": apical_normalized,
            },
        }

    def parse_output(self, vertices_path):
        features = self.extract_features(vertices_path)
        return ParsedResult(
            contours={
                "primary": features["contours"]["basal_raw"],
                "basal_raw": features["contours"]["basal_raw"],
                "basal_normalized": features["contours"]["basal_normalized"],
                "apical_raw": features["contours"]["apical_raw"],
                "apical_normalized": features["contours"]["apical_normalized"],
            },
            features=features,
            artifacts={"vertices_path": vertices_path},
        )

    def execute_candidate(self, model_parameters, param_pressure, run_dir=None):
        self.write_geometry(model_parameters, param_pressure)
        self.run_simulation(cwd=run_dir)
        vertices_path = os.path.join(run_dir or ".", "vertices.txt")
        return self.parse_output(vertices_path)

    def cleanup_generated_files(self, run_dir=None):
        for filename in self.cleanup_files:
            file_path = os.path.join(run_dir or ".", filename)
            if os.path.exists(file_path):
                os.remove(file_path)

    def _load_xy_contour(self, geometry_data):
        if os.stat(geometry_data).st_size == 0:
            raise ValueError(f"Target geometry file is empty: {geometry_data}")
        x_coords = []
        y_coords = []
        with open(geometry_data) as handle:
            for line in handle:
                data = line.split()
                x_coords.append(float(data[0]))
                y_coords.append(float(data[1]))
        contour = np.zeros((len(x_coords), 2))
        contour[:, 0] = x_coords
        contour[:, 1] = y_coords
        return contour
