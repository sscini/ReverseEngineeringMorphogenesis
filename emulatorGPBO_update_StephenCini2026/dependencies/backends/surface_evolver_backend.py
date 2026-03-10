from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import similaritymeasures
import spatial_efd

from dependencies.feature_extractor_4 import FeatureExtractor
from dependencies.geometry_writer import GeometryWriter


@dataclass
class TargetData:
    contour: np.ndarray
    metadata: dict = field(default_factory=dict)


@dataclass
class BackendEvaluation:
    objective_value: float
    contours: dict
    features: dict
    artifacts: dict = field(default_factory=dict)


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
        subprocess.run(command, check=True, cwd=cwd)

    def build_command(self, se_input_filename=None):
        if self.evolver_command is not None:
            if isinstance(self.evolver_command, str):
                return self.evolver_command
            return list(self.evolver_command)
        evolver_bin = os.environ.get("EVOLVER_BIN", "evolver")
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
            primary_contour = self._coefficients_to_contour(coeff)
            return TargetData(
                contour=primary_contour,
                metadata={
                    "source": target_source,
                    "target_type": target_type,
                    "coefficients": coeff,
                    "rotation": rotation,
                },
            )

        if target_type == "surface_evolver":
            features = self.extract_features(target_source)
            return TargetData(
                contour=features["contours"]["basal_raw"],
                metadata={
                    "source": target_source,
                    "target_type": target_type,
                    "features": features,
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

    def extract_features(self, vertices_path):
        extractor = FeatureExtractor(vertices_path, self.edge_data_path)
        efd = extractor.extract_tissue_efd_components(self.num_harmonics_efd)
        basal_raw = self._coefficients_to_contour(efd["basal_coefficients"])
        basal_normalized = self._coefficients_to_contour(efd["basal_normalized_coefficients"])
        apical_raw = self._coefficients_to_contour(efd["apical_coefficients"])
        apical_normalized = self._coefficients_to_contour(
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

    def evaluate(self, model_parameters, target, param_pressure, run_dir=None):
        self.write_geometry(model_parameters, param_pressure)
        self.run_simulation(cwd=run_dir)

        vertices_path = os.path.join(run_dir or ".", "vertices.txt")
        features = self.extract_features(vertices_path)
        primary_contour = features["contours"]["basal_raw"]
        objective_value = similaritymeasures.frechet_dist(target.contour, primary_contour)

        return BackendEvaluation(
            objective_value=float(objective_value),
            contours={
                "primary": primary_contour,
                "basal_raw": features["contours"]["basal_raw"],
                "basal_normalized": features["contours"]["basal_normalized"],
                "apical_raw": features["contours"]["apical_raw"],
                "apical_normalized": features["contours"]["apical_normalized"],
                "target": target.contour,
            },
            features=features,
            artifacts={"vertices_path": vertices_path},
        )

    def evaluate_training_data(self, target_contour, simulated_contour):
        return float(similaritymeasures.frechet_dist(target_contour, simulated_contour))

    def save_artifacts(self, iteration, evaluation, output_config=None):
        output_config = output_config or {}
        vertices_dir = output_config.get("vertices_dir")
        contour_dir = output_config.get("contour_dir")
        contour_prefix = output_config.get("contour_prefix", "")

        vertices_path = evaluation.artifacts.get("vertices_path")
        if vertices_dir and vertices_path and os.path.exists(vertices_path):
            os.makedirs(vertices_dir, exist_ok=True)
            archived_vertices = os.path.join(vertices_dir, f"vertices_{iteration}.txt")
            shutil.copyfile(vertices_path, archived_vertices)
            evaluation.artifacts["archived_vertices_path"] = archived_vertices

        if contour_dir:
            os.makedirs(contour_dir, exist_ok=True)
            contour_path = os.path.join(
                contour_dir, f"{contour_prefix}{iteration}_sampled_target_xy_plot.png"
            )
            target = evaluation.contours["target"]
            sampled = evaluation.contours["primary"]
            plt.scatter(target[:, 0], target[:, 1], color="black")
            plt.scatter(sampled[:, 0], sampled[:, 1], color="blue")
            plt.xlabel("x [nondimensional]")
            plt.ylabel("y [nondimensional]")
            plt.savefig(contour_path)
            plt.close()
            evaluation.artifacts["contour_plot_path"] = contour_path

        self.cleanup_generated_files()

    def cleanup_generated_files(self):
        for filename in self.cleanup_files:
            if os.path.exists(filename):
                os.remove(filename)

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

    def _coefficients_to_contour(self, coefficients):
        xt, yt = spatial_efd.inverse_transform(
            coefficients, harmonic=self.num_harmonics_efd
        )
        contour = np.zeros((len(xt), 2))
        contour[:, 0] = xt
        contour[:, 1] = yt
        return contour
