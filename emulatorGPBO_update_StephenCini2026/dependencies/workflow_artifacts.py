from __future__ import annotations

import os
import shutil

import matplotlib.pyplot as plt


def archive_artifact(source_path, destination_path):
    if not source_path or not os.path.exists(source_path):
        return None
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.copyfile(source_path, destination_path)
    return destination_path


def save_contour_overlay_plot(
    target_contour,
    sampled_contour,
    output_path,
    sampled_style="scatter",
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if sampled_style == "line":
        plt.plot(target_contour[:, 0], target_contour[:, 1], "black", label="Target")
        plt.plot(sampled_contour[:, 0], sampled_contour[:, 1], "blue", label="Sampled")
        plt.legend()
    else:
        plt.scatter(target_contour[:, 0], target_contour[:, 1], color="black")
        plt.scatter(sampled_contour[:, 0], sampled_contour[:, 1], color="blue")
    plt.xlabel("x [nondimensional]")
    plt.ylabel("y [nondimensional]")
    plt.savefig(output_path)
    plt.close()
    return output_path
