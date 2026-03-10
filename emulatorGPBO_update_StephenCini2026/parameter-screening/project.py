# -project.py
"""
The code executes the job or surface evolver input files in this case generates by init.py

Created on Thu Feb 18 22:10:00 2021

@author: Nilay Kumar
email: nkumar4@nd.edu
Multicellular Systems Engineering Lab (MSELab)
Department of Chemical and Biomolecular Engineering
Institution: University of Notre Dame
"""

# importing the flow library
import os

import flow
from flow import FlowProject, directives

from dependencies.backends import get_backend


BACKEND = get_backend("SurfaceEvolver")


# Checking if an operation has been executed by checking f the initialization file exists
@FlowProject.label
def SE_file_exist_check(job):
    return os.path.isfile(job.fn("wingDisc.fe"))


@FlowProject.label
def output_file_exist_check(job):
    return os.path.isfile(job.fn("vertices.txt"))


# a) write_SE_initialization_file has been defined as a function under the Flowproject class
# 1) The function writes geometry files for initializing SE simulations
# 2) operation decorated identifies the function as an operation while running the job
# b) A post condition is defined to check if the job has been executed
@FlowProject.operation
@FlowProject.post(SE_file_exist_check)
def write_SE_initialization_file(job):
    BACKEND.write_geometry(
        job.sp.parameter_model,
        job.sp.parameter_pressure,
        job.fn(job.sp.output_file_name).replace(".fe", ""),
    )


# Defining a project operation for simulating the output geometry files
@FlowProject.operation
@flow.with_job
@flow.cmd  # It ensures that the functions returns a hell command through this decorator
def simulate_SE_file(job):
    # Fetching evolver installation from the src location. Running the file
    command = BACKEND.build_command(se_input_filename="wingDisc.fe")
    if isinstance(command, str):
        return command
    return " ".join(command)


@FlowProject.operation
@FlowProject.post(output_file_exist_check)
def write_geometrical_features(job):
    features = BACKEND.extract_features(job.fn("vertices.txt"))
    job.document["length"] = features["edge_length"]
    job.document["e_f_d"] = features["efd"]["basal_coefficients"]
    job.document["e_f_d_norm"] = features["efd"]["basal_normalized_coefficients"]
    job.document["e_f_d_rot"] = features["efd"]["basal_rotation"]
    job.document["curvature"] = features["curvature"]


if __name__ == '__main__':
    FlowProject().main()
