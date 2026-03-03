#!/bin/bash

#$ -M scini@nd.edu   # Email address for job notification
#$ -m abe            # Send mail when job begins, ends and aborts
#$ -q long           # Specify queue
#$ -pe smp 3         # Specify parallel environment
#$ -N sens_analysis_01252026_group5_nothreshold_real1    # Specify job name
#$ -t 1-10           # Specify job array range

# Start in submit directory
cd "$SGE_O_WORKDIR" || exit 1

# ---- Surface Evolver environment (CRC) ----
export EVOLVERHOME="$HOME/software/evolver-2.70"
export EVOLVERPATH="$EVOLVERHOME/fe:$EVOLVERHOME/doc"
export PATH="$HOME/bin:$EVOLVERHOME/src:$PATH"

# ---- Python environment ----
module load conda
conda activate pyomo-crc

# ---- Task parameters ----
runs=(1 2 3 4 5 6 7 8 9 10)
D_p_starts=(0 5 10 15 20 25 30 35 40 45)
D_p_ends=(5 10 15 20 25 30 35 40 45 50)

task_id=$SGE_TASK_ID
D_p_start_idx=${D_p_starts[$((task_id-1))]}
D_p_end_idx=${D_p_ends[$((task_id-1))]}
run=${runs[$((task_id-1))]}

job_name="sens_analysis_01252026_group5_nothreshold_real1_run${task_id}"

# ---- Logs ----
log_dir="/users/scini/github_data/sensitivity_01212026/logs"
mkdir -p "$log_dir"

output_file="$log_dir/$job_name.out"
error_file="$log_dir/$job_name.err"

# ---- Run ----
{
  echo "=== Job info ==="
  echo "Date: $(date)"
  echo "Host: $(hostname)"
  echo "Workdir: $(pwd)"
  echo "Task ID: $task_id"
  echo "Python: $(which python)"
  echo "Evolver: $(which evolver)"
  echo "EVOLVERPATH: $EVOLVERPATH"
  echo ""
  echo "Running task $task_id with D_p_start_idx $D_p_start_idx, D_p_end_idx $D_p_end_idx, run number $run"
  echo "Job name: $job_name"
  echo "============="
  echo ""

  python sens_analysis_crc_script.py "$job_name" "$D_p_start_idx" "$D_p_end_idx" "$run"
} > "$output_file" 2> "$error_file"
