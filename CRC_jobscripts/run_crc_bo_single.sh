#!/bin/bash

#$ -M scini@nd.edu   # Email address for job notification
#$ -m abe            # Send mail when job begins, ends and aborts
#$ -pe smp 4         # Specify parallel environment and legal core size
#$ -q long           # Run on the CPU cluster
#$ -N evolver_job_run10
#$ -o /users/scini/github_data/<YOUR_PROJECT>/logs/run10.out
#$ -e /users/scini/github_data/<YOUR_PROJECT>/logs/run10.err

# Always start in the directory where qsub was launched
cd "$SGE_O_WORKDIR" || exit 1

# ---- Surface Evolver environment (CRC) ----
export EVOLVERHOME="$HOME/software/evolver-2.70"
export EVOLVERPATH="$EVOLVERHOME/fe:$EVOLVERHOME/doc"

# Put evolver on PATH (works whether you use ~/bin symlink or not)
export PATH="$HOME/bin:$EVOLVERHOME/src:$PATH"

# (Optional) sanity prints
echo "Host: $(hostname)"
echo "Workdir: $(pwd)"
echo "Evolver: $(which evolver)"
echo "EVOLVERPATH: $EVOLVERPATH"

# ---- Python environment ----
module load conda
conda activate pyomo-crc

# ---- Run your python script ----
python sens_analysis_crc_script.py
