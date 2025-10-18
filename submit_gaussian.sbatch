#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=HighDimensionalGaussian
#SBATCH --mail-type=END
#SBATCH --mail-user=cm6627@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

module purge

OVERLAY_FILE=/scratch/cm6627/hemcee_env/overlay-15GB-500K.ext3:ro
SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif

PYTHON_FILE=high_dimensional_gaussian.py

singularity exec --nv \
  --overlay $OVERLAY_FILE \
  $SINGULARITY_IMAGE \
  /bin/bash -c "source /ext3/env.sh; python $PYTHON_FILE"