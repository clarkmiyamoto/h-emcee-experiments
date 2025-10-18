#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0:10:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64GB
#SBATCH --job-name=HighDimensionalGaussian
#SBATCH --mail-type=END
#SBATCH --mail-user=cm6627@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

module purge

OVERLAY_FILE=/scratch/cm6627/hemcee_env/overlay-15GB-500K.ext3:ro
SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif

PYTHON_FILE=gaussian.py
export PYTHONUNBUFFERED=1

singularity exec --nv \
  --overlay $OVERLAY_FILE \
  $SINGULARITY_IMAGE \
  /bin/bash -c "source /ext3/env.sh; python $PYTHON_FILE --move=hmc_walk"
