#!/bin/bash

# Script to generate and submit sbatch jobs for all possible move types
# Based on the move types defined in configuration.py

# Define all possible move types from configuration.py
MOVE_TYPES=("hmc" "hmc_walk" "hmc_side" "stretch" "walk" "side")

# Define the Python files to run (you can modify this list as needed)
PYTHON_FILES=("gaussian.py" "allen_cahn.py" "ring.py")

# Base sbatch template (will be modified for each job)
SBATCH_TEMPLATE="submit_template.sbatch"

# Create the base sbatch template
cat > $SBATCH_TEMPLATE << 'EOF'
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --job-name=JOB_NAME_PLACEHOLDER
#SBATCH --mail-type=END
#SBATCH --mail-user=cm6627@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

module purge

OVERLAY_FILE=/scratch/cm6627/hemcee_env/overlay-15GB-500K.ext3:ro
SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif

PYTHON_FILE=PYTHON_FILE_PLACEHOLDER
MOVE_TYPE=MOVE_TYPE_PLACEHOLDER

singularity exec --nv \
  --overlay $OVERLAY_FILE \
  $SINGULARITY_IMAGE \
  /bin/bash -c "source /ext3/env.sh; python $PYTHON_FILE --move $MOVE_TYPE"
EOF

echo "Generated sbatch template: $SBATCH_TEMPLATE"

# Function to create and submit sbatch for a specific move type and python file
create_and_submit_sbatch() {
    local move_type=$1
    local python_file=$2
    local job_name="${python_file%.*}_${move_type}"
    local sbatch_file="submit_${job_name}.sbatch"
    
    echo "Creating sbatch file for $move_type with $python_file..."
    
    # Create the sbatch file by replacing placeholders
    sed "s/JOB_NAME_PLACEHOLDER/$job_name/g; s/PYTHON_FILE_PLACEHOLDER/$python_file/g; s/MOVE_TYPE_PLACEHOLDER/$move_type/g" $SBATCH_TEMPLATE > $sbatch_file
    
    # Make the sbatch file executable
    chmod +x $sbatch_file
    
    echo "Submitting job: $job_name"
    sbatch $sbatch_file
    
    echo "Submitted: $sbatch_file"
    echo "---"
}

# Main loop: iterate over all combinations of move types and python files
echo "Starting job submission for all move types and python files..."
echo "Move types: ${MOVE_TYPES[*]}"
echo "Python files: ${PYTHON_FILES[*]}"
echo "=========================================="

for move_type in "${MOVE_TYPES[@]}"; do
    for python_file in "${PYTHON_FILES[@]}"; do
        # Check if the python file exists
        if [ -f "$python_file" ]; then
            create_and_submit_sbatch "$move_type" "$python_file"
        else
            echo "Warning: $python_file not found, skipping..."
        fi
    done
done

# Clean up the template file
rm $SBATCH_TEMPLATE

echo "=========================================="
echo "All jobs submitted!"
echo "Check job status with: squeue -u $USER"
echo "Check job details with: scontrol show job <job_id>"
