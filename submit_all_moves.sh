#!/bin/bash

# Script to generate and submit sbatch jobs for all possible move types
# Based on the move types defined in configuration.py

# Define all possible move types from configuration.py
MOVE_TYPES=("hmc" "hmc_walk" "hmc_side" "stretch" "walk" "side")

# Define HMC-based move types that need parameter variations
HMC_MOVE_TYPES=("hmc" "hmc_walk" "hmc_side")

# Define parameter combinations for HMC moves: (step_size, integration_length)
HMC_PARAMS=("0.5,2" "0.1,10")

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
STEP_SIZE=STEP_SIZE_PLACEHOLDER
INTEGRATION_LENGTH=INTEGRATION_LENGTH_PLACEHOLDER

singularity exec --nv \
  --overlay $OVERLAY_FILE \
  $SINGULARITY_IMAGE \
  /bin/bash -c "source /ext3/env.sh; python $PYTHON_FILE --move $MOVE_TYPE --hamiltonian_step_size $STEP_SIZE --hamiltonian_L $INTEGRATION_LENGTH"
EOF

echo "Generated sbatch template: $SBATCH_TEMPLATE"

# Function to create and submit sbatch for a specific move type and python file
create_and_submit_sbatch() {
    local move_type=$1
    local python_file=$2
    local step_size=$3
    local integration_length=$4
    local job_name="${python_file%.*}_${move_type}_s${step_size}_L${integration_length}"
    local sbatch_file="submit_${job_name}.sbatch"
    
    echo "Creating sbatch file for $move_type with $python_file (step_size=$step_size, L=$integration_length)..."
    
    # Create the sbatch file by replacing placeholders
    sed "s/JOB_NAME_PLACEHOLDER/$job_name/g; s/PYTHON_FILE_PLACEHOLDER/$python_file/g; s/MOVE_TYPE_PLACEHOLDER/$move_type/g; s/STEP_SIZE_PLACEHOLDER/$step_size/g; s/INTEGRATION_LENGTH_PLACEHOLDER/$integration_length/g" $SBATCH_TEMPLATE > $sbatch_file
    
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
echo "HMC parameter combinations: ${HMC_PARAMS[*]}"
echo "=========================================="

for move_type in "${MOVE_TYPES[@]}"; do
    for python_file in "${PYTHON_FILES[@]}"; do
        # Check if the python file exists
        if [ -f "$python_file" ]; then
            # Check if this is an HMC-based move type
            if [[ " ${HMC_MOVE_TYPES[@]} " =~ " ${move_type} " ]]; then
                # For HMC moves, create jobs with different parameter combinations
                for params in "${HMC_PARAMS[@]}"; do
                    IFS=',' read -r step_size integration_length <<< "$params"
                    create_and_submit_sbatch "$move_type" "$python_file" "$step_size" "$integration_length"
                done
            else
                # For non-HMC moves, use default parameters (they won't be used anyway)
                create_and_submit_sbatch "$move_type" "$python_file" "0.1" "10"
            fi
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
