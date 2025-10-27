#!/bin/bash

# Script to generate and submit sbatch jobs for all possible move types
# Based on the move types defined in configuration.py

# Define all possible move types from configuration.py
MOVE_TYPES=("hmc" "hmc_walk" "hmc_side" "stretch" "walk" "side")

# Define HMC-based move types that need parameter variations
HMC_MOVE_TYPES=("hmc" "hmc_walk" "hmc_side")

# Define parameter combinations for HMC moves: (step_size, integration_length, adapt_step_size, adapt_length)
# Format: "step_size,integration_length,adapt_step_size,adapt_length"
HMC_PARAMS=("0.5,2,false,false" "0.1,10,false,false")

# Define the Python files to run (you can modify this list as needed)
PYTHON_FILES=("gaussian.py" "allen_cahn.py" "ring.py" "gaussian_mixture.py")

# Base sbatch template (will be modified for each job)
SBATCH_TEMPLATE="submit_template.sbatch"

# Create the base sbatch template
cat > $SBATCH_TEMPLATE << 'EOF'
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=TIME_PLACEHOLDER
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=200GB
#SBATCH --job-name=JOB_NAME_PLACEHOLDER
#SBATCH --mail-type=END
#SBATCH --mail-user=cm6627@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

module purge

OVERLAY_FILE=/scratch/cm6627/hemcee_env/overlay-15GB-500K.ext3:ro
SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif

PYTHON_FILE=PYTHON_FILE_PLACEHOLDER
export PYTHONUNBUFFERED=1

MOVE_TYPE=MOVE_TYPE_PLACEHOLDER
STEP_SIZE=STEP_SIZE_PLACEHOLDER
INTEGRATION_LENGTH=INTEGRATION_LENGTH_PLACEHOLDER
ADAPT_STEP_SIZE=ADAPT_STEP_SIZE_PLACEHOLDER
ADAPT_LENGTH=ADAPT_LENGTH_PLACEHOLDER

# Build command with adaptation flags
CMD="python $PYTHON_FILE --move $MOVE_TYPE --hamiltonian_step_size $STEP_SIZE --hamiltonian_L $INTEGRATION_LENGTH"
if [ "$ADAPT_STEP_SIZE" = "true" ]; then
    CMD="$CMD --adapt_step_size"
fi
if [ "$ADAPT_LENGTH" = "true" ]; then
    CMD="$CMD --adapt_length"
fi

singularity exec --nv \
  --overlay $OVERLAY_FILE \
  $SINGULARITY_IMAGE \
  /bin/bash -c "source /ext3/env.sh; $CMD"
EOF

echo "Generated sbatch template: $SBATCH_TEMPLATE"

# Function to determine time limit based on Python file
get_time_limit() {
    local python_file=$1
    case "$python_file" in
        "gaussian.py"|"gaussian_mixture.py")
            echo "3:00:00"
            ;;
        *)
            echo "2:00:00"
            ;;
    esac
}

# Function to create and submit sbatch for a specific move type and python file
create_and_submit_sbatch() {
    local move_type=$1
    local python_file=$2
    echo "Creating sbatch file for $move_type with $python_file (step_size=$step_size, L=$integration_length, adapt_step_size=$adapt_step_size, adapt_length=$adapt_length)..."
    local step_size=$3
    local integration_length=$4
    local adapt_step_size=$5
    local adapt_length=$6
    local time_limit=$7
    local job_name="${python_file%.*}_${move_type}_s${step_size}_L${integration_length}_adapt${adapt_step_size}${adapt_length}"
    local sbatch_file="submit_${job_name}.sbatch"
    
    # Create the sbatch file by replacing placeholders
    sed "s/JOB_NAME_PLACEHOLDER/$job_name/g; s/PYTHON_FILE_PLACEHOLDER/$python_file/g; s/MOVE_TYPE_PLACEHOLDER/$move_type/g; s/STEP_SIZE_PLACEHOLDER/$step_size/g; s/INTEGRATION_LENGTH_PLACEHOLDER/$integration_length/g; s/ADAPT_STEP_SIZE_PLACEHOLDER/$adapt_step_size/g; s/ADAPT_LENGTH_PLACEHOLDER/$adapt_length/g; s/TIME_PLACEHOLDER/$time_limit/g" $SBATCH_TEMPLATE > $sbatch_file
    
    # Make the sbatch file executable
    chmod +x $sbatch_file
    
    echo "Submitting job: $job_name"
    sbatch $sbatch_file
    
    # Wait a bit (0.5 seconds) to ensure the HPC scheduler sees the sbatch file before removal
    sleep 0.5  # units: seconds
    rm $sbatch_file
    
    echo "Submitted: $job_name (sbatch file removed)"
    echo "---"
}

# Special adaptation experiments for hmc_walk
echo "Starting adaptation experiments for hmc_walk..."
echo "=========================================="

# Experiment 1: Adaptive step size and integration length
echo "Experiment 1: Adaptive parameters (adapt_step_size=True, adapt_length=True)"
for python_file in "${PYTHON_FILES[@]}"; do
    if [ -f "$python_file" ]; then
        time_limit=$(get_time_limit "$python_file")
        create_and_submit_sbatch "hmc_walk" "$python_file" "0.1" "10" "true" "true" "$time_limit"
    fi
done

# Experiment 2: Fixed parameters (small step size, long integration)
echo "Experiment 2: Fixed parameters (step_size=0.1, integration_length=10)"
for python_file in "${PYTHON_FILES[@]}"; do
    if [ -f "$python_file" ]; then
        time_limit=$(get_time_limit "$python_file")
        create_and_submit_sbatch "hmc_walk" "$python_file" "0.1" "10" "false" "false" "$time_limit"
    fi
done

# Experiment 3: Fixed parameters (large step size, short integration)
echo "Experiment 3: Fixed parameters (step_size=0.5, integration_length=2)"
for python_file in "${PYTHON_FILES[@]}"; do
    if [ -f "$python_file" ]; then
        time_limit=$(get_time_limit "$python_file")
        create_and_submit_sbatch "hmc_walk" "$python_file" "0.5" "2" "false" "false" "$time_limit"
    fi
done

echo "=========================================="
echo "Adaptation experiments submitted!"
echo "=========================================="

# Main loop: iterate over all combinations of move types and python files
echo "Starting job submission for all other move types..."
echo "Move types: ${MOVE_TYPES[*]}"
echo "Python files: ${PYTHON_FILES[*]}"
echo "HMC parameter combinations: ${HMC_PARAMS[*]}"
echo "=========================================="

for move_type in "${MOVE_TYPES[@]}"; do
    for python_file in "${PYTHON_FILES[@]}"; do
        # Check if the python file exists
        if [ -f "$python_file" ]; then
            # Skip hmc_walk since we already handled it above
            if [ "$move_type" = "hmc_walk" ]; then
                continue
            fi
            
            # Get time limit for this Python file
            time_limit=$(get_time_limit "$python_file")
            
            # Check if this is an HMC-based move type
            if [[ " ${HMC_MOVE_TYPES[@]} " =~ " ${move_type} " ]]; then
                # For HMC moves, create jobs with different parameter combinations
                for params in "${HMC_PARAMS[@]}"; do
                    IFS=',' read -r step_size integration_length adapt_step_size adapt_length <<< "$params"
                    create_and_submit_sbatch "$move_type" "$python_file" "$step_size" "$integration_length" "$adapt_step_size" "$adapt_length" "$time_limit"
                done
            else
                # For non-HMC moves, use default parameters (they won't be used anyway)
                create_and_submit_sbatch "$move_type" "$python_file" "0.1" "10" "false" "false" "$time_limit"
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
