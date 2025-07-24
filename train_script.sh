#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your_username>

WORK_DIR="/vol/bitbucket/kst24/fyp/icl-fyp"
CUDA_VERSION="11.8.0"
ENV_NAME="icl-fyp"
GIT_BRANCH="your-branch-name"  # Change this to your desired branch

export PATH=/vol/bitbucket/kst24/fyp/icl-fyp/:$PATH
source "/vol/cuda/${CUDA_VERSION}/setup.sh"

# Navigate to work directory
cd "${WORK_DIR}"

# Git operations - fetch latest and checkout specific branch
echo "Fetching latest git changes..."
git fetch origin
echo "Checking out branch: ${GIT_BRANCH}"
git checkout ${GIT_BRANCH}
git pull origin ${GIT_BRANCH}

# load conda env from .yml file
# Load conda
module load Anaconda3/2022.05

# Check if environment exists, if not create it
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "Creating conda environment from yml file..."
    conda env create -f "${WORK_DIR}/environment.yml" --name ${ENV_NAME}
else
    echo "Environment ${ENV_NAME} already exists, activating..."
fi

# Activate the environment
conda activate ${ENV_NAME}

# Install required packages (if not already installed)
pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install --user pygame numpy matplotlib

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

python3 -u "${WORK_DIR}/train.py"