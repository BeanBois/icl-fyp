#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your_username>

WORK_DIR="/vol/bitbucket/kst24/fyp/icl-fyp"
CUDA_VERSION="11.8.0"
ENV_NAME="icl-fyp"


export PATH=/vol/bitbucket/kst24/fyp/icl-fyp/:$PATH
source "/vol/cuda/${CUDA_VERSION}/setup.sh"



# Install required packages (if not already installed)
pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install --user pygame numpy matplotlib

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
python -m venv "${ENV_NAME}"
source "${ENV_NAME}/bin/activate"  # On Windows: your_env_name\Scripts\activate
pip install -r requirements.txt

python3 -u "${WORK_DIR}/train.py"