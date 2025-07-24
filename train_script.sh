#!/bin/bash
#SBATCH --job-name=instant_policy_training
#SBATCH --output=instant_policy_%j.out
#SBATCH --error=instant_policy_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpgpuB
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@imperial.ac.uk

# Load required modules
module load tools/prod
module load Python/3.9.6-GCCcore-11.2.0
module load CUDA/11.4.1
module load cuDNN/8.2.4.15-CUDA-11.4.1

# Activate virtual environment (if you have one)
# source /path/to/your/venv/bin/activate

# Or create conda environment
# module load Miniconda3/4.10.3
# conda activate your_env

# Install required packages (if not already installed)
pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install --user pygame numpy matplotlib

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:/rds/general/user/$USER/home/your_project_path

# Change to your project directory
cd /rds/general/user/$USER/home/your_project_path

# Run your training script
python train.py

# Optional: Run geometry encoder pre-training first
# python geometry_encoder.py

echo "Job completed at $(date)"