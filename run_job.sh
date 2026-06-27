#!/bin/bash
#SBATCH --job-name=sim_run
#SBATCH --partition=universe
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/sim_%j.out
#SBATCH --error=logs/sim_%j.err

module load python/anaconda3
module load cuda/cuda_11.7

export CPATH="$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))'):$CPATH"
export C_INCLUDE_PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))'):$C_INCLUDE_PATH"
export LIBRARY_PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_path("stdlib"))'):$LIBRARY_PATH"

export VLLM_DEVICE=cuda
export CUDA_HOME=$CUDA_INSTALL_PATH

uv run python scripts/run_patient_comparison.py --config-name patient_comparison_default
