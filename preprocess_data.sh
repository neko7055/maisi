#!/bin/bash
#SBATCH --job-name=preprocess_data
#SBATCH --account=MST111121
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err

#SBATCH --mail-type=BEGIN,END,FAIL               # send email at job start, end, and on failure
#SBATCH --mail-user=r12946008@ntu.edu.tw        # your address

#SBATCH --time=1-00:00:00

#SBATCH --export=ALL

PROJECT_DIR="/work/r12946008/maisi"

cd "$PROJECT_DIR" || { echo "切換目錄失敗: $PROJECT_DIR"; exit 1; }
# Ensure the logs directory exists

mkdir -p logs

# Activate conda environment
source "$HOME/.bashrc"
source .venv/bin/activate

# Force Python to flush stdout/stderr immediately
export PYTHONUNBUFFERED=1

# Run the training script
start_time=$(date +%s)
uv run preprocess_data.py
end_time=$(date +%s)
echo "Job finished at $(date)"

duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))
echo "Execution time: ${minutes} minutes ${seconds} seconds"
