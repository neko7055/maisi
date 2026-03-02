#!/bin/bash
#SBATCH --job-name=vae_training
#SBATCH --account=MST111121
#SBATCH --partition=normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/vae_acc_%j.out
#SBATCH --error=logs/vae_acc_%j.err

#SBATCH --mail-type=BEGIN,END,FAIL               # send email at job start, end, and on failure
#SBATCH --mail-user=r12946008@ntu.edu.tw        # your address

#SBATCH --time=01:00:00

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
export HOST_NODE_ADDR=127.0.0.1:29501
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501

echo "Master Address set to: $MASTER_ADDR"
echo "Master Port set to: $MASTER_PORT"

# Show some diagnostics at start
echo "Job started on $(hostname) at $(date)"
nvidia-smi

# Run the training script
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 1 > ./logs/vae_gpu_log.csv & MONITOR_PID=$!
start_time=$(date +%s)
uv run maisi_train_vae.py
end_time=$(date +%s)
echo "Job finished at $(date)"
kill $MONITOR_PID
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))
echo "Execution time: ${minutes} minutes ${seconds} seconds"