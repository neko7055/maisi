#!/bin/bash
#SBATCH --job-name=rflow_training
#SBATCH --account=MST111121
#SBATCH --partition=normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/rflow_acc_%j.out
#SBATCH --error=logs/rflow_acc_%j.err

#SBATCH --mail-type=BEGIN,END,FAIL               # send email at job start, end, and on failure
#SBATCH --mail-user=r12946008@ntu.edu.tw        # your address

#SBATCH --time=01:00:00

#SBATCH --export=ALL

# Load your shared environment (if needed)
# module load Anaconda3/2021.11  # uncomment if your site uses module system
cd /work/r12946008/maisi
# Ensure the logs directory exists

mkdir -p logs

# Activate conda environment
source ~/.bashrc
source .ven/bin/activate

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
echo "Using conda env: $(conda info --envs | grep monai)"

# Run the training script
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 1 > ./logs/gpu_log.csv & MONITOR_PID=$!
start_time=$(date +%s)
uv run maisi_train_rflow.py
end_time=$(date +%s)
echo "Job finished at $(date)"
kill $MONITOR_PID
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))
echo "執行時間: ${minutes} 分鐘 ${seconds} 秒"