#!/bin/bash
#SBATCH --job-name=compress_data
#SBATCH --account=MST111121
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/compress_%j.out
#SBATCH --error=logs/compress_%j.err

#SBATCH --mail-type=BEGIN,END,FAIL               # send email at job start, end, and on failure
#SBATCH --mail-user=r12946008@ntu.edu.tw        # your address

#SBATCH --time=1-00:00:00

#SBATCH --export=ALL

# Load your shared environment (if needed)
# module load Anaconda3/2021.11  # uncomment if your site uses module system
cd /work/r12946008/maisi
# Ensure the logs directory exists

mkdir -p logs

# Activate conda environment
source ~/.bashrc
conda activate maisi

# Force Python to flush stdout/stderr immediately
export PYTHONUNBUFFERED=1


# Run the training script
start_time=$(date +%s)
tar -cf - /work/r12946008/CENC_CEfixed | zstd --ultra -22 --long=31 -T0 -o /work/r12946008/CENC_CEfixed.tar.zst
end_time=$(date +%s)
echo "Job finished at $(date)"

duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))
echo "執行時間: ${minutes} 分鐘 ${seconds} 秒"
