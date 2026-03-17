import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import subprocess

import torch
from monai.config import print_config

from scripts.diff_model_setting import setup_logging
def run_torchrun(module, module_args,accelerate_config_path, num_gpus=1):
    # Define the arguments for torchrun
    num_nodes = 1

    # Build the torchrun command
    # torchrun_command = [
    #     "torchrun",
    #     "--standalone",
    #     "--nproc_per_node",
    #     str(num_gpus),
    #     "--nnodes",
    #     str(num_nodes),
    #     "-m",
    #     module,
    # ] + module_args

    accelerate_command = [
        "accelerate",
        "launch",
        "--config_file",
        accelerate_config_path,
        "-m",
        module,
    ] + module_args


    # Set the OMP_NUM_THREADS environment variable
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"

    # Execute the command
    process = subprocess.Popen(accelerate_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

    # Print the output in real-time
    try:
        for line in process.stdout:  # 比 readline() 更 Pythonic
            print(line, end="")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        process.stdout.close()
        return_code = process.wait()  # ← 用 wait() 而非 communicate()
        print(f"Process exited with code: {return_code}")
    return

if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()

    print_config()

    logger = setup_logging("train_maisi_rflow")

    # load config
    work_dir = "./temp_work_dir"
    configs_dir = os.path.join(work_dir, "configs")
    env_config_path = os.path.join(configs_dir, "environment_train_infer_config.json")
    model_config_path = os.path.join(configs_dir, "model_train_infer_config.json")
    model_def_path = os.path.join(configs_dir, "model_def.json")

    accelerate_config_path = "./accelerate_configs/cuda_8gpu_1node_bf16.yaml"

    with open(env_config_path, "r") as f:
        env_config = json.load(f)

    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    with open(model_def_path, "r") as f:
        model_def = json.load(f)


    logger.info("Training the model...")
    module = "scripts.vae_model_train"
    module_args = [
        "--env_config",
        env_config_path,
        "--model_config",
        model_config_path,
        "--model_def",
        model_def_path,
        "--num_gpus",
        str(num_gpus),
    ]

    run_torchrun(module, module_args,accelerate_config_path, num_gpus=num_gpus)