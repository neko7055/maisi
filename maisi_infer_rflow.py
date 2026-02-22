import os
import json
import subprocess

import torch
from monai.config import print_config

from scripts.diff_model_setting import setup_logging
def run_torchrun(module, module_args, num_gpus=1):
    # Define the arguments for torchrun
    num_nodes = 1

    # Build the torchrun command
    torchrun_command = [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        str(num_gpus),
        "--nnodes",
        str(num_nodes),
        "-m",
        module,
    ] + module_args


    # Set the OMP_NUM_THREADS environment variable
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"

    # Execute the command
    process = subprocess.Popen(torchrun_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

    # Print the output in real-time
    try:
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                print(output.strip())
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Capture and print any remaining output
        stdout, stderr = process.communicate()
        print(stdout)
        if stderr:
            print(stderr)
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

    with open(env_config_path, "r") as f:
        env_config = json.load(f)

    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    with open(model_def_path, "r") as f:
        model_def = json.load(f)


    logger.info("Training the model...")
    module = "scripts.diff_model_train_with_accelerate"
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

    run_torchrun(module, module_args, num_gpus=num_gpus)