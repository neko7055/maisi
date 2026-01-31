import copy
import os
import json
import numpy as np
import nibabel as nib
import subprocess

import torch
from monai.config import print_config

from scripts.diff_model_setting import setup_logging

def list_gz_files(folder_path):
    """List all .gz files in the folder and its subfolders."""
    gz_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".gz"):
                gz_files.append(os.path.join(root, file))
    return gz_files


def create_json_files(gz_files, modality, include_body_region):
    """Create .json files for each .gz file with the specified keys and values."""
    for gz_file in gz_files:
        # Load the NIfTI image
        img = nib.load(gz_file)

        # Get the dimensions and spacing
        dimensions = img.shape
        dimensions = dimensions[:3]
        spacing = img.header.get_zooms()[:3]
        spacing = spacing[:3]
        spacing = [float(_item) for _item in spacing]

        # Create the dictionary with the specified keys and values
        data = {"dim": dimensions, "spacing": spacing, "modality": modality}
        if include_body_region:
            # The region can be selected from one of four regions from top to bottom.
            # [1,0,0,0] is the head and neck, [0,1,0,0] is the chest region, [0,0,1,0]
            # is the abdomen region, and [0,0,0,1] is the lower body region.
            data["top_region_index"] = [0, 1, 0, 0]  # chest region
            data["bottom_region_index"] = [0, 0, 1, 0]  # abdomen region
        logger.info(f"data: {data}.")

        # Create the .json filename
        json_filename = gz_file + ".json"

        # Write the dictionary to the .json file
        with open(json_filename, "w") as json_file:
            json.dump(data, json_file, indent=4)
        logger.info(f"Save json file to {json_filename}")

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

    logger = setup_logging("transform_data_to_laten_space")

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


    logger.info("Creating training data...")
    module = "scripts.diff_model_create_training_data"
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

    folder_path = env_config["embedding_base_dir"]
    gz_files = list_gz_files(folder_path)
    create_json_files(gz_files, "ct", model_def["include_body_region"])

    logger.info("Completed creating .json files for all embedding files.")
