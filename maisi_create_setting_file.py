import os
import json
import logging

logger = logging.getLogger("create_setting_file")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# network config
generate_version = "rflow-ct"
if generate_version == "ddpm-ct":
    model_def_path = "./configs/config_network_ddpm.json"
elif generate_version == "rflow-ct" or generate_version == "rflow-mr":
    model_def_path = "./configs/config_network_rflow.json"
else:
    raise ValueError(f"generate_version has to be chosen from ['ddpm-ct', 'rflow-ct', 'rflow-mr'], yet got {generate_version}.")
with open(model_def_path, "r") as f:
    model_def = json.load(f)
include_body_region = model_def["include_body_region"]
logger.info(f"Models are {generate_version}, whether to use body_region is {include_body_region}")


# Set up directories based on configurations

## data path setup
env_config_out = dict()
env_config_out["data_base_dir"] = "./sim_data_dir" # data path
env_config_out["json_data_list"] = "./sim_data_dir/sim_datalist.json" # data list
env_config_out["embedding_base_dir"] = env_config_out["data_base_dir"] + "_embeddings"
os.makedirs(env_config_out["embedding_base_dir"], exist_ok=True)

# work space setup
env_config_out["work_dir"] = "./temp_work_dir"
env_config_out["model_dir"] = os.path.join(env_config_out["work_dir"], "models")
env_config_out["model_filename"] = "my_train_rflow.pt"
env_config_out["output_dir"] = os.path.join(env_config_out["work_dir"], "predictions")
env_config_out["output_prefix"] = "unet_3d"
env_config_out["trained_autoencoder_path"] = os.path.join(env_config_out["model_dir"], "autoencoder_v1.pt")
env_config_out["existing_ckpt_filepath"] = os.path.join(env_config_out["model_dir"], "diff_unet_3d_rflow-ct.pt")
env_config_out["modality_mapping_path"] = os.path.join(env_config_out["work_dir"], "configs", "modality_mapping.json")

# download pre-trained model from https://huggingface.co/nvidia/NV-Generate-CT
# trained_autoencoder_path_url = (
#     "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/"
#     "model_zoo/model_maisi_autoencoder_epoch273_alternative.pt"
# )
# if not os.path.exists(env_config_out["trained_autoencoder_path"]):
#     download_url(url=trained_autoencoder_path_url, filepath=env_config_out["trained_autoencoder_path"])

# Create necessary directories
os.makedirs(env_config_out["work_dir"], exist_ok=True)
os.makedirs(env_config_out["model_dir"], exist_ok=True)
os.makedirs(env_config_out["output_dir"], exist_ok=True)

# Update model configuration for demo
model_config_out = dict()
model_config_out["diffusion_unet_train"] = dict()
model_config_out["diffusion_unet_train"]["batch_size"] = 1
model_config_out["diffusion_unet_train"]["gradient_accumulation_steps"] = 4
model_config_out["diffusion_unet_train"]["cache_rate"] = 0
model_config_out["diffusion_unet_train"]["lr"] = 0.0001
model_config_out["diffusion_unet_train"]["n_epochs"] = 1000
model_config_out["diffusion_unet_train"]["validation_num_steps"] = 2

# modality mapping json

modality_mapping = {
    "unknown":0,
    "ct":1,
    "ct_wo_contrast":2,
    "ct_contrast":3,
    "mri":8,
    "mri_t1":9,
    "mri_t2":10,
    "mri_flair":11,
    "mri_pd":12,
    "mri_dwi":13,
    "mri_adc":14,
    "mri_ssfp":15,
    "mri_mra":16,
    "mri_t1c":17
}

# save config into work space
configs_dir = os.path.join(env_config_out["work_dir"], "configs")
os.makedirs(configs_dir, exist_ok=True)
env_config_path = os.path.join(configs_dir, "environment_train_infer_config.json")
model_config_path = os.path.join(configs_dir, "model_train_infer_config.json")
model_def_path = os.path.join(configs_dir, "model_def.json")
modality_mapping_path = os.path.join(configs_dir, "modality_mapping.json")
with open(env_config_path, "w") as f:
    json.dump(env_config_out, f, sort_keys=True, indent=4)

with open(model_config_path, "w") as f:
    json.dump(model_config_out, f, sort_keys=True, indent=4)

with open(model_def_path, "w") as f:
    json.dump(model_def, f, sort_keys=True, indent=4)

with open(modality_mapping_path, "w") as f:
    json.dump(modality_mapping, f, sort_keys=True, indent=4)