# Env setup

## Install uv

```bash
curl -Ls https://uv.vc/install.sh | sh
```

## create venv from pyproject.toml

```bash
uv venv --python 3.13
uv sync
```

## pyproject.toml
```
[project]
name = "maisi"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "parameterized",
  "torch",
  "torchvision",
  "numpy",
  "tqdm",
  "wheel-stub",
  "simpleitk",
  "nibabel",
  "ultralytics-thop",
  "accelerate",
  "torchmetrics",
  "transformers[torch]",
  "monai[all]",
  "focal-frequency-loss"
]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }
```

# Usage

## preprocess_data.py
resize the data and split the data according to the json_split.json, and save the processed data in the data/data dir, and save the datalist.json in the data dir.
## origin data dir structure
```data/
├── floder1
│   ├── image1.nii.gz
│   ├── image2.nii.gz
│   └── ...
├── floder2
│   ├── image1.nii.gz
│   ├── image2.nii.gz
│   └── ...
├── ...
├──json_split.json
```
### json_split.json
```json
{
  "training": ["floder1/image1.nii.gz", "floder2/image1.nii.gz", ...],
  "validation": ["floder1/image2.nii.gz", "floder2/image2.nii.gz", ...],
  "test": ["floder1/image3.nii.gz", "floder2/image3.nii.gz", ...]
}
```

## processed data dir structure
```data/
├── data
│   ├── train
│   │   ├── image1.nii.gz
│   │   ├── image2.nii.gz
│   │   └── ...
│   ├── val
│   │   ├── image1.nii.gz
│   │   ├── image2.nii.gz
│   │   └── ...
│   └── test
│       ├── image1.nii.gz
│       ├── image2.nii.gz
│       └── ...
├── datalist.json
```
### datalist.json
```json
{
  "training": [
        {
          "src_image": "CVAI-0001-src.nii.gz",
          "tar_image": "CVAI-0001-tar.nii.gz",
          "modality": "ct_non_contrast_to_contrast"
        },
        ...
    ],
    "validation": [
        {
        "src_image": "CVAI-0002-src.nii.gz",
        "tar_image": "CVAI-0002-tar.nii.gz",
        "modality": "ct_non_contrast_to_contrast"
        },
        ...
    ],
    "test": [
        {
        "src_image": "CVAI-0002-src.nii.gz",
        "tar_image": "CVAI-0002-tar.nii.gz",
        "modality": "ct_non_contrast_to_contrast"
        },
        ...
    ]
}
```
