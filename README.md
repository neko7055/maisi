# install list

conda create -n maisi python=3.13 uv compilers ninja make cmake -c conda-forge

uv pip install numpy tqdm simpleitk nibabel torch torchvision accelerate torchmetrics "transformers[torch]"

BUILD_MONAI=1 uv pip install --no-build-isolation \
  "git+https://github.com/Project-MONAI/MONAI.git#egg=monai[all]"