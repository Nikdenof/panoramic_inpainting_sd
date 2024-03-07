SHELL := /bin/bash

# These are comments
# Also create .virtualenvs for inpainting_sd
#git clone git@github.com:IDEA-Research/Grounded-Segment-Anything.git --recurse-submodules
#source /root/.virtualenvs/panoramic_inpainting_sd/bin/activate

all: system_deps pip_deps set_env install_grounded_segment install_recognize_anything post_install

# Install system dependencies
system_deps:
	sudo apt-get update
	sudo apt-get install -y python3-distutils python3-dev libgl1-mesa-glx

# Install Python packages
pip_deps:
	pip install --upgrade setuptools
	pip install torch
	pip install --upgrade diffusers[torch]
	pip install numpy opencv-python pycocotools matplotlib onnxruntime onnx ipykernel

set_env:
	export AM_I_DOCKER=False
	export BUILD_WITH_CUDA=True
	export CUDA_HOME=/usr/local/cuda/

install_grounded_segment:
	pip install -e GSAM/segment_anything/
	pip install -e GSAM/GroundingDINO/
	cd GSAM/grounded-sam-osx && bash install.sh

# If there is no GPU for some reason, rebuild dino
rebuid_dino:
	pip install torch --upgrade
	pip install torchvision --upgrade
	pip install -e GSAM/GroundingDINO/

install_recognize_anything:
	git clone https://github.com/xinyu1205/recognize-anything.git || true
	pip install -r recognize-anything/requirements.txt
	pip install -e recognize-anything/

post_install:
	@echo "Installation complete. Adjust environment variables as needed."

# if -e does not work redo setup.cfg
# [options]
# packages = find:
# include_package_data = True
# install_requires =
#     timm==0.4.12
#     transformers==4.25.1
#     fairscale==0.4.4
#     pycocoevalcap
#     torch
#     torchvision
#     Pillow
#     scipy
#     clip @ git+https://github.com/openai/CLIP.git

.PHONY: all system_deps pip_deps set_env install_grounded_segment install_recognize_anything post_install


      
 

