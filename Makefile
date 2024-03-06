# Also create .virtualenvs for panoramic_inpainting_sd

git clone git@github.com:IDEA-Research/Grounded-Segment-Anything.git --recurse-submodules

# Docker make sure to run commands 
source /root/.virtualenvs/panoramic_inpainting_sd/bin/activate

# 100 percent should do
apt-get install python3-distutils
apt-get install python3-dev
apt-get install libgl1-mesa-glx

# Not sure, test it
pip install torch

# Double check all, especially AM_I_DOCKER
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/usr/local/cuda/

pip install -e Grounded-Segment-Anything/segment_anything/
pip install -e Grounded-Segment-Anything/GroundingDINO/
pip install --upgrade diffusers[torch]

# Probably do not need as I already used --recurse-submodules
#git submodule update --init --recursive
cd Grounded-Segment-Anything/grounded-sam-osx && bash install.sh
      
# cd ../
git clone https://github.com/xinyu1205/recognize-anything.git
pip install -r ./recognize-anything/requirements.txt
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

pip install -e ./recognize-anything/

pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel


      
 

