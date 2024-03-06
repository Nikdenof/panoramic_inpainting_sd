import sys
import torch
from groundingdino.util.slconfig import SLConfig
import cv2
import numpy as np
from GSAM.VISAM.util.box_ops import box_iou

arr_img = np.array([[5, 8], [0, 255]], dtype=np.uint8)
arr_img_resize = cv2.resize(arr_img, (300, 300))
# arr_img_load = cv2.imread("./data/raw/test1.jpg")
# Get the default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Default Device:", device)

python_version = sys.version
print("Python Version:", python_version)

# Get Python executable location
python_executable = sys.executable
print("Python Executable:", python_executable)
