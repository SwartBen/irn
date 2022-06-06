import torch
import importlib
import numpy as np
import cv2

# Make sure to have the right CAM set in net.resnet50_cam initialisation

# Testing Resnet50 CAM

"""
model = getattr(importlib.import_module('net.resnet50_cam'), 'CAM')()
print("Resnet50_1")
model.load_state_dict(torch.load('/home/bswart/data/result_orig/sess/res50_cam.pth' + '.pth'), strict=True)
print("Resnet50_2")
"""

# Testing Resnet101 CAM
"""
print("Resnet101_1")
model.load_state_dict(torch.load('/home/bswart/data/result_resnet101/sess/res50_cam.pth' + '.pth'), strict=True)
print("Resnet101_2")
"""

# Testing Resnet152 CAM
"""
model = getattr(importlib.import_module('net.resnet50_cam'), 'CAM')()
print("Resnet152_1")
model.load_state_dict(torch.load('/home/bswart/data/result_resnet152/sess/res50_cam.pth' + '.pth'), strict=True)
print("Resnet152_2")
"""

path = '/home/bswart/data/result_orig/sem_seg/2007_000032.png'
#data = np.load(path, allow_pickle=True)
#print(data)

img = cv2.imread(path)
for row in img:
    print(row)
