import cv2
import numpy as np
import torch

import utils

filename = 'images/5.png'

model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_batch = transform(img).to(device)

with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output = prediction.cpu().numpy()

GT_PATH = "images/5fill.png"
gt_depth = cv2.imread(GT_PATH)
singular_gt = cv2.cvtColor(gt_depth, cv2.COLOR_RGB2GRAY)

# Process to relative depth from 0-1
singular_gt = singular_gt-np.min(singular_gt)
singular_gt = 255 * (1-singular_gt/np.max(singular_gt))

output = output-np.min(output)
output = 255 * (output/np.max(output))

utils.show(output)
utils.show(singular_gt)

print(utils.compute_errors(singular_gt, output))

