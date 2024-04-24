import utils
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
import numpy as np


# Snippet from Depth Anything github
# https://github.com/LiheYoung/Depth-Anything?tab=readme-ov-file#import-depth-anything-to-your-project
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder = 'vits' # can also be 'vitb' or 'vitl'
depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder)).to(DEVICE).eval()

transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

# NYU
# f = np.load('raw/nyu_v2/nyu_v2.npz')
# depths = np.transpose(f['depths'], (0, 2, 1))  # 1449, 480, 640
# images = np.transpose(f['images'], (0, 3, 2, 1))  # 1449, 480, 640, 3
# instances = np.transpose(f['instances'], (0, 2, 1))  # 1449, 480, 640
# labels = np.transpose(f['labels'], (0, 2, 1))  # 1449, 480, 640

image = cv2.cvtColor(cv2.imread('images/5.png'), cv2.COLOR_BGR2RGB) / 255.0
# image = images[1][10:-10,10:-10]
utils.show(image)
h, w = image.shape[:2]
image = transform({'image': image})['image']
image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

# depth shape: 1xHxW
with torch.no_grad():
    depth = depth_anything(image)

# GT
GT_PATH = "images/5fill.png"
gt_depth = cv2.imread(GT_PATH)
singular_gt = cv2.cvtColor(gt_depth, cv2.COLOR_RGB2GRAY)
# singular_gt = depths[1][10:-10,10:-10]

# Process to relative depth from 0-1
singular_gt = singular_gt-np.min(singular_gt)
singular_gt = 255 * (1-singular_gt/np.max(singular_gt))
# singular_gt = 255 * (singular_gt/np.max(singular_gt))  #NYU reversed

# Inverse depth
# singular_gt = 1/singular_gt

depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
depth = depth.cpu().numpy().astype(np.uint8)

# cv2.imwrite('images/test.png', depth)
utils.show(singular_gt)
utils.show(depth)

print(utils.compute_errors(singular_gt, depth))
pass
