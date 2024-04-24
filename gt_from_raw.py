"""
Do dense completion from velodyne point cloud
"""
import utils
import cv2
from fill_depth_colorization import fill_depth_colorization


calib_path = 'raw'
velo_path = 'raw/0000000005.bin'
im_path = "raw/5.png"
out_path = "images/5fill_from_velo.png"

gt_raw = utils.velodyne_generate(calib_path, velo_path, 2, True)

image = cv2.imread(im_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255.0

gt_depth = fill_depth_colorization(image, gt_raw)

utils.show(gt_depth)
cv2.imwrite(out_path, gt_depth)  # Need BGR
