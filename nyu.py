import numpy as np
import utils
import cv2

f = np.load('raw/nyu_v2/nyu_v2.npz')
depths = np.transpose(f['depths'], (0, 2, 1))  # 1449, 480, 640
images = np.transpose(f['images'], (0, 3, 2, 1))  # 1449, 480, 640, 3
instances = np.transpose(f['instances'], (0, 2, 1))  # 1449, 480, 640
labels = np.transpose(f['labels'], (0, 2, 1))  # 1449, 480, 640

im = cv2.cvtColor(images[0], cv2.COLOR_RGB2BGR)
cv2.imwrite('images/nyu0.png', im)

gt = cv2.cvtColor(depths[0], cv2.COLOR_GRAY2BGR)
# Process to relative depth from 0-1
gt = gt-np.min(gt)
gt = 255 * (1-gt/np.max(gt))
cv2.imwrite('images/nyu0gt.png', gt)
pass
