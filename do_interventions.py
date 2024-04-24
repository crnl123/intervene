import os
import pickle
from functools import partial
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import utils
from fill_depth_colorization import fill_depth_colorization

IMG_PATH = "images/5.png"
GT_PATH = "images/5gt.png"
FILL_PATH = "images/5fill.png"
SEG_PATH = "images/5masks.pickle"
SAM_PATH = "segment_model/sam_vit_h_4b8939.pth"

TEX1_PATH = "images/5texture_intervention1.png"
TEX2_PATH = "images/5texture_intervention2.png"
HUE1_PATH = "images/5hue_intervention1.png"
HUE2_PATH = "images/5hue_intervention2.png"
VAL1_PATH = "images/5value_intervention1.png"
VAL2_PATH = "images/5value_intervention2.png"
SAT1_PATH = "images/5saturation_intervention1.png"
SAT2_PATH = "images/5saturation_intervention2.png"

# Load image
image = cv2.imread(IMG_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Generate/Load masks
if not os.path.isfile(SEG_PATH):
    # Segmentation Model
    model_type = "vit_h"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=SAM_PATH)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)

    masks = mask_generator.generate(image)
    f = open(SEG_PATH, 'wb')
    pickle.dump(masks, f)
    f.close()
else:
    masks = pickle.load(open(SEG_PATH, 'rb'))

# Prepare image
if not os.path.isfile(FILL_PATH):
    gt_raw = cv2.imread(GT_PATH)
    gt_raw = cv2.cvtColor(gt_raw, cv2.COLOR_BGR2RGB)
    gt_fill = fill_depth_colorization(image.astype(float) / 255, gt_raw[:, :, 0])
    cv2.imwrite(FILL_PATH, gt_fill)
else:
    image = cv2.imread(IMG_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)  # Sort by mask size
objects, textures = utils.find_objects(sorted_masks, .9)

# Texture intervention
acc1 = np.copy(image)
for mask in objects:
    acc1 = utils.masked_avg_HSV(mask, acc1)
acc1 = cv2.cvtColor(acc1, cv2.COLOR_HSV2BGR)
cv2.imwrite(TEX1_PATH, acc1)

acc2 = np.copy(image)
# acc2 = utils.masked_scramble(utils.WHOLE_MASK, acc2)
for mask in objects:
    acc2 = utils.masked_scramble(mask, acc2)
acc2 = cv2.cvtColor(acc2, cv2.COLOR_HSV2BGR)
cv2.imwrite(TEX2_PATH, acc2)

# Hue intervention - random
acc1 = np.copy(image)
for mask in objects:
    acc1 = utils.masked_apply(mask, acc1, partial(utils.make_HSV, (np.random.randint(0,256), None, None)))
acc1 = cv2.cvtColor(acc1, cv2.COLOR_HSV2BGR)
cv2.imwrite(HUE1_PATH, acc1)

# Hue intervention - uniform
acc2 = np.copy(image)
acc2 = utils.masked_avg_HSV(utils.WHOLE_MASK, acc2, hsv=(True, False, False))
acc2 = cv2.cvtColor(acc2, cv2.COLOR_HSV2BGR)
cv2.imwrite(HUE2_PATH, acc2)

# Value intervention - random
acc1 = np.copy(image)
for mask in objects:
    acc1 = utils.masked_apply(mask, acc1, partial(utils.make_HSV, (None, None, np.random.randint(0,256))))
acc1 = cv2.cvtColor(acc1, cv2.COLOR_HSV2BGR)
cv2.imwrite(VAL1_PATH, acc1)

# Value intervention - uniform
acc2 = np.copy(image)
acc2 = utils.masked_avg_HSV(utils.WHOLE_MASK, acc2, hsv=(False, False, True))
acc2 = cv2.cvtColor(acc2, cv2.COLOR_HSV2BGR)
cv2.imwrite(VAL2_PATH, acc2)

# Saturation intervention - random
acc1 = np.copy(image)
for mask in objects:
    acc1 = utils.masked_apply(mask, acc1, partial(utils.make_HSV, (None, np.random.randint(0,256), None)))
acc1 = cv2.cvtColor(acc1, cv2.COLOR_HSV2BGR)
cv2.imwrite(SAT1_PATH, acc1)

# Saturation intervention - uniform
acc2 = np.copy(image)
acc2 = utils.masked_avg_HSV(utils.WHOLE_MASK, acc2, hsv=(False, True, False))
acc2 = cv2.cvtColor(acc2, cv2.COLOR_HSV2BGR)
cv2.imwrite(SAT2_PATH, acc2)
