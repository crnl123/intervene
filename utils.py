import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from collections import Counter
from functools import partial

WHOLE_MASK = np.array([[True]])


def show_anns(anns, preprocessed=False):
    """
    From segment-anything github
    :param preprocessed:
    :param anns:
    :return:
    """
    if len(anns) == 0:
        return
    if not preprocessed:
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)  # Sort by mask size
    else:
        sorted_anns = anns
    ax = plt.gca()
    ax.set_autoscale_on(False)

    if not preprocessed:
        img = np.ones(
            (sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))  # Create mask
    else:
        img = np.ones((sorted_anns[0].shape[0], sorted_anns[0].shape[1], 4))
    img[:, :, 3] = 0  # Set alpha 0
    for ann in sorted_anns:
        if not preprocessed:
            m = ann['segmentation']  # Read mask
        else:
            m = ann
        color_mask = np.concatenate([np.random.random(3), [0.35]])  # Set truth pixels to a random colour
        img[m] = color_mask  # Accumulate onto mask
    ax.imshow(img)  # Plot over


def im_save(im, filepath):
    im = Image.fromarray(im)
    im.save(filepath)


def overlap(arr1, arr2):
    """
    Finds fraction of overlapping pixels between 2 arrays
    :param arr1: Boolean array 1
    :param arr2: Boolean array 2
    :return: Fraction of overlap as float 0-1
    """
    larger_sum = max(np.sum(arr1), np.sum(arr2))
    return np.sum(np.logical_and(arr1, arr2)) / larger_sum


def masked_apply(mask, image, intervention, reverse=False):
    """
    Applies intervention to mask area of image.
    :param mask: Where to apply intervention.
    :param image: Image to intervene on.
    :param intervention: Intervention to perform.
    :param reverse: Apply intervention to deselected area instead.
    :return: Intervention result.
    """
    m = mask[:, :, np.newaxis]
    if reverse:
        return np.where(m, image, intervention(image))
    else:
        return np.where(m, intervention(image), image)


def masked_avg_HSV(mask, image, hsv=(True, True, True), acc=None):
    """
    RGB average of masked area.
    :param acc:
    :param mask:
    :param image:
    :param hsv: What elements of HSV colourspace to average over
    :return:
    """
    masked_image = masked_apply(mask, image, partial(make_HSV, (None, None, 0)), reverse=True)
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_HSV2RGB)
    if np.array_equal(mask, WHOLE_MASK):
        avg_color = np.mean(masked_image, axis=(0, 1)).astype('uint8')
    else:
        channel_sum = np.sum(masked_image, axis=(0, 1))
        avg_color = (channel_sum / np.sum(mask)).astype('uint8')
    avg_color = cv2.cvtColor(np.array([[avg_color]]), cv2.COLOR_RGB2HSV)[0][0]
    intervention = (avg_color[0] if hsv[0] else None,
                    avg_color[1] if hsv[1] else None,
                    avg_color[2] if hsv[2] else None)
    if acc is not None:
        return masked_apply(mask, acc, partial(make_HSV, intervention), reverse=False)
    else:
        return masked_apply(mask, image, partial(make_HSV, intervention), reverse=False)


def scrambletest(image):
    """
    YiHong's method
    :param image:
    :return:
    """
    im = cv2.cvtColor(image, cv2.COLOR_HSV2RGB) / 255.0
    im_size = im.shape
    random_phase = np.angle(np.fft.fft2(np.random.rand(im_size[0], im_size[1])))
    random_phase[0] = 0

    # preallocate
    im_fourier = np.zeros(im_size)
    amp = np.zeros(im_size)
    phase = np.zeros(im_size)
    im_scrambled = np.zeros(im_size)

    for layer in range(im_size[2]):
        im_fourier[:, :, layer] = np.fft.fft2(im[:, :, layer])
        amp[:, :, layer] = abs(im_fourier[:, :, layer])
        phase[:, :, layer] = np.angle(im_fourier[:, :, layer])
        phase[:, :, layer] = phase[:, :, layer] + random_phase
        # combine Amp and Phase then perform inverse Fourier
        im_scrambled[:, :, layer] = np.fft.ifft2(amp[:, :, layer] * np.exp(np.sqrt(-1 + 0j) * (phase[:, :, layer])))
    return im_scrambled.real


def masked_scramble(mask, image):
    """
    As per scramblery
    :param mask: True/False binary mask
    :param image: HSV origin image
    :return: HSV scrambled image
    """
    masked_image = masked_apply(mask, image, partial(make_HSV, (None, None, 0)), reverse=True)
    masked_rgb = cv2.cvtColor(masked_image, cv2.COLOR_HSV2RGB)
    masked_gray = cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2GRAY)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    scrambled = phase_scramble(masked_gray)

    src = np.where(mask, scrambled, gray_image)

    clone_mask = np.zeros(mask.shape).astype(np.uint8)
    clone_mask[mask] = 255

    if np.equal(mask, WHOLE_MASK):
        clone_mask = np.ones_like(gray_image)*255

    # Border code exists for cv2 (-215:Assertion failed) error with WHOLE_MASK #

    border = (20, 20, 20, 20)
    border_dst = cv2.copyMakeBorder(rgb_image, *border, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    border_src = cv2.copyMakeBorder(src, *border, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    border_msk = cv2.copyMakeBorder(clone_mask, *border, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # border_dst = rgb_image
    # border_src = src
    # border_msk = clone_mask

    indices = np.where(border_msk)
    min_y, max_y, min_x, max_x = indices[0].min(), indices[0].max(), indices[1].min(), indices[1].max()
    center = tuple(map(round, ((min_x + max_x) / 2, (min_y + max_y) / 2)))

    output = cv2.seamlessClone(border_src, border_dst, border_msk, center, cv2.NORMAL_CLONE)
    output = output[border[0]:-border[1], border[2]:-border[3]]

    return cv2.cvtColor(output, cv2.COLOR_RGB2HSV)


def phase_scramble(gray_image, scramble_ratio=1.0):
    """
    Modified from scramblery
    :param gray_image: Grayscale image to scramble dims h*w
    :param scramble_ratio: Phase scrambling ratio
    :return: Scrambled image
    """
    f_transform = fft2(gray_image)

    f_transform_shift = fftshift(f_transform)

    magnitude = np.abs(f_transform_shift)
    phase = np.angle(f_transform_shift)

    random_phase = np.exp(1j * (2 * np.pi * np.random.rand(*phase.shape) - np.pi))

    new_phase = (1 - scramble_ratio) * phase + scramble_ratio * np.angle(random_phase)

    new_transform = magnitude * np.exp(1j * new_phase)

    f_ishift = ifftshift(new_transform)
    img_back = ifft2(f_ishift)

    img_back = np.real(img_back)

    img_back = (img_back - np.min(img_back)) / (np.max(img_back) - np.min(img_back))
    final_image = (255 * img_back).astype(np.uint8)

    return final_image


def make_HSV(intervention, image):
    """
    Sets pixel value to desired. Leaving Nones in place.
    :param intervention:
    :param image:
    :return:
    """
    inter = np.copy(image.T)
    h, s, v = intervention
    if h is not None:
        inter[0] = [h]
    if s is not None:
        inter[1] = [s]
    if v is not None:
        inter[2] = [v]
    return inter.T


def texture_of(mask1, mask2, threshold):
    m1_size, m2_size = np.sum(mask1), np.sum(mask2)
    intersect_size = np.sum(np.logical_and(mask1, mask2))
    if intersect_size / min(m1_size, m2_size) >= threshold:
        return True
    return False


def find_objects(sorted_masks, threshold):
    objects = []
    textures = []
    for m_dict in sorted_masks:
        texture_found = False
        mask = m_dict['segmentation']
        for i, obj in enumerate(objects):
            if texture_of(obj, mask, threshold):
                texture_found = True
                textures.append((mask, i))
                break
        if not texture_found:
            objects.append(mask)
    return objects, textures


def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def save_mask(masks, file_path, decode=(lambda x: x)):
    for count, m in enumerate(masks):
        mask = decode(m)
        path = file_path(count, m)
        im_save(mask, path)


def show(img):
    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return a1, a2, a3, abs_rel, rmse, log_10


"""
KITTI Velodyne
From MonoDepth 2
"""


def load_velodyne_points(filename):
    """Load 3D point cloud from KITTI file format
    (adapted from https://github.com/hunse/kitti)
    """
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points


def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices
    """
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1


def velodyne_generate(calib_dir, velo_filename, cam=2, vel_depth=False):
    """
    Generate a depth map from velodyne data
    (Taken from MonoDepth2)
    :param calib_dir:
    :param velo_filename:
    :param cam:
    :param vel_depth:
    :return:
    """
    # load calibration files
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # get image shape
    im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_filename)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape[:2]))
    depth[velo_pts_im[:, 1].astype(int), velo_pts_im[:, 0].astype(int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    return depth


if __name__ == '__main__':
    image = cv2.cvtColor(cv2.imread('images/truck.jpg'), cv2.COLOR_BGR2HSV)
    scrambletest(image)
    masked_scramble(WHOLE_MASK, image)
    pass
