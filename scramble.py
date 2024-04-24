import cv2
import numpy as np


def phaseScramble_depth(rgb_path, depth_path, saving_root='./'):

    img_BGR = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    # img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    img = img_BGR

    # remove '\n' at the end of the depth_path if there is one
    if depth_path[-1] == '\n':
        depth_path = depth_path[:-1]
    depth_BGR = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    # depth = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    depth = cv2.merge((depth_BGR,depth_BGR,depth_BGR))

    # rescale = 'off'
    p = 1

    # imclass = class(im); % get class of image

    # im = np.double(img)
    im = img
    imSize = img.shape
    depthSize = depth.shape

    # RandomPhase = p * np.angle(np.fft.fft2(np.random.rand(imSize[1], imSize[2]))) # generate random phase structure in range p (between 0 and 1)
    RandomPhase = p * np.angle(np.fft.fft2(np.random.rand(imSize[0], imSize[1]))) # generate random phase structure in range p (between 0 and 1)
    # RandomPhase(1) = 0 # leave out the DC value
    RandomPhase[0] = 0 # leave out the DC value ????

    if len(imSize) == 2:
        imSize[2] = 1

    # preallocate
    imFourier = np.zeros(imSize)
    imFourier_depth = np.zeros(depthSize)
    Amp = np.zeros(imSize)
    Amp_depth = np.zeros(depthSize)
    Phase = np.zeros(imSize)
    Phase_depth = np.zeros(depthSize)
    imScrambled = np.zeros(imSize)
    imScrambled_depth = np.zeros(depthSize)

    # for layer = 1:imSize(3)
    for layer in range(imSize[2]):
        imFourier[:,:,layer] = np.fft.fft2(im[:,:,layer])         # Fast-Fourier transform
        Amp[:,:,layer] = abs(imFourier[:,:,layer])         # amplitude spectrum
        Phase[:,:,layer] = np.angle(imFourier[:,:,layer])     # phase spectrum
        Phase[:,:,layer] = Phase[:,:,layer] + RandomPhase  # add random phase to original phase
        # combine Amp and Phase then perform inverse Fourier
        imScrambled[:,:,layer] = np.fft.ifft2(Amp[:,:,layer] * np.exp(np.sqrt(-1+0j)*(Phase[:,:,layer])))
    imScrambled = imScrambled.real # get rid of imaginery part in image (due to rounding error)


    for layer in range(imSize[2]):
        imFourier_depth[:,:,layer] = np.fft.fft2(depth[:,:,layer])         # Fast-Fourier transform
        Amp_depth[:,:,layer] = abs(imFourier_depth[:,:,layer])         # amplitude spectrum
        Phase_depth[:,:,layer] = np.angle(imFourier_depth[:,:,layer])     # phase spectrum
        Phase_depth[:,:,layer] = Phase_depth[:,:,layer] + RandomPhase  # add random phase to original phase
        # combine Amp and Phase then perform inverse Fourier
        imScrambled_depth[:,:,layer] = np.fft.ifft2(Amp_depth[:,:,layer] * np.exp(np.sqrt(-1+0j)*(Phase_depth[:,:,layer])))
    imScrambled_depth = imScrambled_depth.real # get rid of imaginery part in image (due to rounding error)


    rgb_saved_path = saving_root + 'nyu_images/' + rgb_path.split('/')[-1].split('.')[0] + '.jpg'
    depth_saved_path = saving_root + 'nyu_depths/' +  depth_path.split('/')[-1].split('.')[0] + '.png'
    cv2.imwrite(rgb_saved_path, imScrambled.astype(np.float32))
    # cv2.imwrite(depth_saved_path, imScrambled_depth.astype(np.float32)[:, :, 0])
    cv2.imwrite(depth_saved_path, imScrambled_depth[:, :, 0].astype(np.float32))