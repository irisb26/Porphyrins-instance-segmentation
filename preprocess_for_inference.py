import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
from pathlib import Path
from csbdeep.utils import normalize

#get original stack (padded by pyfast) and magnification from h5 file

def trim_image(img):
    rows_uniform = np.all(img == img[:, [0]], axis=1)
    cols_uniform = np.all(img == img[[0], :], axis=0)

    top = 0
    while top < img.shape[0] and rows_uniform[top]:
        top += 1
    bottom = img.shape[0]
    while bottom > top and rows_uniform[bottom - 1]:
        bottom -= 1
    left = 0
    while left < img.shape[1] and cols_uniform[left]:
        left += 1
    right = img.shape[1]
    while right > left and cols_uniform[right - 1]:
        right -= 1
    trimmed=img[top:bottom, left:right]
    offset=(top,left)
    return trimmed,offset

TARGET_SIZE=16
OBJECT_SIZE=1.5 
def get_scaling_factor(mag):
    return (TARGET_SIZE * mag) / OBJECT_SIZE

def scale_image(img,scaling_factor): #changes dimensions of image
    if scaling_factor<1: #downsampling, need to have anti-aliasing (gaussian pre-filter), use linear interpolation
        scaled_image=ski.transform.rescale(img,scaling_factor,order=1,anti_aliasing=True,preserve_range=True)
    else: #upsampling 
        scaled_image = ski.transform.rescale(img,scaling_factor,order=3,anti_aliasing=False,preserve_range=True)
    return scaled_image

#trained on padded images so now need to pad to (256,256)
def pad(img):
    h, w = img.shape[:2]
    pad_h = max(0, 264 - h)
    pad_w = max(0, 264 - w)
    bg_val = np.median(img)  # Match substrate background intensity
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=bg_val)
    return img_padded


def apply_to_stack(stack,magnification):
    ready=[]
    offsets=[]
    for img in stack:
        trimmed,offset=trim_image(img)
        offsets.append(offset)

        factor=get_scaling_factor(magnification)
        scaled=scale_image(trimmed,factor)

        #normalize before padding (affects bg intensity)
        normed = normalize(scaled)
        #padded=pad(normed)

        ready.append(normed)
    return ready,offsets


