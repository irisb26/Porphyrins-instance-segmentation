import numpy as np
from csbdeep.utils import normalize
import skimage as ski

#before trying out stardist labels on it, trim and normalize consistently

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


def select_some_frames(stack):
    #get one 20% through, the middle on and another 80% through the stack
    n=len(stack)
    start_idx = int(n * 0.2)
    middle_idx = n // 2
    end_idx = int(n * 0.8)

    return [stack[start_idx],stack[middle_idx],stack[end_idx]]

def apply_to_stack(stack):
    selected=select_some_frames(stack) #list
    offsets=[]
    ready=[]
    for img in selected:
        trimmed,offset=trim_image(img)
        offsets.append(offset)

        ready.append(trimmed)
    return ready,offsets

OBJECT_SIZE=1.5
def get_scaling_factor(mag,target_size):
    return (target_size * mag) / OBJECT_SIZE

def prepare_image_and_label(img,labels,mag,target_size):
    """ 
    Scale image and label in parallel (must have same dimensions!) and norm the stack images
    """
    scaling_factor=get_scaling_factor(mag,target_size)
    if img.shape != labels.shape:
        raise ValueError(
            f"Image spatial shape {img.shape} and label shape "
            f"{labels.shape} must match."
        )
    #for the image, cubic interpolation (could also have 1 for bilinear)
    if scaling_factor<1: #downsampling, need to have anti-aliasing (gaussian pre-filter), use linear interpolation
        #print(f"down sampled by {scaling_factor}")
        scaled_image=ski.transform.rescale(img,scaling_factor,order=1,anti_aliasing=True,preserve_range=True)
    else: #upsampling 
        #print(f"up sampled by {scaling_factor}")
        scaled_image = ski.transform.rescale(img,scaling_factor,order=3,anti_aliasing=False,preserve_range=True)
        
    #for labels, nearest neighbour interpolation, order MUST be 0, resize to ensure has same shape as rescaled image
    scaled_labels = ski.transform.resize(labels,output_shape=scaled_image.shape,order=0,anti_aliasing=False,preserve_range=True).astype(labels.dtype) 
    normed_image=normalize(scaled_image)
    return normed_image, scaled_labels    



