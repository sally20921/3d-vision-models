import cv2
import torch
import torch.nn.functional as funct
from functools import lru_cache
from PIL import Image

def load_image(path):
    '''
    read an image using PIL

    Parameters
    ____
    path: str
        path to the image

    Returns
    _____
    image: PIL.Image
    '''
    return Image.open(path)

def write_image(filename, image):
    '''
    write an numpy array image to file

    Parameters
    ______
    filename: str
        file where image where image will be saved
    image: np.array [H,W,3]
        RGB image

    ex. rgb = img
        cv2.imfunc(bgr)
        # converts to 8-bit unsigned (CV_8U) and saved 
        # only 8-bit single channel or 3-channel ('bgr') allowed
        
    ex. regular file is RGB image
        PIL reads the RGB image [W,H,C]
        cv2 reads the BGR image [H,W,C] to a numpy array
        # to view cv2 image, change BGR -> RGB
        numpy BGR image convention [H,W,C]
    ex. pytorch RGB image [B,C,H,W]

    note. if the pixel value is represented by 0.0 to 1.0,
    it is necesary to multiply by 255 and convert to uint8
    and then save
    '''
    cv2.imwrite(filename, image[:,:,::-1]) # rgb to bgr in channel dimension

def flip_lr(image):
    '''
    flip image horizontally

    Parameters
    _____
    image: torch.Tensor [B,3,H,W]
        image to be flipped 

    Returns
    _____
    image_flipped: torch.Tensor [B,3,H,W]
        flipped image
    '''
    assert image.dim() == 4, "Please provide [B,C,H,W] image to flip horizontally"
    return torch.flip(image, [3])
    # changed the ordering of the width 

def flip_model(model, image, flip):
    '''
    if flip is true:
        flip input image and then processes model(input) 
        then flip the output again => flip(model(flip(input)))
    else:
        just model(input)

    Parameters
    ______
    model: nn.Module
        nn.Module to be used 
    image: torch.Tensor [B,3,H,W]
        input image
    flip: bool
        true if the flip is happening
    
    Returns
    ______
    inv_depths: list of torch.Tensor [B,1,H,W]
        list of predicted inverse depth maps
    '''
    if flip:
        return [flip_lr(inv_depth) for inv_depth in model(flip_lr(image))]
    else:
        return model(image)




