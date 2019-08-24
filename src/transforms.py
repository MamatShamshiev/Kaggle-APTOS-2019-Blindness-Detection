from typing import List, Dict
import random
import numpy as np
import cv2

import albumentations as albu
from albumentations import (
    Compose, RandomRotate90, HorizontalFlip, VerticalFlip, Rotate,
    Normalize, RandomScale, CenterCrop, RandomGamma,
    IAAPerspective, JpegCompression, ToGray, ChannelShuffle, RGBShift, CLAHE,
    RandomBrightnessContrast, RandomSunFlare, Cutout, OneOf, Resize
)
from albumentations.torch import ToTensor

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def pre_transforms(image_size=224, crop_from_gray=False, circle_crop=False, ben_preprocess=10,
                   random_scale=0.3, random_scale_p=0.75, brightness=0.2, contrast=0.2, color_p=0.5):
    transforms = [Resize(image_size, image_size)]
    if crop_from_gray is True:
        transforms = [Crop_From_Gray()] + transforms
    if (brightness > 0) or (contrast > 0):
        transforms.append(RandomBrightnessContrast(brightness_limit=brightness, contrast_limit=contrast, p=color_p))
    if random_scale > 0:
        transforms.append(RandomScale((0.0, random_scale), p=random_scale_p))
        transforms.append(CenterCrop(image_size, image_size))
    if ben_preprocess > 0:
        transforms.append(Ben_preprocess(ben_preprocess))
    if circle_crop is True:
        transforms.append(Circle_Crop())
    return Compose(transforms)

def my_transforms(hor_flip=0.5, ver_flip=0.2, rotate=120):
    transforms = []
    if hor_flip > 0:
        transforms.append(HorizontalFlip(p=hor_flip))
    if ver_flip > 0:
        transforms.append(VerticalFlip(p=ver_flip))
    if rotate > 0:
        transforms.append(Rotate(limit=rotate, p=1))
    return Compose(transforms)

def post_transforms(normalize=True):
    transforms = [ToTensor()]
    if normalize is True:
        transforms = [Normalize()] + transforms
    return Compose(transforms)

class Ben_preprocess:
    def __init__(self, sigma=10):
        self.sigma = sigma

    def __call__(self, image, force_apply=None):
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0) , self.sigma), -4, 128)
        return {"image": image}

class Crop_From_Gray:
    def __init__(self, tol=7):
        self.tol = tol
        
    def _crop_image_from_gray(self, img):
        if img.ndim == 2:
            mask = img > self.tol
            return img[np.ix_(mask.any(1),mask.any(0))]
        elif img.ndim == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = gray_img>self.tol

            check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
            if (check_shape == 0): # image is too dark so that we crop out everything,
                return img # return original image
            else:
                img1 = img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
                img2 = img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
                img3 = img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
                img = np.stack([img1,img2,img3],axis=-1)
            return img
        
    def __call__(self, image, force_apply=None):
        img = self._crop_image_from_gray(image)
        return {"image": img}

class Circle_Crop:
    def _circle_crop(self, img):
        
        height, width, depth = img.shape
        x = int(width/2)
        y = int(height/2)
        r = np.amin((x,y))

        circle_img = np.zeros((height, width), np.uint8)
        cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
        img = cv2.bitwise_and(img, img, mask=circle_img)
        return img 
        
    def __call__(self, image, force_apply=None):
        img = self._circle_crop(image)
        return {"image": img}

class DictTransformCompose:
    def __init__(self, dict_transforms: List):
        self.dict_transforms = dict_transforms

    def __call__(self, dct):
        for transform in self.dict_transforms:
            dct = transform(dct)
        return dct


def hard_transform(image_size=224, p=0.5):
    transforms = [
        Cutout(
            num_holes=4,
            max_w_size=image_size // 4,
            max_h_size=image_size // 4,
            p=p
        ),
        ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT,
            p=p
        ),
        IAAPerspective(scale=(0.02, 0.05), p=p),
        OneOf(
            [
                HueSaturationValue(p=p),
                ToGray(p=p),
                RGBShift(p=p),
                ChannelShuffle(p=p),
            ]
        ),
        RandomBrightnessContrast(
            brightness_limit=0.5, contrast_limit=0.5, p=p
        ),
        RandomGamma(p=p),
        CLAHE(p=p),
        JpegCompression(quality_lower=50, p=p),
    ]
    transforms = Compose(transforms)
    return transforms