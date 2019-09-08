import json
import collections
import os
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from catalyst.data.augmentor import Augmentor
from catalyst.data.dataset import ListDataset
from catalyst.data.reader import ScalarReader, ReaderCompose, ImageReader
from catalyst.data.sampler import BalanceClassSampler
from catalyst.dl import ConfigExperiment
from catalyst.utils.pandas import read_csv_data

from transforms import (pre_transforms, post_transforms, my_transforms,
                        DictTransformCompose, Compose
                       )


class RetinopathyDatasetTrain(Dataset):
    def __init__(self, csv_file, path_to_data, transform):
        self.data = csv_file
        self.path_to_data = path_to_data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.path_to_data, self.data.loc[idx, 'id_code'])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)
        
        target = self.data.loc[idx, 'diagnosis']
        return {'image': image['image'], 'targets': target}

class Experiment(ConfigExperiment):
    
    def _postprocess_model_for_stage(self, stage: str, model: nn.Module):
        model_ = model
        if isinstance(model, torch.nn.DataParallel):
            model_ = model_.module

        if stage in ["stage1"]:
            for param in model_.model.parameters():
                param.requires_grad = False
            fc = model_.model._fc
            for param in fc.parameters():
                param.requires_grad = True
            print('Learnable parameters: ', sum(p.numel() for p in model_.model.parameters() if p.requires_grad))
        if stage in ["stage2"]:
            for param in model_.model.parameters():
                param.requires_grad = True
            print('Learnable parameters: ', sum(p.numel() for p in model_.model.parameters() if p.requires_grad))
        return model_
    
    @staticmethod
    def get_transforms(
        stage: str = None,
        mode: str = None,
        one_hot_classes=None,
        image_size: int = 224,
        crop_from_gray: bool = False,
        circle_crop: bool = False,
        normalize: bool = True,
        ben_preprocess: int = 10,
        hor_flip: float = 0.5,
        ver_flip: float = 0.2,
        rotate: int = 120,
        random_scale: float = 0.3, 
        random_scale_p: float = 0.75,
        brightness: float = 0.2, 
        contrast: float = 0.2,
        color_p: float = 0.5,              
    ):

        print("Transforms params:")
        print("Image size:", image_size)
        print("crop_from_gray:", crop_from_gray)
        print("circle_crop:", circle_crop)
        print("normalize:", normalize)
        print("ben_preprocess:", ben_preprocess)
        print("hor_flip:", hor_flip)
        print("ver_flip:", ver_flip)
        print("rotate:", rotate)
        print("random_scale:", random_scale)     
        print("random_scale_p:", random_scale_p)
        print("brightness:", brightness)     
        print("contrast:", contrast)
        print("color_p:", color_p)       

        pre_transform_fn = pre_transforms(image_size=image_size, 
                                          crop_from_gray=crop_from_gray, 
                                          circle_crop=circle_crop,
                                          ben_preprocess=ben_preprocess,
                                          random_scale=random_scale, 
                                          random_scale_p=random_scale_p,
                                          brightness=brightness,
                                          contrast=contrast,
                                          color_p=color_p
                                          )
        pre_transform_fn_val = pre_transforms(image_size=image_size, 
                                              crop_from_gray=crop_from_gray, 
                                              circle_crop=circle_crop,
                                              ben_preprocess=ben_preprocess,
                                              random_scale=0.0, 
                                              random_scale_p=0.0,
                                              brightness=0,
                                              contrast=0,
                                              )        
        my_transform_fn = my_transforms(hor_flip=hor_flip,
                                        ver_flip=ver_flip,
                                        rotate=rotate)
        post_transform_fn = post_transforms(normalize=normalize)

        if mode in ["train"]:
            result = Compose([pre_transform_fn, my_transform_fn, post_transform_fn])
        elif mode in ["valid", "infer"]:
            result = Compose([pre_transform_fn_val, post_transform_fn])
        else:
            raise NotImplementedError()

        return result

    def get_datasets(
        self,
        stage: str,
        datapath: str = None,
        in_csv: str = None,
        in_csv_train: str = None,
        in_csv_valid: str = None,
        in_csv_infer: str = None,
        train_folds: str = None,
        valid_folds: str = None,
        tag2class: str = None,
        class_column: str = "class",
        input_column: str = "filepath",
        tag_column: str = None,
        folds_seed: int = 42,
        n_folds: int = 5,
        one_hot_classes: int = None,
        upsampling: bool = False,
        image_size: int = 224,
        crop_from_gray: bool = False,
        circle_crop: bool = False,
        normalize: bool = True,
        ben_preprocess: int = 10,
        hor_flip: float = 0.5,
        ver_flip: float = 0.2,
        rotate: int = 120,
        random_scale: int = 0.3, 
        random_scale_p: int = 0.75,
        brightness: float = 0.2, 
        contrast: float = 0.2,
        color_p: float = 0.5,              
    ):
        datasets = collections.OrderedDict()
        tag2class = json.load(open(tag2class)) if tag2class is not None else None

        df, df_train, df_valid, df_infer = read_csv_data(
            in_csv=in_csv,
            in_csv_train=in_csv_train,
            in_csv_valid=in_csv_valid,
            in_csv_infer=in_csv_infer,
            train_folds=train_folds,
            valid_folds=valid_folds,
            tag2class=tag2class,
            class_column=class_column,
            tag_column=tag_column,
            seed=folds_seed,
            n_folds=n_folds
        )

        for source, mode in zip(
            (df_train, df_valid, df_infer), ("train", "valid", "infer")
        ):
            if len(source) > 0:
                transforms = self.get_transforms(
                    stage=stage,
                    mode=mode,
                    image_size=image_size,
                    one_hot_classes=one_hot_classes,
                    crop_from_gray=crop_from_gray,
                    circle_crop=circle_crop,
                    normalize=normalize,
                    ben_preprocess=ben_preprocess,
                    hor_flip=hor_flip,
                    ver_flip=ver_flip,
                    rotate=rotate,
                    random_scale=random_scale, 
                    random_scale_p=random_scale_p,
                    brightness=brightness, 
                    contrast=contrast,
                    color_p=color_p,                  
                )
                if mode == "valid":
                    dataset = RetinopathyDatasetTrain(pd.DataFrame(source), "./data/train_images", transforms)
                else:
                    dataset = RetinopathyDatasetTrain(pd.DataFrame(source), datapath, transforms)
                if upsampling is True and mode == "train":
                    labels = [x[class_column] for x in source]
                    sampler = BalanceClassSampler(labels, mode="upsampling")
                    dataset = {"dataset": dataset, "sampler": sampler}
                datasets[mode] = dataset

        return datasets