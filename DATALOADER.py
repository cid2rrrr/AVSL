# 10 sec clips are saved in a folder
# get the central(random) frame and audio wav and pass it to the model

# central frame need to be a dict = {} with numpy array(HWC) or tensor(CHW)

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from moviepy.editor import VideoFileClip
from torch.nn import functional as F
import os
import sys
import random
from PIL import Image

from detectron2.config import configurable
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
# from detectron2.data import detection_utils as utils

class VideoDataset(Dataset):
    @configurable
    def __init__(self, training=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
        folder_path,):

        self.folder_path = folder_path
        self.video_list = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
        self.tfm_gens = augmentations

        self.training = training
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_path = os.path.join(self.folder_path, self.video_list[idx])

        # Load video and extract central frame
        video_clip = VideoFileClip(video_path)
        #central_frame = video_clip.get_frame(video_clip.duration / 2)
        #central_frame = np.array(central_frame)

        # Change to random frame, loading randomly changes the frame every batch
        # Generate random timestamp
        random_timestamp = random.uniform(0, video_clip.duration)  
        random_frame = video_clip.get_frame(random_timestamp)
        random_frame = np.array(random_frame)
        
        # Convert image frame to numpy array and normalize
        #central_frame = central_frame / 255.0

        # Additional augmentation
        aug_input = T.AugInput(random_frame)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        random_frame = aug_input.image
        #sem_seg_gt = aug_input.sem_seg

        random_frame = torch.as_tensor(np.ascontiguousarray(random_frame.transpose(2, 0, 1)))
        #central_frame = np.transpose(central_frame, (2, 0, 1))  # Change HWC to CHW
        
        if self.size_divisibility > 0:
            random_frame_size = (random_frame.shape[-2], random_frame.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - random_frame_size[1],
                0,
                self.size_divisibility - random_frame_size[0],
            ]
            random_frame = F.pad(random_frame, padding_size, value=128).contiguous()
        
        # Save audio as WAV file
        audio_path = f"{video_path[:-4]}.wav"
        self.blockPrint()
        video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le', fps=44100)
        self.enablePrint()

        # Return data as dictionary
        item = {'image': random_frame, 'mixed_audio_path': audio_path}
        
        return item
    
    @classmethod    
    def from_config(cls, cfg, training=True):
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())
        augs.extend([
        T.ResizeScale(
            min_scale=cfg.INPUT.MIN_SCALE, max_scale=cfg.INPUT.MAX_SCALE, 
            target_height=cfg.INPUT.IMAGE_SIZE, target_width=cfg.INPUT.IMAGE_SIZE
        ),
        T.FixedSizeCrop(crop_size=(cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE)),
        ])

        ignore_label = False

        ret = {
            "training": training,
            "augmentations": augs,  
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }
        return ret      
    
    def blockPrint(self):
        global backupstdout
        backupstdout=sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def enablePrint(self):
        global backupstdout
        sys.stdout = backupstdout

# Example usage:
def VideoDataLoader(cfg, folder_path, batch_size):
    """ 
    folder_path = 'path/to/your/videos' 
    batch_size = cfg.SOLVER.IMS_PER_BATCH (train)
    """
    video_dataset = VideoDataset(cfg, folder_path=folder_path)
    print(f"{len(video_dataset)} of train videos have loaded")
    dataloader = DataLoader(video_dataset, batch_size, shuffle=True)
    return dataloader

# Iterate through the dataloader
def iteration(dataloader):
    for batch in dataloader:
        image = batch['image']
        audio_path = batch['audio_path']
        # Perform further processing or analysis with the image and audio
        return image, audio_path
    

class ImageDataset(Dataset):
    @configurable
    def __init__(self, training=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
        folder_path,):

        self.folder_path = folder_path
        self.image_list = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
        self.tfm_gens = augmentations

        self.training = training
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.folder_path, self.image_list[idx])
        random_frame = Image.open(image_path)
        random_frame = np.array(random_frame)
        
        # Convert image frame to numpy array and normalize
        #central_frame = central_frame / 255.0

        # Additional augmentation
        aug_input = T.AugInput(random_frame)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        random_frame = aug_input.image
        #sem_seg_gt = aug_input.sem_seg

        random_frame = torch.as_tensor(np.ascontiguousarray(random_frame.transpose(2, 0, 1)))
        #central_frame = np.transpose(central_frame, (2, 0, 1))  # Change HWC to CHW
        
        if self.size_divisibility > 0:
            random_frame_size = (random_frame.shape[-2], random_frame.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - random_frame_size[1],
                0,
                self.size_divisibility - random_frame_size[0],
            ]
            random_frame = F.pad(random_frame, padding_size, value=128).contiguous()

        # Return data as dictionary
        item = {'image': random_frame, 'image_name': self.image_list[idx]}
        
        return item
    
    @classmethod    
    def from_config(cls, cfg, training=True):
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())
        augs.extend([
        T.ResizeScale(
            min_scale=cfg.INPUT.MIN_SCALE, max_scale=cfg.INPUT.MAX_SCALE, 
            target_height=cfg.INPUT.IMAGE_SIZE, target_width=cfg.INPUT.IMAGE_SIZE
        ),
        T.FixedSizeCrop(crop_size=(cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE)),
        ])

        ignore_label = False

        ret = {
            "training": training,
            "augmentations": augs,  
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }
        return ret 
    
def ImageDataLoader(cfg, folder_path, batch_size):
    """ 
    folder_path = 'path/to/your/images' 
    batch_size = 1
    """
    image_dataset = ImageDataset(cfg, folder_path=folder_path)
    print(f"{len(image_dataset)} of train videos have loaded")
    dataloader = DataLoader(image_dataset, batch_size, shuffle=True)
    return dataloader