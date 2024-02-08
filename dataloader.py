# 10 sec clips are save in a folder
# get the central frame and audio wav and pass it to the model

# central frame need to be a dict = {} with numpy array(HWC) or tensor(CHW)

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from moviepy.editor import VideoFileClip
from torch.nn import functional as F
import os
from detectron2.config import configurable
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.data import detection_utils as utils

class VideoDataset(Dataset):
    @configurable
    def __init__(self, is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility, folder_path,):

        self.folder_path = folder_path
        self.video_list = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
        self.tfm_gens = augmentations

        self.is_train = is_train
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_path = os.path.join(self.folder_path, self.video_list[idx])

        # Load video and extract central frame
        video_clip = VideoFileClip(video_path)
        central_frame = video_clip.get_frame(video_clip.duration / 2)
        central_frame = np.array(central_frame)
        # Convert image frame to numpy array and normalize
        #central_frame = central_frame / 255.0

        # Additional augmentation
        aug_input = T.AugInput(central_frame)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        central_frame = aug_input.image
        #sem_seg_gt = aug_input.sem_seg

        central_frame = torch.as_tensor(np.ascontiguousarray(central_frame.transpose(2, 0, 1)))
        #central_frame = np.transpose(central_frame, (2, 0, 1))  # Change HWC to CHW
        
        if self.size_divisibility > 0:
            central_frame_size = (central_frame.shape[-2], central_frame.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - central_frame_size[1],
                0,
                self.size_divisibility - central_frame_size[0],
            ]
            central_frame = F.pad(central_frame, padding_size, value=128).contiguous()
        
        sample = {}
        sample['image'] = central_frame
        
        # Save audio as WAV file
        audio_path = f"{video_path[:-4]}.wav"
        video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le', fps=44100)

        # Return data as dictionary
        #sample = {'image': torch.FloatTensor(central_frame), 'audio_path': audio_path}
        
        return sample
    
    @classmethod    
    def from_config(cls, cfg, is_train=True):
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
            "is_train": is_train,
            "augmentations": augs,  
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }
        return ret      


# Example usage:
folder_path = 'path/to/your/videos'
video_dataset = VideoDataset(folder_path=folder_path)
dataloader = DataLoader(video_dataset, batch_size=1, shuffle=True)

# Iterate through the dataloader
for batch in dataloader:
    image = batch['image']
    audio_path = batch['audio_path']
    # Perform further processing or analysis with the image and audio