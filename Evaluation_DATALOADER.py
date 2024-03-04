"""
Extract Dataset.tar.gz file

tar -xzvf Dataset.tar.gz

extracted in ./DATA/Dataset/.

N_piece_instrument Folder name

• AuMix_N_piece_instrument.wav Audio recording for the piece (mixed from individual recordings)
• AuSep_n_instrument_N_piece.wav Audio recording for track n
• F0s_n_instrument_N_ piece.txt Ground-truth pitch annotations on each frame for audio track n
• Notes_n_instrument_N_ piece.txt Ground-truth annotations on each played note for audio track n
• Sco_N_piece_instrument.mid MIDI score
• Sco_N_piece_instrument.pdf Sheet music
• Vid_N_piece_instrument.mp4 Video recording for the piece (assembled from individual 
recordings)
"""
import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import random
from moviepy.editor import VideoFileClip
# import torchvision.transforms.functional as F
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
# from detectron2.data import detection_utils as utils
# from get_path import get_file_paths

""" 
#We have a video, mixed_audio, separated_audios(2~5) for each 44 sample
# get_path.py
import os

def get_file_paths(song_names, root_video_dir):
    video_paths = []
    mixed_audio_paths = []
    separated_audio_paths = []

    for root, dirs, files in os.walk(root_video_dir):
        for dir_name in dirs:
            for song_name in song_names:
                if song_name in dir_name:
                    song_folder = os.path.join(root, dir_name)
                    for root, dirs, files in os.walk(song_folder):
                        for file in files:
                            file_path = os.path.join(root, file)
                            base_file_name, file_extension = os.path.splitext(file)
                            if file_extension.lower() == ".mp4":
                                video_paths.append(file_path)
                            elif "AuMix" in base_file_name:
                                mixed_audio_paths.append(file_path)
                            elif "AuSep" in base_file_name:
                                separated_audio_paths.append(file_path)
    
    return video_paths, mixed_audio_paths, separated_audio_paths

# Example usage:
#song_names = ["Jupiter", "Sonata"]
song_names = ["{:02d}".format(i) for i in range(1, 45)] # actual usage
root_video_dir = "/workspace/GitHub/AVSL/DATA/Dataset"

video_paths = []
mixed_audio_paths = []
separated_audio_paths = []
for song_name in song_names:
    video_paths_temp, mixed_audio_paths_temp, separated_audio_paths_temp = get_file_paths([song_name], root_video_dir)
    video_paths.extend(video_paths_temp)
    mixed_audio_paths.extend(mixed_audio_paths_temp)
    separated_audio_paths.extend([separated_audio_paths_temp])

print("Video Paths:")
print(video_paths)
print("\nMixed Audio Paths:")
print(mixed_audio_paths)
print("\nSeparated Audio Paths:")
print(separated_audio_paths)

# Results for #song_names = ["Jupiter", "Sonata"]
Video Paths:
['/workspace/GitHub/AVSL/DATA/Dataset/01_Jupiter_vn_vc/Vid_01_Jupiter_vn_vc.mp4', '/workspace/GitHub/AVSL/DATA/Dataset/02_Sonata_vn_vn/Vid_02_Sonata_vn_vn.mp4']
Mixed Audio Paths:
['/workspace/GitHub/AVSL/DATA/Dataset/01_Jupiter_vn_vc/AuMix_01_Jupiter_vn_vc.wav', '/workspace/GitHub/AVSL/DATA/Dataset/02_Sonata_vn_vn/AuMix_02_Sonata_vn_vn.wav']
Separated Audio Paths:
[['/workspace/GitHub/AVSL/DATA/Dataset/01_Jupiter_vn_vc/AuSep_1_vn_01_Jupiter.wav', '/workspace/GitHub/AVSL/DATA/Dataset/01_Jupiter_vn_vc/AuSep_2_vc_01_Jupiter.wav'], ['/workspace/GitHub/AVSL/DATA/Dataset/02_Sonata_vn_vn/AuSep_2_vn_02_Sonata.wav', '/workspace/GitHub/AVSL/DATA/Dataset/02_Sonata_vn_vn/AuSep_1_vn_02_Sonata.wav']]
"""
"""
instrument number

1-7 2

8-23 3

24-37 4

38-44 5
"""

class EvaluationDataset(Dataset):
    @configurable
    def __init__(self, training=False,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility, 
        video_paths, 
        mixed_audio_paths, 
        separated_audio_paths
        ):

        self.video_paths = video_paths
        self.mixed_audio_paths = mixed_audio_paths
        self.separated_audio_paths = separated_audio_paths

        self.tfm_gens = augmentations

        self.training = training
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        # single element
        video_path = self.video_paths[idx]
        # single element
        mixed_audio_path = self.mixed_audio_paths[idx]
        # a list of paths
        separated_audio_path = self.separated_audio_paths[idx]

        # Load video and extract central frame
        video_clip = VideoFileClip(video_path)

        # Change to random frame, loading randomly changes the frame every batch
        # Generate random timestamp
        random_timestamp = random.uniform(0, video_clip.duration)  
        random_frame = video_clip.get_frame(random_timestamp)
        random_frame = np.array(random_frame)
        
        # Convert image frame to numpy array and normalize
        aug_input = T.AugInput(random_frame)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        random_frame = aug_input.image

        random_frame = torch.as_tensor(np.ascontiguousarray(random_frame.transpose(2, 0, 1)))
        
        if self.size_divisibility > 0:
            random_frame_size = (random_frame.shape[-2], random_frame.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - random_frame_size[1],
                0,
                self.size_divisibility - random_frame_size[0],
            ]
            random_frame = F.pad(random_frame, padding_size, value=128).contiguous()

        #item = zip(video_path, mixed_audio_path, separated_audio_paths)

        # Return data as dictionary
        item = {'image': random_frame, 'mixed_audio_path': mixed_audio_path, 'separated_audio_paths': separated_audio_path}
        # item = {'image': random_frame, 'mixed_audio_path': mixed_audio_path}
        
        return item
    
    @classmethod    
    def from_config(cls, cfg, training=False):
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
    
# Example usage:
def EvaluationDataLoader(cfg, root_video_dir, batch_size):
    """ 
    We have a video, mixed_audio, separated_audios(2~5) for each 44 sample

    video_paths = ['path/to/videos/video1.mp4', 'path/to/videos/video2.mp4', ...]

    mixed_audio_path_list = ['path/to/mixed_audio/video1_mix.wav', 'path/to/mixed_audio/video2_mix.wav', ...]
    
    separated_audio_paths_list = [['path/to/separated_audio/video1_sep1.wav', 'path/to/separated_audio/video1_sep2.wav'],
                                  ['path/to/separated_audio/video2_sep1.wav', 'path/to/separated_audio/video2_sep2.wav', 'path/to/separated_audio/video2_sep3.wav'],
                                  ...
                                 ]

    batch_size = cfg.TEST.IMS_PER_BATCH
    root_video_dir = cfg.TEST.DATA_FOLDER
    """
    song_names = ["{:02d}".format(i) for i in range(1, 44)] # actual usage

    video_paths = []
    mixed_audio_paths = []
    separated_audio_paths = []
    for song_name in song_names:
        video_paths_temp, mixed_audio_paths_temp, separated_audio_paths_temp = get_file_paths([song_name], root_video_dir)
        video_paths.extend(video_paths_temp)
        mixed_audio_paths.extend(mixed_audio_paths_temp)
        separated_audio_paths.extend([separated_audio_paths_temp])

    evaluation_dataset = EvaluationDataset(cfg, video_paths=video_paths,
                                           mixed_audio_paths=mixed_audio_paths, 
                                           separated_audio_paths=separated_audio_paths)
    print(f"{len(evaluation_dataset)} of evaluation videos have loaded")
    dataloader = DataLoader(evaluation_dataset, batch_size=batch_size, shuffle=False)
    return dataloader

# Iterate through the dataloader
def evaluation_iteration(dataloader):
    for batch in dataloader:
        image = batch['image']
        mixed_audio_path = batch['mixed_audio_path']
        separated_audio_paths = batch['separated_audio_paths']
        return image, mixed_audio_path, separated_audio_paths
    




def get_file_paths(song_names, root_video_dir):
    video_paths = []
    mixed_audio_paths = []
    separated_audio_paths = []

    for root, dirs, files in os.walk(root_video_dir):
        for dir_name in dirs:
            for song_name in song_names:
                if song_name in dir_name:
                    song_folder = os.path.join(root, dir_name)
                    for root, dirs, files in os.walk(song_folder):
                        for file in files:
                            file_path = os.path.join(root, file)
                            base_file_name, file_extension = os.path.splitext(file)
                            if file_extension.lower() == ".mp4":
                                video_paths.append(file_path)
                            elif "AuMix" in base_file_name:
                                mixed_audio_paths.append(file_path)
                            elif "AuSep" in base_file_name:
                                separated_audio_paths.append(file_path)
    
    return video_paths, mixed_audio_paths, separated_audio_paths