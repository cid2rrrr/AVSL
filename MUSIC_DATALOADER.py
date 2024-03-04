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
"""
JSON file format: youtube_id + frame, class + bbox
{"file": "WkY638E5wy8_000_00000", "class_src1": "acoustic_guitar", "bbox_src1": [0.290625, 0.38333333333333336, 0.1625, 0.32222222222222224], "class_src2": "saxophone", "bbox_src2": [0.6203125, 0.425, 0.1328125, 0.2916666666666667]}, 
{"file": "WkY638E5wy8_001_00037", "class_src1": "acoustic_guitar", "bbox_src1": [0.2546875, 0.4111111111111111, 0.23125, 0.31666666666666665], "class_src2": "saxophone", "bbox_src2": [0.6109375, 0.48055555555555557, 0.1359375, 0.24722222222222223]}, 

{"file": "awCkfdohkzE_000_00000", "class_src1": "trumpet", "bbox_src1": [0.55, 0.26666666666666666, 0.21041666666666667, 0.41388888888888886], "class_src2": "flute", "bbox_src2": [0.025, 0.36666666666666664, 0.46458333333333335, 0.07222222222222222]}, 
{"file": "awCkfdohkzE_000_00007", "class_src1": "trumpet", "bbox_src1": [0.5583333333333333, 0.26944444444444443, 0.20833333333333334, 0.4], "class_src2": "flute", "bbox_src2": [0.09375, 0.37222222222222223, 0.3854166666666667, 0.08055555555555556]}, 
{"file": "awCkfdohkzE_000_00014", "class_src1": "trumpet", "bbox_src1": [0.55, 0.26944444444444443, 0.21458333333333332, 0.4], "class_src2": "flute", "bbox_src2": [0.21666666666666667, 0.36944444444444446, 0.24791666666666667, 0.10277777777777777]}, 
{"file": "awCkfdohkzE_000_00021", "class_src1": "trumpet", "bbox_src1": [0.5395833333333333, 0.26944444444444443, 0.22708333333333333, 0.425], "class_src2": "flute", "bbox_src2": [0.20833333333333334, 0.37222222222222223, 0.25416666666666665, 0.1]}, 

{"file": "vEOyhF8KLPE_001_00037", "class_src1": "acoustic_guitar", "bbox_src1": [0.5359375, 0.3416666666666667, 0.3890625, 0.26666666666666666], "class_src2": "violin", "bbox_src2": [0.3359375, 0.07777777777777778, 0.3078125, 0.44166666666666665]}, 

{"file": "ywH4K8mQ-1s_000_00000", "class_src1": "acoustic_guitar", "bbox_src1": [0.040625, 0.44166666666666665, 0.559375, 0.5277777777777778], "class_src2": "saxophone", "bbox_src2": [0.6875, 0.2722222222222222, 0.1421875, 0.7083333333333334]}]

return : image, audio path, bounding box
"""
import json
import numpy as np

def extract_gt_bboxes_from_json(json_file_path):
    """
    Extracts ground truth bounding boxes from a JSON file containing annotations.

    Args:
    - json_file_path (str): Path to the JSON file containing annotations.

    Returns:
    - dict: A dictionary where keys are file names and values are lists of bounding boxes and corresponding classes.
    """
    gt_bboxes = {}
    with open(json_file_path) as json_file:
        annotations = json.load(json_file)
        for annotation in annotations:
            if "duet" in json_file_path:
                bboxes_src = [annotation['bbox_src1'], annotation['bbox_src2']]
                classes_src = [annotation['class_src1'], annotation['class_src2']]
                bboxes = [(np.clip(np.array(bbox), 0, 1) * 224).astype(int) for bbox in bboxes_src]
                
                bb = -torch.ones((10, 4)).long()
                bb[:len(bboxes)] = torch.from_numpy(np.array([bboxes]))
                gt_map = bbox2gtmap([bboxes[0]], "music_duet")
                gt_map_ss = bbox2gtmap([bboxes[1]], "music_duet")

                anno = {"bboxes": bb, "gt_map": np.stack((gt_map, gt_map_ss),axis=0), "gt_mask": 1}
                # gt_bboxes[annotation['file']] = [bboxes, classes_src, np.stack((gt_map, gt_map_ss),axis=0), 1]
                gt_bboxes[annotation['file']] = [anno, classes_src]
            
            else:
                bboxes_src = [annotation["bbox"]]
                classes_src = [annotation["class"]]
                bboxes = (np.clip(np.array(bboxes_src), 0, 1) * 224).astype(int)
                
                bb = -torch.ones((10, 4)).long()
                bb[:len(bboxes)] = torch.from_numpy(np.array([bboxes]))
                gt_map = bbox2gtmap([bboxes[0]], "music_solo")
                
                anno = {"bboxes": bb, "gt_map": gt_map, "gt_mask": 1}
                # gt_bboxes[annotation["file"]] = [bboxes, classes_src, gt_map, 1]
                gt_bboxes[annotation["file"]] = [anno, classes_src]

    return gt_bboxes


def bbox2gtmap(bboxes, format='flickr'):
    gt_map = np.zeros([224, 224])
    # for coord in bboxes:
    #     xmin, ymin, xmax, ymax = coord
    #     temp = np.zeros([224, 224])
    #     temp[ymin:ymax, xmin:xmax] = 1
    #     gt_map += temp
    for xmin, ymin, w, h in bboxes:
        temp = np.zeros([224, 224])
        temp[ymin:ymin+h, xmin:xmin+w] = 1
        gt_map += temp

    if format == 'flickr':
        # Annotation consensus
        gt_map = gt_map / 2
        gt_map[gt_map > 1] = 1

    elif format in {'vggss', 'music_duet'}:
        # Single annotation
        gt_map[gt_map > 0] = 1

    return gt_map

# # Example usage:
# json_file_path = '/workspace/GitHub/AVSL/DATA/metadata/music_duet.json'
# gt_bboxes = extract_gt_bboxes_from_json(json_file_path)
# print(gt_bboxes)

import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

class MUSICDataset(Dataset):
    """
    {'WkY638E5wy8_000_00000': [[array([65, 85, 36, 72]),
       array([138,  95,  29,  65])],
      ['acoustic_guitar', 'saxophone']],
     'WkY638E5wy8_001_00037': [[array([57, 92, 51, 70]),
       array([136, 107,  30,  55])],
      ['acoustic_guitar', 'saxophone']],
    ...
    """
    def __init__(self, root_frame_dir,root_audio_dir, gt_bboxes, transform=None):
        self.root_frame_dir = root_frame_dir
        self.root_audio_dir = root_audio_dir
        self.gt_bboxes = gt_bboxes
        self.transform = transform
    
    def __len__(self):
        return len(self.gt_bboxes)
    
    def __getitem__(self, idx):
        file_name = list(self.gt_bboxes.keys())[idx]
        image_path = os.path.join(self.root_frame_dir, f'{file_name}.jpg')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        # For input images, the resolution is re￾sized to 224 × 224
        image = cv2.resize(image, (224, 224))
        audio_path = os.path.join(self.root_audio_dir, f'{file_name}.wav')
        
        # Extract bounding boxes and classes
        # bounding_boxes, classes = self.gt_bboxes[file_name]
        anno, classes = self.gt_bboxes[file_name]

        # calculate gt_map
        # anno = {}
        # bboxes = self.gt_bboxes[file_name][0]
        # bb = -torch.ones((10, 4)).long()
        # bb[:len(bboxes)] = torch.from_numpy(np.array([bboxes]))
        # anno['bboxes'] = bb
        # gt_map = bbox2gtmap([bboxes[0]], "music_duet")
        # gt_map_ss = bbox2gtmap([bboxes[1]], "music_duet")
        # anno['gt_map'] = np.stack((gt_map, gt_map_ss),axis=0)
        # anno['gt_mask'] = 1 
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        
        item = {'image': image, 'mixed_audio_path': audio_path, 'bboxes': anno, 'classes': classes}
        return item

def MUSICDataLoader(root_frame_dir, root_audio_dir, json_file_path, batch_size):
    """
    root_frame_dir = 'DATA/MUSIC_DUET/frames/'
    root_audio_dir = 'DATA/MUSIC_DUET/audios'
    json_file_path = "DATA/metadata/music_duet.json
    batch_size = cfg.TEST.IMS_PER_BATCH
    """

    gt_bboxes = extract_gt_bboxes_from_json(json_file_path) # this function is in MUSIC_DATALOADER.py
    transform = ToTensor()  # Convert images to PyTorch tensors

    music_dataset = MUSICDataset(root_frame_dir, root_audio_dir, gt_bboxes, transform=transform)
    music_dataloader = DataLoader(music_dataset, batch_size=batch_size, shuffle=True)

    return music_dataloader


class SOLODataset(Dataset):
    """
    not complete
    """
    def __init__(self, root_frame_dir,root_audio_dir, transform=None):
        self.root_frame_dir = root_frame_dir
        self.root_audio_dir = root_audio_dir
        self.transform = transform
    
    def __len__(self):
        return
    
    def __getitem__(self, idx):
        file_name = list(self.gt_bboxes.keys())[idx]
        image_path = os.path.join(self.root_frame_dir, f'{file_name}.jpg')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        # For input images, the resolution is re￾sized to 224 × 224
        image = cv2.resize(image, (224, 224))
        audio_path = os.path.join(self.root_audio_dir, f'{file_name}.wav')
        
        # Extract bounding boxes and classes
        bounding_boxes, classes = self.gt_bboxes[file_name]
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        
        item = {'image': image, 'mixed_audio_path': audio_path, 'bboxes': bounding_boxes, 'classes': classes}
        return item


