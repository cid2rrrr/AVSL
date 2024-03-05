# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
import librosa
import numpy as np

from detectron2.config import configurable
# from detectron2.data import MetadataCatalog
# from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
# from detectron2.modeling.backbone import Backbone
# from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.modeling import build_backbone

from .build import build_swin_backbone, build_pixel_decoder, build_mask_predictor, build_mask_predictor_light
from .criterion_custom import SetCriterion
from .utils import AverageMeter
# from criterion_custom import SetCriterion

class AVSLModelLight(nn.Module):
    """
    1. data is stored as 10 sec video clip
    2. dataloader.py gets the 10 sec clip(or pre-extracted audio) and returns 10 sec audio.wav & random frame
    3. image goes to the backbone
    4. audio goes to the separation model
    5. AVSLModel returns the parameters needed for the loss in the output dictionary
    Args:
        backbone: imported swin backbone
        pixel_decoder: imported pixel decoder
        mask_predictor: imported mask predictor
            - include transformer_decoder, audio_separation_model, audio_encoder, predictor
        num_tokens: number of object queries
        criterion: losses
        config: weights, parameters
    """
    
    @configurable
    def __init__(
        self,
        training: bool,
        *,
        backbone: nn.Module,
        pixel_decoder: nn.Module,
        mask_predictor: nn.Module,
        criterion,
        num_tokens: int,
        # panoptic_on: bool,
        # object_mask_threshold: float,
        # overlap_threshold: float,
        # metadata,
        size_divisibility: int,
        # sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        device: str,
        transformer_in_feature: str,
        return_dict=True
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            pixel_decoder: a module that predicts image features and per pixel features from backbone features and
            mask_predictor: a module that predicts audio features with audio module,
                                        predicts semantic segmentation from per pixel features and audio features
            criterion: a module that defines the loss
            num_tokens: int, number of tokens
            panoptic_on: bool, whether to output panoptic segmentation prediction
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__()
        
        self.device = device
        
        self.backbone = backbone
        self.backbone.to(torch.device(self.device))
        for param in self.backbone.parameters():
            param.requires_grad = False

        # self.pixel_decoder = pixel_decoder
        # pixel_decoder.to(torch.device(self.device))

        self.mask_predictor = mask_predictor
        self.mask_predictor.to(torch.device(self.device))
        
        self.training = training
        self.criterion = criterion
        self.num_tokens = num_tokens
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.return_dict = return_dict

        self.transformer_in_feature = transformer_in_feature


    @classmethod
    def from_config(cls, cfg):
        ###
        if cfg.MODEL.BACKBONE.NAME == "D2SwinTransformer":
            backbone = build_swin_backbone(cfg, cfg.MODEL.PIXEL_MEAN, cfg.MODEL.WEIGHTS)
        else:
            backbone = build_backbone(cfg) # ResNet50
        pixel_decoder = build_pixel_decoder(cfg, backbone.output_shape())
        mask_predictor = build_mask_predictor_light(cfg, backbone.output_shape()[cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE].channels)
        # mask_predictor = build_mask_predictor_light(cfg,cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM)
        ###

        weight_dict = {
                    "AS_loss":1e-2, 
                    "LS_loss":1, 
                    "AD_loss":1, 
                    "DF_loss":1, 
                    "DT_loss":1, 
                    # "Scoremap_loss":1e-2,
                    # "CLIP_loss":1e+2
                    }

        criterion = SetCriterion(
            weight_dict=weight_dict,
            batch_size=cfg.SOLVER.IMS_PER_BATCH,
            num_tokens=cfg.MODEL.MASK_FORMER.NUM_OBJECT_TOKENS,
            use_hann = False
        )

        return {
            "backbone": backbone,
            "pixel_decoder": pixel_decoder,
            "mask_predictor": mask_predictor,
            "criterion": criterion,
            "num_tokens": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "device": cfg.MODEL.DEVICE,
            "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
        }


    def forward(self, batched_inputs, mode = 'train', visualization = False):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`VideoDataset`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "audio_path": str, path to audio file 'video_name.wav'
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
        """
        images = [x for x in batched_inputs["image"]]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        audio_list = []
        for path in batched_inputs['mixed_audio_path']:
            mixed_audio, sr = librosa.load(path, mono=True, sr=16000, duration=10)
            if (mixed_audio.shape[0] < 16000 * 10):
                mixed_audio = np.append(mixed_audio, np.zeros(3000, dtype=np.float32))
                mixed_audio = mixed_audio[:160000]
                # mixed_audio = mixed_audio.type(torch.cuda.FloatTensor)
            audio_list.append(torch.from_numpy(mixed_audio).unsqueeze(1))
        mixed_audio = ImageList.from_tensors(audio_list).tensor
        
        del audio_list

        with torch.no_grad():
            image_feature = self.backbone(images.tensor.to(self.device))
        
        if mode == 'train':
            
            # pp_embeds, image_features = self.pixel_decoder(features=image_feature)
            pp_embeds=None
            outputs = self.mask_predictor(image_feature[self.transformer_in_feature], pp_embeds, mixed_audio.to(self.device))
            
            return outputs

            losses = self.criterion(outputs, self.device)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] = losses[k] * self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            if self.return_dict:
                return losses
            else:
                loss = 0
                for key in losses.keys():
                    loss += losses[key]
                return loss
            
        elif mode == 'eval':
            sep_audio_list = []
            sep = []
            for path in batched_inputs["separated_audio_paths"]:
                sep_mixed_audio, sr = librosa.load(path[0], mono=True, sr=16000)
                sep_mixed_audio = np.append(sep_mixed_audio, np.zeros(3000))
                sep_mixed_audio = sep_mixed_audio[:160000]
                sep.append(torch.from_numpy(sep_mixed_audio).unsqueeze(1))
            # N = len(batched_inputs["separated_audio_paths"])
            # for n in range(N):
            #     sep_mixed_audio, sr = librosa.load(batched_inputs["separated_audio_paths"][n], mono=True, sr=16000)
            #     sep_mixed_audio = np.append(sep_mixed_audio, np.zeros(3000))
            #     sep_mixed_audio = sep_mixed_audio[:160000]
            #     sep.append(torch.from_numpy(sep_mixed_audio).unsqueeze(1))
            sep_audio_list.append(torch.cat(sep, dim=1))
            sep_audio_gts = ImageList.from_tensors(sep_audio_list).tensor
            del sep, sep_audio_list

            with torch.no_grad():
                pp_embeds, image_features = self.pixel_decoder(features=image_feature)
                outputs = self.mask_predictor(image_features, pp_embeds, mixed_audio.to(self.device), mode='eval')
            
            mask_pred_results = F.interpolate(
                                    outputs["pred_masks"],
                                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                                    mode="bilinear",
                                    align_corners=False,
                                )
            outputs = {"sep_audio_wavs_wo_noise": outputs["sep_audio_wavs_wo_noise"],
                        "pred_masks": outputs["pred_masks"],
                        "mask_pred_results": mask_pred_results}
            # outputs["mask_pred_results"] = mask_pred_results
            """(batch_size, num_tokens, img_width, img_height)"""
            
            processed_results = []
            if visualization:
                for mask_pred_result, input_per_image, image_size in zip(mask_pred_results, batched_inputs["image"], images.image_sizes):
                    mask_pred = mask_pred_result
                    mask_pred = mask_pred.sigmoid()
                    result = mask_pred; img_size = image_size
                    output_width = input_per_image.shape[2]; output_height = input_per_image.shape[1]

                    result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
                    result = F.interpolate(
                        result, size=(output_height, output_width), mode="bilinear", align_corners=False)[0]
                    
                    processed_results.append({"seg": result})

            return mixed_audio, sep_audio_gts.permute(0, 2, 1), outputs, processed_results

