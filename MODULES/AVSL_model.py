# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
import librosa

from detectron2.config import configurable
# from detectron2.data import MetadataCatalog
# from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
# from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList

from .build import build_swin_backbone, build_pixel_decoder, build_mask_predictor
from .criterion_custom import SetCriterion

class AVSLModel(nn.Module):
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

        self.pixel_decoder = pixel_decoder
        pixel_decoder.to(torch.device(self.device))

        self.mask_predictor = mask_predictor.to(torch.device(self.device))
        self.mask_predictor.to(torch.device(self.device)) # 왜 두 번??

        self.training = training
        self.criterion = criterion
        self.num_tokens = num_tokens
        # self.overlap_threshold = overlap_threshold
        # self.panoptic_on = panoptic_on
        # self.object_mask_threshold = object_mask_threshold
        # self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        # self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)


    @classmethod
    def from_config(cls, cfg):
        ###
        backbone = build_swin_backbone(cfg, cfg.MODEL.PIXEL_MEAN)
        pixel_decoder = build_pixel_decoder(cfg, backbone.output_shape())
        mask_predictor = build_mask_predictor(cfg, cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM, 
                                                            {'vggish': 'MODULES/ckpt/vggish-10086976.pth',
                                                             'pca': 'MODULES/ckpt/vggish_pca_params-970ea276.pth'})
        ###

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        weight_dict = {"loss_ce": 1, "loss_mask": mask_weight, "loss_dice": dice_weight}
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = None
        # criterion = SetCriterion(
        #     num_classes=10, #sem_seg_head.num_classes,
        #     matcher=None, #matcher,
        #     weight_dict=weight_dict,
        #     eos_coef=no_object_weight,
        #     losses=losses,
        #     batch_size=cfg.SOLVER.IMS_PER_BATCH,
        #     num_tokens=cfg.MODEL.MASK_FORMER.NUM_OBJECT_TOKENS,
        # )

        return {
            "backbone": backbone,
            "pixel_decoder": pixel_decoder,
            "mask_predictor": mask_predictor,
            "criterion": criterion,
            "num_tokens": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            # "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            # "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            # "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            # "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            # "sem_seg_postprocess_before_inference": (
            #     cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
            #     or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
            # ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "device": cfg.MODEL.DEVICE,
        }

    # @property
    # def device(self):
    #     return self.pixel_mean.device

    def forward(self, batched_inputs):
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
        # images = [x.to(self.device) for x in batched_inputs["image"]]
        images = [x for x in batched_inputs["image"]]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        with torch.no_grad():
            image_feature = self.backbone(images.tensor.to(self.device))
        pp_embeds, image_features = self.pixel_decoder(image_feature)
        # image_features.to(torch.device(self.device))
        # pp_embeds.to(torch.device(self.device))

        audio_list = []
        for path in batched_inputs['audio_path']:
            mixed_audio, sr = librosa.load(path, mono=True, sr=16000)
            audio_list.append(torch.from_numpy(mixed_audio).unsqueeze(1))
        mixed_audio = ImageList.from_tensors(audio_list).tensor
        del audio_list

        outputs = self.mask_predictor(image_features, pp_embeds, mixed_audio.to(self.device))
        return outputs

        if self.training:
            # mask classification target
            # if "instances" in batched_inputs[0]:
            #     gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            #     targets = self.prepare_targets(gt_instances, images)
            # else:
            #     targets = None

            # bipartite matching-based loss
            # losses = self.criterion(output, targets)
            losses = self.criterion(outputs)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

            return losses
        else:
            # mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            processed_results = []
            # for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
            #     mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            # ):
            for mask_pred_result, input_per_image, image_size in zip(
                mask_pred_results, batched_inputs["image"], images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = sem_seg_postprocess(
                        mask_pred_result, image_size, height, width
                    )
                processed_results.append({"sem_seg": mask_pred_result})

                # semantic segmentation inference
                # r = self.semantic_inference(mask_cls_result, mask_pred_result)
                # if not self.sem_seg_postprocess_before_inference:
                #     r = sem_seg_postprocess(r, image_size, height, width)
                # processed_results.append({"sem_seg": r})

                # panoptic segmentation inference
                # if self.panoptic_on:
                #     panoptic_r = self.panoptic_inference(mask_cls_result, mask_pred_result)
                #     processed_results[-1]["panoptic_seg"] = panoptic_r

            return processed_results

    # def prepare_targets(self, targets, images):
    #     h, w = images.tensor.shape[-2:]
    #     new_targets = []
    #     for targets_per_image in targets:
    #         # pad gt
    #         gt_masks = targets_per_image.gt_masks
    #         padded_masks = torch.zeros((gt_masks.shape[0], h, w), dtype=gt_masks.dtype, device=gt_masks.device)
    #         padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
    #         new_targets.append(
    #             {
    #                 "labels": targets_per_image.gt_classes,
    #                 "masks": padded_masks,
    #             }
    #         )
    #     return new_targets

    # def semantic_inference(self, mask_cls, mask_pred):
    #     mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
    #     mask_pred = mask_pred.sigmoid()
    #     semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
    #     return semseg

    # def panoptic_inference(self, mask_cls, mask_pred):
    #     scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
    #     mask_pred = mask_pred.sigmoid()

    #     keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
    #     cur_scores = scores[keep]
    #     cur_classes = labels[keep]
    #     cur_masks = mask_pred[keep]
    #     cur_mask_cls = mask_cls[keep]
    #     cur_mask_cls = cur_mask_cls[:, :-1]

    #     cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

    #     h, w = cur_masks.shape[-2:]
    #     panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
    #     segments_info = []

    #     current_segment_id = 0

    #     if cur_masks.shape[0] == 0:
    #         # We didn't detect any mask :(
    #         return panoptic_seg, segments_info
    #     else:
    #         # take argmax
    #         cur_mask_ids = cur_prob_masks.argmax(0)
    #         stuff_memory_list = {}
    #         for k in range(cur_classes.shape[0]):
    #             pred_class = cur_classes[k].item()
    #             isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
    #             mask = cur_mask_ids == k
    #             mask_area = mask.sum().item()
    #             original_area = (cur_masks[k] >= 0.5).sum().item()

    #             if mask_area > 0 and original_area > 0:
    #                 if mask_area / original_area < self.overlap_threshold:
    #                     continue

    #                 # merge stuff regions
    #                 if not isthing:
    #                     if int(pred_class) in stuff_memory_list.keys():
    #                         panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
    #                         continue
    #                     else:
    #                         stuff_memory_list[int(pred_class)] = current_segment_id + 1

    #                 current_segment_id += 1
    #                 panoptic_seg[mask] = current_segment_id

    #                 segments_info.append(
    #                     {
    #                         "id": current_segment_id,
    #                         "isthing": bool(isthing),
    #                         "category_id": int(pred_class),
    #                     }
    #                 )

    #         return panoptic_seg, segments_info
