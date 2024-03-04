import torch
import os

from fvcore.common.checkpoint import Checkpointer
from detectron2.layers import ShapeSpec
from .MaskFormer.modeling.backbone.swin import D2SwinTransformer

def build_swin_backbone(cfg, pixel_mean, weight_path = None):
    """
    pixel_mean = cfg.MODEL.PIXEL_MEAN
    weight_path = cfg.MODEL.WEIGHTS
    """
    input_shape = ShapeSpec(channels=len(pixel_mean)) 
    model = D2SwinTransformer(cfg, input_shape)

    if weight_path is not None:
        checkpoint = torch.load(os.path.join("MODULES/ckpt", weight_path))
        checkpointer = Checkpointer(model)
        checkpointer._convert_ndarray_to_tensor(checkpoint["model"])

        try:
            model.load_state_dict(checkpoint["model"])
        except:
            pass

    return model
# backbone = build_swin_backbone(cfg)


from .MaskFormer.modeling.pixel_decoder.pixel_decoder import TransformerEncoderPixelDecoder

def build_pixel_decoder(cfg, input_shape):
    """
    input_shape = backbone.output_shape()
    """
    model = TransformerEncoderPixelDecoder(cfg, input_shape) 
    return model
# pixel_decoder = build_pixel_decoder(cfg, backbone.output_shape())


from .MaskFormer.modeling.transformer.mask_predictor import MaskPredictor

def build_mask_predictor(cfg, in_channels):
    """
    in_channles = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
    """
    model = MaskPredictor(cfg, in_channels)
    return model
# mask_predictor = MaskPredictor(cfg, cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM)


from .MaskFormer.modeling.transformer.mask_predictor_light import MaskPredictorLight

def build_mask_predictor_light(cfg, in_channels):
    """
    in_channles = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
    """
    model = MaskPredictorLight(cfg, in_channels)
    return model
# mask_predictor = MaskPredictorLight(cfg, backbone.output_shape()[cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE].channels)

