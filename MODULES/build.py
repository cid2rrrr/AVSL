import torch

from detectron2.layers import ShapeSpec
from .MaskFormer.modeling.backbone.swin import D2SwinTransformer

def build_swin_backbone(cfg, pixel_mean):
    """
    pixel_mean = cfg.MODEL.PIXEL_MEAN
    """
    input_shape = ShapeSpec(channels=len(pixel_mean)) 
    model = D2SwinTransformer(cfg, input_shape)
    model.init_weights(cfg.MODEL.WEIGHTS) # 이렇게 하면 안됨. 고쳐야 함.
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

def build_mask_predictor(cfg, in_channels, audiomodule_weight):
    """
    in_channles = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
    audio_module_weight = {
            'vggish': 'ckpt/vggish-10086976.pth',
            'pca': 'ckpt/vggish_pca_params-970ea276.pth'
            }
    """
    model = MaskPredictor(cfg, in_channels, audiomodule_weight)
    return model
# mask_predictor = MaskPredictor(cfg, cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM, audiomodule_weight)

