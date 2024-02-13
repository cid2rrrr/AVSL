import torch
from torch import nn
from torch.nn import functional as F

# change the name to the actual modified .py name
# from AVSL import backbone as backbone # from detectron2
from detectron2.layers import ShapeSpec
from mask2former.modeling.backbone.swin import D2SwinTransformer
# from AVSL import pixel_decoder as pixel_decoder # from maskformer
from mask2former.modeling.pixel_decoder.fpn import TransformerEncoderPixelDecoder
# from AVSL import transformer_decoder as transformer_decoder # modified 
from mask2former.modeling.transformer_decoder.maskformer_transformer_decoder import StandardTransformerDecoder
from AVSL import audio_separation_model as audio_separation_model # modified
from AVSL import audio_encoder as audio_encoder # from VGGish
from AVSL import mlp # MLP layer for alignment btw audio & object
from AVSL import criterion # loss functions
from AVSL import config as cfg # parameters

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
        transformer_decoder: imported transformer decoder
        audio_separation_model: imported audio separation model
        audio_encoder: imported audio encoder, with frozen weights
        mlp: additional mlp layer for training loss
        num_queries: number of object queries
        noise_query: single query for noise
        criterion: losses
        config: weights, parameters
    """
    def __init__(self, 
                 backbone, 
                 pixel_decoder, 
                 transformer_decoder, 
                 audio_separation_model, 
                 audio_encoder, 
                 num_queries,
                 mlp,):
        super(AVSLModel, self).__init__()
        self.backbone = self.build_swin_backbone(cfg)
        self.pixel_decoder = self.build_pixel_decoder(cfg, backbone.output_shape())
        self.transformer_decoder = self.build_transformer_decoder(cfg)
        self.audio_separation_model = audio_separation_model
        self.audio_encoder = audio_encoder
        # num_queries is a parameter in config
        self.num_queries = num_queries
        # additional MLP for aligning in scoremap loss
        self.mlp = mlp()
    
    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs:  {“image”: torch.Tensor([3*1024*1024]), “audio_path”: str(audio file path)}
        return:
            per_pixel_embeddings: scoremap loss
            learned_object_queries: audio dominance loss
            mask_embedding: scoremap loss, clip loss
            mask_features: clip loss
            pred_masks: final output 
            original_audio: audio separation loss
            separated_audio_wav_list: audio separation loss, less separation loss(separation penalty), final output
            separated_audio_wav_features_list: 
        """
        if self.training == True :
            # training
            # num_query is imported from config
            image, audio = batched_inputs
            image_feature = self.backbone(image)
            # pixel_decoder_features
            mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder(image_feature)
            # change learned_object_queries to something
            learned_object_queries, mask_embedding, mask_features, pred_masks = self.transformer_decoder(transformer_encoder_features, mask_features, mask=None)
            original_audio, separated_audio_wav_list = self.audio_separation_model(audio, learned_object_queries)
            separated_audio_wav_features_list = self.audio_encoder(separated_audio_wav_list)

            # parameters need for loss
            output = {per_pixel_embeddings, mask_embedding, mask_features, learned_object_queries, original_audio, 
            separated_audio_wav_list, separated_audio_wav_features_list}

        else:
            # inference
            # num_query is imported from config
            image, audio = batched_inputs
            image_feature = self.backbone(image)
            # pixel_decoder_features
            per_pixel_embeddings = self.pixel_decoder(image_feature)
            learned_object_queries, mask_embedding, mask_features, pred_masks = self.transformer_decoder()
            separated_audio_wav_features_list = self.audio_encoder(separated_audio_wav_list)
            
            # final outputs, separated sound and its masks
            output = {pred_masks, separated_audio_wav_list}

        return output
    
    def build_swin_backbone(self, cfg, input_shape):
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
        model = D2SwinTransformer(cfg, input_shape)
        return model
    
    def build_pixel_decoder(self, cfg, input_shape):
        model = TransformerEncoderPixelDecoder(cfg, input_shape)
        return model
    
    def build_transformer_decoder(self, cfg):
        if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "transformer_encoder":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "pixel_embedding":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "multi_scale_pixel_decoder":  # for maskformer2
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        model = StandardTransformerDecoder(cfg, in_channels=transformer_predictor_in_channels, mask_classification=False)
        return model

    def from_config(cfg):
        backbone = self.build_swin_backbone(cfg)
        pixel_decoder = self.build_pixel_decoder(cfg, backbone.output_shape())
        transformer_decoder = self.build_transformer_decoder(cfg)
        return 

# config file parameters 
"""
def add_avsl_config(cfg):
    # configs from original maskformer
    # data config

    # select the dataset mapper
    # we are not using a dataset mapper
    #cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"

    # may need adjustments, keep
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # keep 
    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.AVSL = CN()
    #cfg.MODEL.MASK_FORMER = CN()

    # loss
    # need adjustments
    cfg.MODEL.AVSL.DEEP_SUPERVISION = True
    cfg.MODEL.AVSL.AS_WEIGHT = 1.0 #initialized 1.0
    cfg.MODEL.AVSL.LS_WEIGHT = 1.0 #initialized 1.0
    cfg.MODEL.AVSL.AD_WEIGHT = 1.0 #initialized 1.0
    cfg.MODEL.AVSL.SM_WEIGHT = 1.0 #initialized 1.0
    cfg.MODEL.AVSL.CLIP_WEIGHT = 1.0 #initialized 1.0
    #cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    #cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    #cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    #cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    #cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.AVSL.NHEADS = 8
    cfg.MODEL.AVSL.DROPOUT = 0.1 #initialized 0.1
    cfg.MODEL.AVSL.DIM_FEEDFORWARD = 2048 #initialized 2048
    cfg.MODEL.AVSL.ENC_LAYERS = 0 #initialized 0
    cfg.MODEL.AVSL.DEC_LAYERS = 6 #initialized 6
    cfg.MODEL.AVSL.PRE_NORM = False
    #cfg.MODEL.MASK_FORMER.NHEADS = 8
    #cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    #cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    #cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    #cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    #cfg.MODEL.MASK_FORMER.PRE_NORM = False

    # need adjustments 128
    cfg.MODEL.AVSL.HIDDEN_DIM = 128
    cfg.MODEL.AVSL.NUM_OBJECT_QUERIES = 100
    #cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    #cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.AVSL.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.AVSL.ENFORCE_INPUT_PROJ = False
    #cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    #cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.AVSL.TEST = CN()
    cfg.MODEL.AVSL.TEST.SEMANTIC_ON = True
    cfg.MODEL.AVSL.INSTANCE_ON = False
    cfg.MODEL.AVSL.TEST.PANOPTIC_ON = False
    cfg.MODEL.AVSL.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.AVSL.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.AVSL.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False
    #cfg.MODEL.MASK_FORMER.TEST = CN()
    #cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    #cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    #cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    #cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    #cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    #cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # keep
    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.AVSL.SIZE_DIVISIBILITY = 32
    #cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # keep
    # pixel decoder config 
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 128
    #cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # keep
    # for video
    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False # change to true we are using frozen backbone

    # need adjustments 
    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.AVSL.TRANSFORMER_DECODER_NAME = "transformer_decoder"
    #cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # keep
    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # may need adjustments, keep
    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # need adjustments
    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.AVSL.TRAIN_NUM_POINTS = 112 * 112
    #cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.AVSL.OVERSAMPLE_RATIO = 3.0
    #cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.AVSL.IMPORTANCE_SAMPLE_RATIO = 0.75
    #cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75
    """
