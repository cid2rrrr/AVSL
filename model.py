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
        ...
        return ...
    