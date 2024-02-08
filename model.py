# change the name to the actual modified .py name
import torch
from torch import nn
from torch.nn import functional as F

from AVSL import backbone as backbone
from AVSL import pixel_decoder as pixel_decoder
from AVSL import transformer_decoder as transformer_decoder
from AVSL import audio_separation_model as audio_separation_model
from AVSL import audio_encoder as audio_encoder
from AVSL import MLP
from AVSL import criterion
from AVSL import config as cfg

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
                 MLP,):
        super(AVSLModel, self).__init__()
        self.backbone = backbone
        self.pixel_decoder = pixel_decoder
        self.transformer_decoder = transformer_decoder
        self.audio_separation_model = audio_separation_model
        self.audio_encoder = audio_encoder
        self.num_queries = num_queries
        # additional MLP for aligning in scoremap loss
        self.MLP = MLP()
    
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
            per_pixel_embeddings = self.pixel_decoder(image_feature)
            # change learned_object_queries to something
            learned_object_queries, mask_embedding, mask_features, pred_masks = self.transformer_decoder()
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
            
            # final outputs
            output = {pred_masks, separated_audio_wav_list}

        return output
    
    def from_config(cfg):
        ...
        return ...
    