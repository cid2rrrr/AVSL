"""
criterion.
recommend using at least 4 GPU cards with above 20GB memories per card.
"""
import logging
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from numpy import dot
from numpy.linalg import norm
from math import log
from math import log10
from librosa.feature import spectral_flatness
from detectron2.structures import ImageList
from torch.nn import CosineSimilarity

def AS_loss(mixed_audio_spec, sep_audio_specs):
    """
    Parameters: 
    - mixed_audio_spec: batch_size*(num_tokens+1), channel_num, time_steps, freq_bins
    - sep_audio_specs: batch_size*(num_tokens+1), channel_num, time_steps, freq_bins

    Returns: 
    - AS_loss for batch
    """
    return torch.sum(torch.pow(torch.sum(sep_audio_specs, dim=0) - mixed_audio_spec, 2))
    # loss = 0
    # for x in range(len(mixed_audio_spec)):
    #     sp_sum = 0
    #     for i in range(sep_audio_specs.shape[0]):
    #         # add all separated spectrogram in the batch 
    #         sp_sum += sep_audio_specs[i][x]
    #     # add all mixed spectrogram in the batch and get the MSE   
    #     loss += (sp_sum - mixed_audio_spec[x])**2
    # return loss

def hann_func(x):
    return 0.5 * (1 - np.cos(2 * np.pi * x))

# Original function
# def get_power(x, dB=False):
#     power = 0
#     for i in range(len(x)):
#         power += x[i]**2
#     if dB:
#         power = 10 * log(power)
#     return power

def get_power(x, dB=False):
    """
    Calculate the power of an audio signal.

    Parameters:
    - x: channel_num, time_steps, freq_bins # 1*502*128
    - dB: bool, Whether to return power in dB. Defaults to False.

    Returns:
    - power: a scalar or array(if channel_num is not 1)
    """
    # Sum of squares along time and frequency axes
    power = np.sum(np.abs(x)**2, axis=(1, 2))

    if dB:
        power = 10 * log10(power)
    
    return power

def get_energy(y):
    """
    x.shape = (batch, n, time_steps)
    return = (batch, energy(0~1))
    """
    return torch.mean(torch.pow(y, 2), dim=-1)

def gmean(input_x, dim):
    """
    get geometric mean
    """
    log_x = torch.log(input_x)
    return torch.exp(torch.mean(log_x, dim=dim))

def get_spectral_flatness(spec):
    """
    input : (batch, n, time, freq)
    return : (batch, n, time_scalar)
    """
    return torch.mean(gmean(spec,dim=-1)/(torch.mean(spec,dim=-1)),dim=2)

def get_power_LSJver(spec):
    """
    input : (batch, n, time, freq)
    return : (batch, n, time_scalar)
    """
    return torch.sum(torch.pow(spec, 2), dim=-1)


# Original function
# def LS_loss(sep_audio_specs):
#     """
#     sf: spectral_flatness of the audio 
#     p: power of the audio
#     """
#     # check if get_power & spectral_flatness can use spectrogram inputs
#     sf = spectral_flatness(sep_audio_specs)
#     p = get_power(sep_audio_specs)
#     y = sf * p
#     return hann_func(y)

def LS_loss_LSJver(sep_audio_specs,use_hann=False):
    flatness = get_spectral_flatness(sep_audio_specs)
    power = get_power_LSJver(sep_audio_specs)

    if use_hann:
        """
        !!! power should be normalized !!!
        if not flat(-> 1), power-> 1
        if flat(-> 0), power-> 0
        """
        return torch.sum(hann_func((1-flatness) * power))
    else:
        """
        작은 소리(power-> 0)는 flat(-> 1)할 테니까
        flatness, power 둘 다 0으로 갈 확률은 적을 것이라는 전제 하에 사용 가능성 있음
        """
        return torch.sum(flatness * power)


def LS_loss(sep_audio_specs):
    """
    sf: spectral_flatness of the audio, a scalar per audio
    p: power of the audio, a scalar per audio
    
    Parameters:
    - sep_audio_specs: batch_size*(num_tokens+1), channel_num, time_steps, freq_bins
    
    Returns: 
    - LS_loss for batch
    """
    # Initialize
    loss = 0

    # Shape (batch_size*5)*1*502*128
    # Index for sf & p is same & we are going to add the loss anyway
    # So just use batch_size*(num_tokens+1) as index
    batch_and_token_size = sep_audio_specs.shape[0]

    # Create an empty list to store the separated spectrograms
    specs = []

    # Iterate over each batch and extract the spec(1*502*128)
    for i in range(batch_and_token_size):
        spec = sep_audio_specs[i]
        specs.append(spec)
    
    for i in range(specs): 
        sf = spectral_flatness(specs[i])
        p = get_power(specs[i])
        # y = sf * p
        y = (1 - sf) * p 
        y = hann_func(y)

        # get the sum of losses for each separated audio
        loss += y

    return loss

def cos_sim(a,b):
    return dot(a,b)/(norm(a) * norm(b))

# Need modification
# Original function
cos = CosineSimilarity(dim=-1)
def AD_loss_LSJver(audio_sep_tokens, sep_audio_features, sep_audio_specs):
    """
    input
    - sep_audio_tokens : (batch, n, embed=128)
    - sep_audio_features(_comp): (batch, n, embed=128)
    """
    return torch.sum(get_spectral_flatness(sep_audio_specs) * cos(audio_sep_tokens, sep_audio_features))
    


def AD_loss(num_tokens, aligned_sep_audio_features, mixed_audio_spec, aligned_audio_sep_tokens):
    
    # need to align sep_audio_features & audio_sep_tokens for the loss
    
    """
    Parameters:
    - num_tokens: 4
    - aligned_sep_audio_features: batch_size*num_tokens, ?
    - mixed_audio_spec: batch_size*(num_tokens+1)?????, channel_num, time_steps, freq_bins # why does mixed audio have tokens?
    - aligned_audio_sep_tokens: batch_size*num_tokens, ?
    - batch_size: batch size

    Returns: 
    - AD_loss for batch
    """
    loss = 0
    # original function
    # for n in range(len(num_tokens)):
    #     loss +=  -log(spectral_flatness(mixed_audio_spec[n]))* cos_sim(np.max(aligned_sep_audio_features[n], axis=1), 
    #                                                                    aligned_audio_sep_tokens[n])

    # batch_size*(num_tokens+1) # ????? why are tokens in the mixed audio?
    #mixed_audio_spec.shape[0]
    # batch_size*num_tokens # audio features do not have noise token
    #aligned_audio_sep_tokens.shape[0]
    # need to match the shape

    return loss

# Original function
# def Scoremap_loss(masks: torch.Tensor) -> torch.Tensor:
#     loss_map = [map.max(dim=0).values / map.sum(dim=0) for map in masks]
#     losses = ImageList.from_tensors(loss_map).tensor # N*256*256
#     losses = -torch.log(losses)
#     return losses.sum()

def Scoremap_loss(masks_in_batch: torch.Tensor) -> torch.Tensor:
    """
    Parameters:
    - masks_in_batch: batch_size, num_tokens, mask_dim, mask_dim # batch_size*4*256*256

    Returns: 
    - Scoremap_loss for batch
    """
    # Initialize
    loss = 0

    # Shape batch_size*4*256*256
    batch_size = masks_in_batch.shape[0]

    # Create an empty list to store the separated chunks
    chunks = []

    # Iterate over each batch and extract the 4*256*256 masks
    for i in range(batch_size):
        chunk = masks_in_batch[i]
        chunks.append(chunk)

    # get each masks in a single video in batch        
    for i in range(chunks): 
        masks = chunks[i]
        # the code expects masks: num_tokens, mask_dim, mask_dim # 4*256*256
        loss_map = [map.max(dim=0).values / map.sum(dim=0) for map in masks]
        losses = ImageList.from_tensors(loss_map).tensor # N*256*256
        losses = -torch.log(losses)
        losses = losses.sum()

        # get the sum of losses in the batch
        loss += losses

    return loss

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def CLIP_loss(sep_audio_features_embeds: torch.Tensor, pp_embeds: torch.Tensor, batch_size) -> torch.Tensor:
    """
    Paramters:
    - sep_audio_features_embeds: batch_size, num_tokens, mask_dim
    - pp_embeds: batch_size, mask_dim, mask_dim, mask_dim
    - batch_size: batch size
        
    Returns: 
    - CLIP_loss for batch
    """
    logit_scores = torch.zeros((batch_size, batch_size))

    for i in range(batch_size):
        for j in range(batch_size):
          mask_outputs = torch.einsum("bqc,bchw->bqhw", sep_audio_features_embeds[j].unsqueeze(0), pp_embeds[i].unsqueeze(0))
          score = mask_outputs.sum()
          logit_scores[i][j] = score

    logit_scores = logit_scores / logit_scores.norm(p=1, dim=-1, keepdim=True)

    feature_loss = contrastive_loss(logit_scores)
    embed_loss = contrastive_loss(logit_scores.t())
    return (feature_loss + embed_loss) / 2.0

## basic setup 
class SetCriterion(nn.Module):
    
    def __init__(self, 
                 AS_loss, 
                 LS_loss, 
                 AD_loss, 
                 Scoremap_loss, 
                 CLIP_loss, 
                 AS_weight, 
                 LS_weight, 
                 AD_weight, 
                 Scoremap_weight, 
                 CLIP_weight, 
                 batch_size, 
                 num_tokens):
        """Create the criterion.

        Args:
        - Losses
        - Loss weights
        - batch_size
        - num_tokens
        """
        super().__init__()
        self.AS_loss = AS_loss
        self.LS_loss = LS_loss
        self.AD_loss = AD_loss
        self.Scoremap_loss = Scoremap_loss
        self.CLIP_loss = CLIP_loss
        self.AS_weight = AS_weight
        self.LS_weight = LS_weight
        self.AD_weight = AD_weight
        self.Scoremap_weight = Scoremap_weight
        self.CLIP_Weight = CLIP_weight
        self.batch_size = batch_size
        self.num_tokens = num_tokens

    def forward(self, 
                mixed_audio_spec, 
                sep_audio_specs, 
                num_tokens, 
                sep_audio_features, 
                audio_sep_tokens, 
                masks, 
                sep_audio_features_embeds, 
                pp_embeds, 
                batch_size):
        """Compute the losses

        - mixed_audio_spec: batch_size*(num_tokens+1)??, channel_num, time_steps, freq_bins
        - sep_audio_specs: batch_size*(num_tokens+1), channel_num, time_steps, freq_bins
        - num_tokens: 4
        - sep_audio_features: batch_size*num_tokens, time_steps, hidden_dim
        - audio_sep_tokens: batch_size, num_tokens, hidden_dim
        - masks: batch_size, num_tokens, mask_dim, mask_dim
        - sep_audio_features_embeds: batch_size, num_tokens, mask_dim
        - pp_embeds: batch_size, mask_dim, mask_dim, mask_dim
        - batch_size: batch size
        """
        AS_loss = self.AS_loss(mixed_audio_spec, sep_audio_specs)

        # Other losses uses a single video, need to get the sum of the batch
        # e.g. in batch of n videos there are multiple videos
        LS_loss = self.LS_loss(sep_audio_specs)
        AD_loss = self.AD_loss(num_tokens, sep_audio_features, mixed_audio_spec, audio_sep_tokens)
        Scoremap_loss = self.Scoremap_loss(masks)

        # CLIP loss uses the whole batch
        CLIP_loss = self.CLIP_loss(sep_audio_features_embeds, pp_embeds, batch_size)


        # weights for each losses
        weighted_AS_loss = self.AS_weight * AS_loss
        weighted_LS_loss = self.LS_weight * LS_loss
        weighted_AD_loss = self.AD_weight * AD_loss
        weighted_Scoremap_loss = self.Scoremap_weight * Scoremap_loss
        weighted_CLIP_loss = self.CLIP_Weight * CLIP_loss
        
        # sum
        total_loss = weighted_AS_loss + weighted_LS_loss + weighted_AD_loss + weighted_Scoremap_loss + weighted_CLIP_loss
        
        return total_loss

"""
Example usage
AS_weight = 1 
LS_weight = 1
AD_weight = 1 
Scoremap_weight = 1 
CLIP_weight = 1
batch_size = 2
num_tokens = 4

criterion = SetCriterion(AS_loss, 
                         LS_loss, 
                         AD_loss, 
                         Scoremap_loss, 
                         CLIP_loss, 
                         AS_weight, 
                         LS_weight, 
                         AD_weight, 
                         Scoremap_weight, 
                         CLIP_weight, 
                         batch_size, 
                         num_tokens)

# Calculate the output
model = Model(...)
batched_input = dataloader1(...)
output = model(batched_input)

# Calculate the loss using the SetCriterion instance
loss = criterion(mixed_audio_spec, 
                sep_audio_specs, num_tokens, 
                sep_audio_features, audio_sep_tokens, 
                masks, 
                sep_audio_features_embeds, 
                pp_embeds, 
                batch_size)

# Backpropagate

"""