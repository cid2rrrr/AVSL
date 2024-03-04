"""
criterion.
recommend using at least 4 GPU cards with above 20GB memories per card.
"""
# import logging
import torch
# import torch.nn.functional as F
from torch import nn
import numpy as np
from numpy import dot
from numpy.linalg import norm
from math import log10
from librosa.feature import spectral_flatness
# from detectron2.structures import ImageList
from torch.nn import CosineSimilarity

def AS_loss(mixed_audio_spec, sep_audio_specs, condition='pow_sum_sqrt'):
    """
    Parameters: 
    - mixed_audio_spec: batch_size*(num_tokens+1), channel_num, time_steps, freq_bins
    - sep_audio_specs: batch_size*(num_tokens+1), channel_num, time_steps, freq_bins

    Returns: 
    - AS_loss for batch
    """
    # mixed_audio_spec -> chan
    if condition == 'pow_sum':
        return torch.sum(torch.pow(torch.sum(sep_audio_specs, dim=0) - mixed_audio_spec, 2))
    elif condition == 'abs_sum':
        return torch.sum(torch.abs(torch.sum(sep_audio_specs, dim=0) - mixed_audio_spec, 2))
    elif condition == 'pow_sum_sqrt':
        return torch.sqrt(torch.sum(torch.pow(torch.sum(sep_audio_specs, dim=0) - mixed_audio_spec, 2)))


def hann_func(x):
    return 0.5 * (1 - np.cos(2 * np.pi * x))


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


def get_spectral_flatness(S,dim=-1, return_time_seq = True):
    x_thresh = torch.maximum(torch.tensor(1e-4).type_as(S), torch.pow(S,2))
    gmean = torch.exp(torch.mean(torch.log(x_thresh),dim=dim))
    amean = torch.mean(x_thresh,dim=dim)
    if return_time_seq:
        return gmean / amean
    else:
        return torch.mean(gmean/amean, dim=-1)



def get_power_LSJver(spec, calc='asdf'):
    """
    input : (batch, n, time, freq)
    return : (batch, n, time_scalar)
    """
    if calc == 'mean':
        return torch.mean(torch.pow(spec, 2), dim=-1)
    elif calc == 'sum':
        return torch.sum(torch.pow(spec, 2), dim=-1)
    elif calc == 'abs_mean':
        return torch.mean(torch.abs(spec), dim=-1)
    else:
        return torch.sum(torch.abs(spec), dim=-1)




cos_ = CosineSimilarity(dim=0)
def diff_feature_loss(sep_audio_features, batch_size):
    loss = torch.tensor(0)
    for b in range(batch_size):
        for i in range(batch_size):
            for j in range(batch_size):
                if i < j:
                    loss = loss + (1.0 - torch.abs(cos_(sep_audio_features[b][i],sep_audio_features[b][j])) + 1e-4)
    return loss

def LS_loss_LSJver(sep_audio_specs,use_hann=False, device = 'cuda:1'):
    power = get_power_LSJver(sep_audio_specs, calc='mean')

    # np_sep_audio_specs = sep_audio_specs.cpu().detach().numpy()
    # flatness = get_spectral_flatness(S=np.transpose(np_sep_audio_specs,(0,1,3,2)))
    # flatness = torch.from_numpy(flatness).to(device)
    flatness = get_spectral_flatness(sep_audio_specs)
    # flatness = torch.mean(flatness, dim=-1).squeeze()
    

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
        # return torch.sum(flatness * torch.mean(torch.log1p(power),dim=-1))
        return torch.sum(flatness * torch.log1p(power))


# Need modification
# Original function
cos = CosineSimilarity(dim=-1)
def AD_loss_LSJver(audio_sep_tokens, sep_audio_features, sep_audio_specs, device='cuda:1'):
    """
    input
    - sep_audio_tokens : (batch * (n+1), 1, embed=128)
    - sep_audio_features(_comp): (batch, n, embed=128)
    """
    # np_sep_audio_specs = sep_audio_specs.cpu().detach().numpy()
    # flatness = torch.from_numpy(spectral_flatness(S=np.transpose(np_sep_audio_specs,(0,1,3,2)))).to(device).squeeze()
    flatness = get_spectral_flatness(sep_audio_specs, return_time_seq=False)
    return torch.sum(flatness * torch.abs(cos(audio_sep_tokens, sep_audio_features)))

    

def Scoremap_loss_LSJver(masks):
    return torch.sum(-torch.log((masks.max(dim=1).values)/(masks.sum(dim=1))))


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
                batch_size, 
                num_tokens,
                small_value = 1e-4,
                use_hann = False,
                weight_dict = {
                    "AS_weight":1, 
                    "LS_weight":1, 
                    "AD_weight":1, 
                    "Scoremap_weight":0.001,
                    "CLIP_weight":1}
                ):
        """Create the criterion.

        Args:
        - Losses
        - Loss weights
        - batch_size
        - num_tokens
        """
        super().__init__()
        self.AS_loss = AS_loss
        self.LS_loss = LS_loss_LSJver
        self.diff_feature_loss = diff_feature_loss
        self.diff_token_loss = diff_feature_loss
        self.AD_loss = AD_loss_LSJver
        self.Scoremap_loss = Scoremap_loss_LSJver
        self.CLIP_loss = CLIP_loss
        self.weight_dict = weight_dict
        self.batch_size = batch_size
        self.num_tokens = num_tokens
        self.use_hann = use_hann
        self.small_value = small_value
        # self.device = 'cuda:1'

    def forward(self, output, device):
        """Compute the losses

        - mixed_audio_spec: batch_size*(num_tokens+1)??, channel_num, time_steps, freq_bins
        - sep_audio_specs: batch_size, channel_num, time_steps, freq_bins
        - num_tokens: 4
        - sep_audio_features: batch_size*num_tokens, time_steps, hidden_dim
        - audio_sep_tokens: batch_size, num_tokens, hidden_dim
        - masks: batch_size, num_tokens, mask_dim, mask_dim
        - sep_audio_features_embeds: batch_size, num_tokens, mask_dim
        - pp_embeds: batch_size, mask_dim, mask_dim, mask_dim
        - batch_size: batch size
        """
        loss_dict = {}
        loss_dict["AS_loss"] = self.AS_loss(output["mixed_audio_spec"], output["sep_audio_specs"])
        loss_dict["LS_loss"] = self.LS_loss(output["sep_audio_specs_wo_noise"], self.use_hann, device=device)
        loss_dict["DF_loss"] = self.diff_feature_loss(output["sep_audio_features"], self.batch_size)
        loss_dict["DT_loss"] = self.diff_token_loss(output['audio_sep_tokens'], self.batch_size)
        loss_dict["AD_loss"] = self.AD_loss(output["audio_sep_tokens"], output["sep_audio_features_comp"], output["sep_audio_specs_wo_noise"], device=device)
        loss_dict["Scoremap_loss"] = self.Scoremap_loss(output["pred_masks"])
        loss_dict["CLIP_loss"] = self.CLIP_loss(output["sep_audio_features_embeds"], output["pp_embeds"], self.batch_size)


        return loss_dict
        

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