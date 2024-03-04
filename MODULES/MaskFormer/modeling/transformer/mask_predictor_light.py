# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d
# from detectron2.structures import ImageList
# from detectron2.utils.registry import Registry

from .position_encoding import PositionEmbeddingSine
from .transformer import Transformer

from torch import from_numpy, index_select, tensor, stack 
# from ....AudioModule.LAAS import ZeroShotASP
# from ....AudioModule.torchvggish import vggish
# from ....AudioModule import config as Audioconfig
from ....AudioModule.LAAS import ZeroShotASP
# from ....AudioModule.torchvggish import vggish
from ....AudioModule.vggish.vggish import VGGish
from ....AudioModule.vggish.audioset import vggish_input
from ....AudioModule import config as Audioconfig

# TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_MODULE")
# TRANSFORMER_DECODER_REGISTRY.__doc__ = """
# Registry for transformer module in MaskFormer.
# """


# def build_transformer_decoder(cfg, in_channels, mask_classification=True):
#     """
#     Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
#     """
#     name = cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME
#     return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, in_channels, mask_classification)


# @TRANSFORMER_DECODER_REGISTRY.register()
class MaskPredictorLight(nn.Module):
    @configurable
    def __init__(
        self,
        in_channels,
        # mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_tokens: int,
        nheads: int,
        dropout: float,
        dim_feedforward: int,
        enc_layers: int,
        dec_layers: int,
        pre_norm: bool,
        deep_supervision: bool,
        mask_dim: int,
        enforce_input_project: bool,
        
        #####
        batch_size: int,
        eval_batch_size: int,
        device,
        # use_feature_MLP: bool
        #####       
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_tokens: number of queries
            nheads: number of heads
            dropout: dropout in Transformer
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            deep_supervision: whether to add supervision to every decoder layers
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        self.device = device

        # self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=deep_supervision,
        )

        self.num_tokens = num_tokens # num_objects -> num_tokens
        # self.eval_num_tokens = eval_num_tokens # add
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.batch_size = batch_size # add
        self.eval_batch_size = eval_batch_size # add

        self.learnable_input_queries = nn.Embedding(num_tokens, hidden_dim)
        self.noise_token = nn.Embedding(1, hidden_dim) # add

        if in_channels != hidden_dim or enforce_input_project:
            self.input_proj = Conv2d(in_channels, hidden_dim, kernel_size=1)
            weight_init.c2_xavier_fill(self.input_proj)
        else:
            self.input_proj = nn.Sequential()
        self.aux_loss = deep_supervision

        # audio module
        self.ASP = ZeroShotASP(config=Audioconfig)
        # self.VGGish = VGGish()
        # self.VGGish.load_state_dict(torch.load('./MODULES/ckpt/pytorch_vggish.pth'))
        
        # self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3,
        #                       self.batch_size, self.eval_batch_size, self.num_tokens)

    @classmethod
    # def from_config(cls, cfg, in_channels, mask_classification):
    def from_config(cls, cfg, in_channels):
        ret = {}
        ret["in_channels"] = in_channels
        # ret["mask_classification"] = mask_classification

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_tokens"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_TOKENS
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        ret["enc_layers"] = cfg.MODEL.MASK_FORMER.ENC_LAYERS
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["deep_supervision"] = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["batch_size"] = cfg.SOLVER.IMS_PER_BATCH
        ret["eval_batch_size"] = cfg.TEST.IMS_PER_BATCH
        ret["device"] = cfg.MODEL.DEVICE

        return ret

    def forward(self, x, pp_embeds, mixed_audio, mode = 'train'):

        out = {}
        
        """pp_embeds: (batch_size, mask_dim, mask_dim, mask_dim)"""
        # out["pp_embeds"] = pp_embeds


        ############## transformer decoder ##############         
        pos = self.pe_layer(x)

        src = x
        mask = None
        hs, memory = self.transformer(self.input_proj(src), mask, self.learnable_input_queries.weight, pos)
        ################################################ 

        audio_sep_tokens = hs[-1]
        """(batch_size, num_tokens, hidden_dim)"""
        out["audio_sep_tokens"] = audio_sep_tokens
        
        if mode=='train':
            noise_token = self.noise_token.weight.unsqueeze(0).repeat(self.batch_size, 1, 1)
        elif mode=='eval':
            noise_token = self.noise_token.weight.unsqueeze(0).repeat(self.eval_batch_size, 1, 1)
        """(batch_size, 1, hiddne_dim)"""

        input_tokens = torch.cat([audio_sep_tokens, noise_token], dim=1)
        """(batch_size, num_tokens+1, hidden_dim)"""

        ############## audio module ############## 
        # audio separator
        sep_result, sp = self.ASP(mixed_audio, input_tokens)
        """sp: (batch_size*(num_tokens+1), channel_num:1, 502, 1025)"""
        out["mixed_audio_spec"] = sp[0::(self.num_tokens + 1)].squeeze()
        sep_audio_wavs = sep_result['wav'] #.detach().numpy()
        """(batch_size*(num_tokens+1), 160480, channel_num:1)"""
        out["sep_audio_wavs"] = sep_audio_wavs
        sep_audio_specs = sep_result['sp'] #.detach().numpy()
        """(batch_size*(num_tokens+1), channel_num:1, 502, 1025)"""
        out["sep_audio_specs"] = sep_audio_specs
        
        return out
    
        ########## run until here ##########
        # exclude n'+1'st audio
        # ########## 
        indices_to_exclude = list(range(self.num_tokens, sep_audio_specs.shape[0], self.num_tokens+1))
        indices_to_exclude = tensor([i for i in range(sep_audio_specs.shape[0]) if i not in indices_to_exclude]).to(self.device)
        sep_audio_specs = index_select(sep_audio_specs, 0, indices_to_exclude).squeeze()
        """(batch_size*num_tokens, 502, 1025)"""
        if mode=='train':
            out["sep_audio_specs_wo_noise"] = sep_audio_specs.reshape(self.batch_size, -1, sep_audio_specs.shape[-2],sep_audio_specs.shape[-1])
        elif mode=='eval':
            out["sep_audio_specs_wo_noise"] = sep_audio_specs.reshape(self.eval_batch_size, -1, sep_audio_specs.shape[-2],sep_audio_specs.shape[-1])
        # ##########
            
        # indices_to_exclude = list(range(self.num_tokens, sep_audio_wavs.shape[0], self.num_tokens+1))
        # indices_to_exclude = tensor([i for i in range(sep_audio_wavs.shape[0]) if i not in indices_to_exclude]).to(self.device)
        # sep_audio_wavs = index_select(sep_audio_wavs, 0, indices_to_exclude).squeeze()
        # if mode=='train':
        #     out["sep_audio_wavs_wo_noise"] = sep_audio_wavs.reshape(self.batch_size, -1, sep_audio_wavs.shape[-1])
        # if mode=='eval':
        #     out["sep_audio_wavs_wo_noise"] = sep_audio_wavs.reshape(self.eval_batch_size, -1, sep_audio_wavs.shape[-1])
        # #!#!#!#
        # sep_audio_features = []
        # for i in range(sep_audio_wavs.shape[0]):
        #     with torch.no_grad():
        #         input_data = sep_audio_wavs[i].cpu().numpy()
        #         input_data = vggish_input.waveform_to_examples(input_data,16000)
        #         input_data = torch.from_numpy(input_data).unsqueeze(dim=1)
        #         input_data = input_data.float().to(self.device)
        #         sep_audio_features.append(self.VGGish(input_data))
        # sep_audio_features = stack(sep_audio_features)

        """(batch_size*num_tokens, time_steps:5, hidden_dim)"""
        # sep_audio_features = sep_audio_features.reshape((self.batch_size, self.num_tokens, sep_audio_features.shape[-2], sep_audio_features.shape[-1]))#.to(self.device)
        """(batch_size, num_tokens, time_steps:5, hidden_dim)"""
        # mp = nn.MaxPool1d(kernel_size=sep_audio_features.shape[-1])
        # sep_audio_features = mp(sep_audio_features.permute(0, 2, 1))
        # ##########################################
        # ############## mask prediction ############## 
        # #!#!#!#
        # sep_audio_features, sep_audio_features_embeds = self.mask_embed(sep_audio_features, 'sum')
        # if mode=='train':
        #     sep_audio_features, sep_audio_features_embeds, sep_audio_features_comp = self.mask_embed(sep_audio_features, 'mean')
        # elif mode=='eval':
        #     sep_audio_features, sep_audio_features_embeds, sep_audio_features_comp = self.mask_embed(sep_audio_features, 'mean', mode='eval')
        
        # """
        # sep_audio_features: (batch_size, num_tokens, hidden_dim)
        # sep_audio_features_embeds: (batch_size, num_tokens, make_dim)
        # """
        # out["sep_audio_features"] = sep_audio_features
        # #!#!#!#
        # out["sep_audio_features_comp"] = sep_audio_features_comp
        # #!#!#!#
        # out["sep_audio_features_embeds"] = sep_audio_features_embeds
        # pred_masks = torch.einsum("bqc,bchw->bqhw", sep_audio_features_embeds, pp_embeds)
        # """(batch_size, num_tokens, make_dim, make_dim)"""
        # # out["pred_masks"] = torch.sigmoid(pred_masks)
        # out["pred_masks"] = torch.maximum(torch.tensor(1e-4).type_as(pred_masks),torch.sigmoid(pred_masks))

        ############################################# 
        
    @torch.jit.unused
    def _set_aux_loss(self, pred_masks):
        return [{"pred_masks": m} for m in pred_masks]


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 batch_size, eval_batch_size, num_tokens):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        
        #!#!#!#
        self.layers4compare = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [input_dim])
        )
        #!#!#!#
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_tokens = num_tokens

    def forward(self, x, pooling_mode='sum', mode='train'):
        x = x.unsqueeze(1)
        size = x.shape
        if mode=='train':
            x = x.reshape(self.batch_size, self.num_tokens, size[2], size[3])
        elif mode=='eval':
            x = x.reshape(self.eval_batch_size, self.num_tokens, size[2], size[3])

        if pooling_mode == 'sum':
            x = x.sum(dim=2)
        elif pooling_mode == 'mean':
            x = x.mean(dim=2)
        elif pooling_mode == 'max':
            x = x.max(dim=2).values
        #!#!#!#
        x_comp = x.clone().detach()
        # features = torch.tensor(x)
        features = x.clone().detach() # 권장된다고 함
        #!#!#!#

        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        #!#!#!#
        for i, layer in enumerate(self.layers4compare):
            x_comp = F.relu(layer(x_comp)) if i < self.num_layers - 1 else layer(x_comp)

        return features, x, x_comp
        #!#!#!#
        return features, x