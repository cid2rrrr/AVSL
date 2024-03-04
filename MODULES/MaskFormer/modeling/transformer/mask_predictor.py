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
from ....AudioModule.torchvggish import vggish
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
class MaskPredictor(nn.Module):
    @configurable
    def __init__(
        self,
        in_channels,
        audiomodule_weight,
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

        self.num_tokens = num_tokens # num_tokens -> num_tokens
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.batch_size = batch_size # add

        self.learnable_input_queries = nn.Embedding(num_tokens, hidden_dim)
        self.noise_token = nn.Embedding(1, hidden_dim) # add

        if in_channels != hidden_dim or enforce_input_project:
            self.input_proj = Conv2d(in_channels, hidden_dim, kernel_size=1)
            weight_init.c2_xavier_fill(self.input_proj)
        else:
            self.input_proj = nn.Sequential()
        self.aux_loss = deep_supervision

        # audio module
        # self.audio_separator = ASP(config=Audioconfig, channels=self.num_tokens) # config not imported
        self.ASP = ZeroShotASP(config=Audioconfig)
        self.ASP.to(torch.device(self.device))
        # self.urls = {
        #     'vggish': '../ckpt/vggish-10086976.pth',
        #     'pca': '../ckpt/vggish_pca_params-970ea276.pth'
        #     }
        # print(sys.path)
        self.VGGish = vggish.VGGish(audiomodule_weight, device=self.device)
        self.VGGish.to(torch.device(self.device))
        # self.vggish_param_dict = {
        #     'vggish': '../ckpt/vggish-10086976.pth',
        #     'pca': '../ckpt/vggish_pca_params-970ea276.pth'}
        
        # self.VGGish = VGGish(self.vggish_param_dict)
        for param in self.VGGish.parameters():
            param.requires_grad = False

        # self.use_feature_MLP = use_feature_MLP # init에서 define됨
        # self.feature_MLP = MLP(input_dim=, hidden_dim=, output_dim=, num_layers=) # not defined yet
        #####

        # output FFNs
        # if self.mask_classification:
        #     self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3,
                              self.batch_size, self.num_tokens)
        self.mask_embed.to(torch.device(self.device))

    @classmethod
    # def from_config(cls, cfg, in_channels, mask_classification):
    def from_config(cls, cfg, in_channels, audiomodule_weight):
        ret = {}
        ret["in_channels"] = in_channels
        # ret["mask_classification"] = mask_classification
        ret["audiomodule_weight"] = audiomodule_weight

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
        ret["device"] = cfg.MODEL.DEVICE

        return ret

    def forward(self, x, pp_embeds, mixed_audio):
        # if mask is not None:
        #     mask = F.interpolate(mask[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        # pos = self.pe_layer(x, mask)
        out = {}
        
        """pp_embeds: (batch_size, mask_dim, mask_dim, mask_dim)"""
        out["pp_embeds"] = pp_embeds

        ############## transformer decoder ##############         
        pos = self.pe_layer(x)

        src = x
        mask = None
        hs, memory = self.transformer(self.input_proj(src), mask, self.learnable_input_queries.weight, pos)
        ################################################ 

        audio_sep_tokens = hs[-1]
        """(batch_size, num_tokens, hidden_dim)"""
        out["audio_sep_tokens"] = audio_sep_tokens
        
        noise_token = self.noise_token.weight.unsqueeze(0).repeat(self.batch_size, 1, 1)
        """(batch_size, 1, hiddne_dim)"""
        input_tokens = torch.cat([audio_sep_tokens, noise_token], dim=1)
        """(batch_size, num_tokens+1, hidden_dim)"""

        # if self.mask_classification:
        #     outputs_class = self.class_embed(hs)
        #     out = {"pred_logits": outputs_class[-1]}
        # else:
        #     out = {}

        ############## audio module ############## 
        # audio separator
        sep_result, sp = self.ASP(mixed_audio, input_tokens)
        """sp: (batch_size*(num_tokens+1), channel_num:1, 502, 1025)"""
        out["mixed_audio_spec"] = sp
        sep_audio_wavs = sep_result['wav'] #.detach().numpy()
        """(batch_size*(num_tokens+1), 160480, channel_num:1)"""
        out["sep_audio_wavs"] = sep_audio_wavs
        sep_audio_specs = sep_result['sp'] #.detach().numpy()
        """(batch_size*(num_tokens+1), channel_num:1, 502, 1025)"""
        out["sep_audio_specs"] = sep_audio_specs
        # exclude n'+1'st audio 
        indices_to_exclude = list(range(self.num_tokens, sep_audio_specs.shape[0], self.num_tokens+1))
        indices_to_exclude = tensor([i for i in range(sep_audio_specs.shape[0]) if i not in indices_to_exclude]).to(self.device)
        sep_audio_specs = index_select(sep_audio_specs, 0, indices_to_exclude).squeeze()
        """(batch_size*num_tokens, 502, 1025)"""
        # audio encoder
        sep_audio_features = []
        for i in range(sep_audio_specs.shape[0]):
            sep_audio_features.append(self.VGGish(sep_audio_specs[i], fs=16000))
        sep_audio_features = stack(sep_audio_features)
        """(batch_size*num_tokens, time_steps:5, hidden_dim)"""
        # sep_audio_features = sep_audio_features.reshape((self.batch_size, self.num_tokens, sep_audio_features.shape[-2], sep_audio_features.shape[-1]))#.to(self.device)
        """(batch_size, num_tokens, time_steps:5, hidden_dim)"""
        # mp = nn.MaxPool1d(kernel_size=sep_audio_features.shape[-1])
        # sep_audio_features = mp(sep_audio_features.permute(0, 2, 1))
        ##########################################

        ############## mask prediction ############## 
        #!#!#!#
        # sep_audio_features, sep_audio_features_embeds = self.mask_embed(sep_audio_features, 'sum')
        sep_audio_features, sep_audio_features_embeds, sep_audio_features_comp = self.mask_embed(sep_audio_features, 'sum')
        #!#!#!#
        """
        sep_audio_features: (batch_size, num_tokens, hidden_dim)
        sep_audio_features_embeds: (batch_size, num_tokens, make_dim)
        """
        out["sep_audio_features"] = sep_audio_features
        #!#!#!#
        out["sep_audio_features_comp"] = sep_audio_features_comp
        #!#!#!#
        out["sep_audio_features_embeds"] = sep_audio_features_embeds
        pred_masks = torch.einsum("bqc,bchw->bqhw", sep_audio_features_embeds, pp_embeds)
        """(batch_size, num_tokens, make_dim, make_dim)"""
        out["pred_masks"] = pred_masks
        ############################################# 

        # if self.aux_loss:
        #     # [l, bs, queries, embed]
        #     mask_embed = self.mask_embed(hs)
        #     outputs_seg_masks = torch.einsum("lbqc,bchw->lbqhw", mask_embed, mask_features)
        #     out["pred_masks"] = outputs_seg_masks[-1]
        #     # out["aux_outputs"] = self._set_aux_loss(
        #     #     outputs_class if self.mask_classification else None, outputs_seg_masks
        #     # )
        # else:
        #     # FIXME h_boxes takes the last one computed, keep this in mind
        #     # [bs, queries, embed]
        #     mask_embed = self.mask_embed(hs[-1])
        #     outputs_seg_masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        #     out["pred_masks"] = outputs_seg_masks
        return out


    # @torch.jit.unused
    # def _set_aux_loss(self, outputs_class, outputs_seg_masks):
    #     # this is a workaround to make torchscript happy, as torchscript
    #     # doesn't support dictionary with non-homogeneous values, such
    #     # as a dict having both a Tensor and a list.
    #     if self.mask_classification:
    #         return [
    #             {"pred_logits": a, "pred_masks": b}
    #             for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
    #         ]
    #     else:
    #         return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
        
    @torch.jit.unused
    def _set_aux_loss(self, pred_masks):
        return [{"pred_masks": m} for m in pred_masks]


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 batch_size, num_tokens):
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
        self.num_tokens = num_tokens

    def forward(self, x, pooling_mode='sum'):
        x = x.unsqueeze(1)
        size = x.shape
        x = x.reshape(self.batch_size, self.num_tokens, size[2], size[3])

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
        for i, layer in enumerate(self.layers):
            x_comp = F.relu(layer(x_comp)) if i < self.num_layers - 1 else layer(x_comp)

        return features, x, x_comp
        #!#!#!#
        return features, x