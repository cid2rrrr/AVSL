# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d
from detectron2.utils.registry import Registry
from detectron2.structures import ImageList

from .position_encoding import PositionEmbeddingSine
from .transformer import Transformer


TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_MODULE")
TRANSFORMER_DECODER_REGISTRY.__doc__ = """
Registry for transformer module in MaskFormer.
"""


def build_transformer_decoder(cfg, in_channels, mask_classification=True):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, in_channels, mask_classification)


@TRANSFORMER_DECODER_REGISTRY.register()
class StandardTransformerDecoder(nn.Module):
    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dropout: float,
        dim_feedforward: int,
        enc_layers: int,
        dec_layers: int,
        pre_norm: bool,
        deep_supervision: bool,
        mask_dim: int,
        enforce_input_project: bool,
        batch_size: int,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
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

        self.mask_classification = mask_classification

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

        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.batch_size = batch_size

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        if in_channels != hidden_dim or enforce_input_project:
            self.input_proj = Conv2d(in_channels, hidden_dim, kernel_size=1)
            weight_init.c2_xavier_fill(self.input_proj)
        else:
            self.input_proj = nn.Sequential()
        self.aux_loss = deep_supervision

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
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

        return ret

    def forward(self, x, mask_features, mixed_audio, mask=None):
        if mask is not None:
            mask = F.interpolate(mask[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        pos = self.pe_layer(x, mask)

        src = x
        hs, memory = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos)

        decoder_output = hs[-1]
        output_size = hs[-1].shape

        # if self.mask_classification:
        #     outputs_class = self.class_embed(hs)
        #     out = {"pred_logits": outputs_class[-1]}
        # else:
        #     out = {}

        out = {}

        # audio_module
        noise_query = torch.rand(output_size[0], 1, output_size[2]) # noise_query 버전 여러 개 
        mixed_audio, separated_audio_spectrograms = self.audio_separation_model(mixed_audio, decoder_output, noise_query)
        separated_audio_features = self.audio_encoder(separated_audio_spectrograms)
        # av_feature = self.audio_module(mixed_audio, decoder_output, noise_query, self.batch_size)

        # scoremap prediction
        av_feature_list = self.av_feature_embed(separated_audio_features)
        separated_audio_features_embed = self.mask_embed(av_feature_list)
        pred_masks = torch.einsum("bqc,bchw->bqhw", separated_audio_features_embed, mask_features)

        out["original_audio"] = mixed_audio
        out["separated_audio_spectrograms"] = separated_audio_spectrograms 
        out["separated_audio_features"] = separated_audio_features
        out["separated_audio_features_embed"] = separated_audio_features_embed
        out["pred_masks"] = pred_masks

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
    
    def audio_separation_model(self, mixed_audio, decoder_output, noise_query): # mixed_audio: [batch_size, _, _]
        query_input = torch.cat([decoder_output, noise_query], dim=1) # [batch_size, num_queries+1, 128]
        separated_audio_spectrograms = []
        # 이 아래는 바꿔야 함
        for i in range(self.num_queries):
            separated_audio_spectrograms.append(torch.rand(self.batch_size, 1, 903, self.hidden_dim))
        return mixed_audio, separated_audio_spectrograms

    def audio_encoder(self, separated_audio_spectrograms):
        separated_audio_features = []
        for wav in separated_audio_spectrograms:
            out = torch.rand(self.batch_size, 903, self.hidden_dim) # num_queries = len(separated_audio_features)
            separated_audio_features.append(out)
        return separated_audio_features

    # def audio_module(self, mixed_audio, query_output, noise_query, batch_size):
    #     out = torch.rand(batch_size, self.num_queries, 903, self.hidden_dim)
    #     return out

    def av_feature_embed(self, separated_audio_features, pooling_mode='sum'):
        av_feature_list = ImageList.from_tensors(separated_audio_features).tensor
        av_feature_list = av_feature_list.permute(1, 0, 2, 3)
        if pooling_mode == 'sum':
            av_feature_list = av_feature_list.sum(dim=2)
        elif pooling_mode == 'mean':
            av_feature_list = av_feature_list.mean(dim=2)
        elif pooling_mode == 'max':
            av_feature_list = av_feature_list.max(dim=2).values
        return av_feature_list

    @torch.jit.unused
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

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
