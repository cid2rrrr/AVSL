# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn

#####
from torch import from_numpy, index_select, tensor, stack 
import torchvggish.vggish as VGGish
import separator.ZeroShotASP as ASP
import separator.config as asp_config
#####
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .position_encoding import PositionEmbeddingSine
from .transformer import Transformer


class TransformerPredictor(nn.Module):
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

        #####
        use_feature_MLP: bool
        #####        
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

        #####
        self.audio_separator = ASP(config=asp_config, channels=self.num_queries) # config not imported

        self.vggish_param_dict = {
            'vggish': '../ckpt/vggish-10086976.pth',
            'pca': '../ckpt/vggish_pca_params-970ea276.pth'}
        self.VGGish = VGGish(self.vggish_param_dict)
        for param in self.VGGish.parameters():
            param.requires_grad = False

        self.use_feature_MLP = use_feature_MLP # init에서 define됨
        self.feature_MLP = MLP(input_dim=, hidden_dim=, output_dim=, num_layers=) # not defined yet
        self.audio_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        #####

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

        return ret

    def forward(self, x, mask_features, audio): # audio를 추가적으로 받아야 함

        #####
        out = {}
        out['mask_features'] = mask_features
        #####

        pos = self.pe_layer(x)
        # x = transformer_encoder_feature인데 None을 입력으로 받을 수 있음
        # pos가 N queries

        src = x
        mask = None
        hs, memory = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos)
        # hs가 출력된 N queries

        #####
        # N queries에 noise query를 추가하는 작업
        hs = self.do_something(hs)

        separated_result = self.audio_separator(input=audio ,condition=hs) # audio_separator not defined yet (ZeroshotASP)
        separated_wav = separated_result['wav'].detach().numpy()
        out["sep_audios"] = separated_wav
        separated_spec = separated_result['sp'].detach().numpy()
        out["sep_specs"] = separated_spec

        # exclude n'+1'st audio 
        indices_to_exclude = list(range(self.num_queries, separated_spec.shape[0], self.num_queries+1)) 
        separated_spec = index_select(from_numpy(x), 0, tensor([i for i in range(separated_spec.shape[0]) if i not in indices_to_exclude])).squeeze()

        # extract audio features w/ for loop and VGGish
        audio_features = []
        for i in range(separated_spec.shape[0]):
            audio_features.append(self.VGGish(x[i], fs=16000))
        audio_features = stack(audio_features)

        if self.use_feature_MLP: # feature_MLP를 넣어서 loss를 계산한다면
            out["MLP_feaures"] = self.feature_MLP(audio_features) # feature_MLP not defined yet

        audio_embed = self.audio_embed(audio_features) # self.audio_embed == self.mask_embed
        out["audio_embed"] = audio_embed

        outputs_seg_masks = self.torch.einsum("bqc,bchw->bqhw", audio_embed, mask_features) #einsum을 하는데 잘 모르겠음 여기서부터 추가 수정 필요
        out["pred_masks"] = outputs_seg_masks
        #####

        # ##### Original Codes
        # if self.mask_classification:
        #     outputs_class = self.class_embed(hs)
        #     out = {"pred_logits": outputs_class[-1]}
        # else:
        #     out = {}

        # if self.aux_loss:
        #     # [l, bs, queries, embed]
        #     mask_embed = self.mask_embed(hs)
        #     outputs_seg_masks = torch.einsum("lbqc,bchw->lbqhw", mask_embed, mask_features)
        #     out["pred_masks"] = outputs_seg_masks[-1]
        #     out["aux_outputs"] = self._set_aux_loss(
        #         outputs_class if self.mask_classification else None, outputs_seg_masks
        #     )
        # else:
        #     # FIXME h_boxes takes the last one computed, keep this in mind
        #     # [bs, queries, embed]
        #     mask_embed = self.mask_embed(hs[-1])
        #     outputs_seg_masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        #     out["pred_masks"] = outputs_seg_masks
        # #####
        return out # contains mask_features, sep_audios, sep_specs, MLP_features, audio_embed, pred_masks

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]


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
