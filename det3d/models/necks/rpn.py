import time
import numpy as np
import math

import torch

from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet
from torch.nn.modules.batchnorm import _BatchNorm

from det3d.torchie.cnn import constant_init, kaiming_init, xavier_init
from det3d.torchie.trainer import load_checkpoint
from det3d.models.utils import Empty, GroupNorm, Sequential
from det3d.models.utils import change_default_args
from det3d.ops.pointnet2.pointnet2_batch import pointnet2_utils

from .. import builder
from ..registry import NECKS
from ..utils import build_norm_layer

from det3d.models.necks.proposal_econding_module import ProposalEncodingLayerV1, ProposalEncodingLayerV2

@NECKS.register_module
class RPN(nn.Module):
    def __init__(
        self,
        layer_nums,
        ds_layer_strides,
        ds_num_filters,
        us_layer_strides,
        us_num_filters,
        num_input_features,
        norm_cfg=None,
        name="rpn",
        logger=None,
        **kwargs
    ):
        super(RPN, self).__init__()
        self._layer_strides = ds_layer_strides
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = us_layer_strides
        self._num_upsample_filters = us_num_filters
        self._num_input_features = num_input_features

        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self._norm_cfg = norm_cfg

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)
        assert len(self._num_upsample_filters) == len(self._upsample_strides)

        self._upsample_start_idx = len(self._layer_nums) - len(self._upsample_strides)

        must_equal_list = []
        for i in range(len(self._upsample_strides)):
            # print(upsample_strides[i])
            must_equal_list.append(
                self._upsample_strides[i]
                / np.prod(self._layer_strides[: i + self._upsample_start_idx + 1])
            )

        for val in must_equal_list:
            assert val == must_equal_list[0]

        in_filters = [self._num_input_features, *self._num_filters[:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(self._layer_nums):
            block, num_out_filters = self._make_layer(
                in_filters[i],
                self._num_filters[i],
                layer_num,
                stride=self._layer_strides[i],
            )
            blocks.append(block)
            if i - self._upsample_start_idx >= 0:
                stride = (self._upsample_strides[i - self._upsample_start_idx])
                if stride >= 1:
                    deblock = Sequential(
                        nn.ConvTranspose2d(
                            num_out_filters,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride,
                            bias=False,
                        ),
                        build_norm_layer(
                            self._norm_cfg,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                        )[1],
                        nn.ReLU(),
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    deblock = Sequential(
                        nn.Conv2d(
                            num_out_filters,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride,
                            bias=False,
                        ),
                        build_norm_layer(
                            self._norm_cfg,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                        )[1],
                        nn.ReLU(),
                    )
                deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

        logger.info("Finish RPN Initialization")

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):

        block = Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
            build_norm_layer(self._norm_cfg, planes)[1],
            # nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        for j in range(num_blocks):
            block.add(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block.add(
                build_norm_layer(self._norm_cfg, planes)[1],
                # nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
            )
            block.add(nn.ReLU())

        return block, planes

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, x):
        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x))
        if len(ups) > 0:
            x = torch.cat(ups, dim=1)

        return x


@NECKS.register_module
class SelfSupNeck(nn.Module):
    def __init__(self,
                 mode,
                 embed_layer,
                 radii,
                 npoints,
                 **kw,
                 ):

        super(SelfSupNeck, self).__init__()

        assert mode == "joint_instance_class_embed"

        radius = radii
        nsample = npoints

        self.groupers = nn.ModuleList()
        self.groupers = pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=False)

        self.attn = ProposalEncodingLayerV2(
            dim=embed_layer[0],
            pos_mlp_hidden_dim=64,
            attn_mlp_hidden_mult=4,
            downsample=4,
        )

        self._norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)
        self.share_linear = nn.Sequential(
            nn.Linear(embed_layer[0], embed_layer[1], bias=False),
            build_norm_layer(self._norm_cfg, embed_layer[1])[1],
            nn.ReLU(),)
        self.patch_neck = nn.Sequential(
            nn.Linear(embed_layer[1], embed_layer[2], bias=False),
            build_norm_layer(self._norm_cfg, embed_layer[2])[1],
        )

        self.projection_head = nn.Sequential(
            nn.Linear(embed_layer[1], embed_layer[2], bias=False),
        )
        self.prototypes = nn.Linear(embed_layer[2], 128, bias=False)


    def forward(self, type, x: torch.Tensor = None, xyz: torch.Tensor = None, features: torch.Tensor = None, new_xyz=None, point_feat=None) -> (torch.Tensor):

        if type == "joint_instance_class_embed":

            neighbor_feat, neighbor_point = self.groupers(xyz, new_xyz, features)  # (B, C, npoint, nsample)
            B, C, K, P = neighbor_feat.size()
            neighbor_feat = neighbor_feat.transpose(1, 2).transpose(2, 3).reshape([-1, P, C])
            neighbor_point = neighbor_point.transpose(1, 2).transpose(2, 3).reshape([-1, P, 3])
            point_feat = point_feat.reshape([-1, 1, C])
            input_features = (point_feat, neighbor_feat)
            new_xyz = new_xyz.reshape([-1, 1, 3])
            input_xyz = (new_xyz, neighbor_point)

            proposal_features = self.attn(input_features, input_xyz).squeeze()

            z = torch.stack(torch.chunk(proposal_features, B, dim=0))
            channel_num = z.size(-1)
            z_batch = list(torch.chunk(z, z.size(0)//2, dim=0))
            for i in range(len(z_batch)):
                z_split = z_batch[i]
                z_batch[i] = z_split.transpose(0, 1).reshape([-1, channel_num])

            z_batch = torch.cat(z_batch, dim=0)  # (2N)xd
            z_batch = self.share_linear(z_batch)
            z_insatnce = self.patch_neck(z_batch)

            return z_insatnce, z_batch


        if type == "cluster":
            # normalize the prototypes
            with torch.no_grad():
                w = self.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                self.prototypes.weight.copy_(w)

            x = self.projection_head(x)
            x = nn.functional.normalize(x, dim=1, p=2)

            return x, self.prototypes(x)
