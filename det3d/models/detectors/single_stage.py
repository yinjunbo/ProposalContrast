import torch.nn as nn

from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from ..utils.finetune_utils import FrozenBatchNorm2d
from det3d.torchie.trainer import load_checkpoint

import torch
from torch.nn import functional as F
from copy import deepcopy
import numpy as np
import math
import sys
# from det3d.ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from det3d.ops.pointnet2.pointnet2_batch import pointnet2_utils as pointnet2_batch_utils

@DETECTORS.register_module
class SingleStageDetector(BaseDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck=None,
        bbox_head=None,
        embed_neck=None,
        ssl_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(SingleStageDetector, self).__init__()
        self.reader = builder.build_reader(reader)
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        if embed_neck is not None:
            self.embed_neck = builder.build_neck(embed_neck)
            self.npoints = embed_neck.samples_num
            self.voxel_range = embed_neck.voxel_cfg['range']
            self.voxel_size = embed_neck.voxel_cfg['voxel_size']

        if ssl_head is not None:
            self.ssl_head = builder.build_head(ssl_head)

        if bbox_head is not None:
            self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        if pretrained is None:
            return 
        try:
            load_checkpoint(self, pretrained, strict=False)
            print("init weight from {}".format(pretrained))
        except:
            print("no pretrained model at {}".format(pretrained))
            
    def extract_feat(self, data):
        input_features = self.reader(data)
        x = self.backbone(input_features)
        if self.with_neck:
            x = self.neck(x)
        return x

    def aug_test(self, example, rescale=False):
        raise NotImplementedError

    def forward(self, example, return_loss=True, **kwargs):
        pass

    def predict(self, example, preds_dicts):
        pass

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self

    @staticmethod
    def _create_buffer(N, s):
        mask = 1 - torch.eye(N * 2, dtype=torch.uint8, device=s.device)
        pos_ind = (torch.arange(N * 2, device=s.device),  # for each row
                   2 * torch.arange(N, dtype=torch.long, device=s.device).unsqueeze(1).repeat(
                       1, 2).view(-1, 1).squeeze())  # select pos samples
        neg_mask = torch.ones((N * 2, N * 2 - 1), dtype=torch.uint8, device=s.device)
        neg_mask[pos_ind] = 0
        return mask, pos_ind, neg_mask

    def bilinear_interpolate_torch(self, im, x, y):
        """
        Args:
            im: (H, W, C) [y, x]
            x: (N)
            y: (N)
        Returns:
        """
        x0 = torch.floor(x).long()
        x1 = x0 + 1

        y0 = torch.floor(y).long()
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, im.shape[1] - 1)
        x1 = torch.clamp(x1, 0, im.shape[1] - 1)
        y0 = torch.clamp(y0, 0, im.shape[0] - 1)
        y1 = torch.clamp(y1, 0, im.shape[0] - 1)

        Ia = im[y0, x0]
        Ib = im[y1, x0]
        Ic = im[y0, x1]
        Id = im[y1, x1]

        # d_x0, d_x1, d_y0, d_y1 = x - x0.type_as(x), x1.type_as(x) - x, y - y0.type_as(y), y1.type_as(y) - y
        # wa = d_x1 * d_y1
        # wb = d_x1 * d_y0
        # wc = d_x0 * d_y1
        # wd = d_x0 * d_y0

        wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
        wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
        wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
        wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
        ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(
            torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
        return ans

    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride, voxel=False):
        if not voxel:
            x_idxs = (keypoints[:, :, 0] - self.voxel_range[0]) / self.voxel_size[0]
            y_idxs = (keypoints[:, :, 1] - self.voxel_range[1]) / self.voxel_size[1]
            x_idxs = x_idxs / bev_stride
            y_idxs = y_idxs / bev_stride
        else:
            x_idxs = keypoints[:, :, 2].type_as(bev_features)
            y_idxs = keypoints[:, :, 1].type_as(bev_features)

        point_bev_features_list = []
        for k in range(batch_size):
            cur_x_idxs = x_idxs[k]
            cur_y_idxs = y_idxs[k]
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = self.bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features.unsqueeze(dim=0))

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (B, N, C0)
        return point_bev_features

    def forward_ssl(self, example, x):

        stride = example["shape"][0][0] / x.size(-1)
        batch_size = len(example['metadata'])
        count = 0
        proposal_feature_list = []
        proposal_center_list = []
        points_ds_list = []
        points_feature_list= []
        ret_dict = {}
        
        # sample proposal centers in the original point cloud
        points_overlap_org = torch.stack(example["points_overlap_org"])[..., :2]
        height_points = torch.zeros([*points_overlap_org.shape[:2], 1]).type_as(points_overlap_org) # ignore height
        points_overlap_org = torch.cat([points_overlap_org, height_points], dim=-1)
        fps_choice = pointnet2_batch_utils.furthest_point_sample(
            points_overlap_org.contiguous(), self.npoints
        ).long().squeeze()

        # compute correspondence
        for kk in range(batch_size):
            key_idx = example['overlap'][kk][fps_choice[kk]].type(torch.float32)
            example['correspondence'][kk] = torch.min(torch.abs(torch.sub(key_idx.unsqueeze(dim=-1),example['correspondence'][kk].unsqueeze(dim=0).type(torch.float32))),dim=-1)[1]
            example['correspondence_aug'][kk] = torch.min(torch.abs(torch.sub(key_idx.unsqueeze(dim=-1),example['correspondence_aug'][kk].unsqueeze(dim=0).type(torch.float32))),dim=-1)[1]

        for batch_itt in range(batch_size):
            z = x[count:2 * (batch_itt + 1)]
            points_list = example['points'][count:2 * (batch_itt + 1)]
            correspondence = example['correspondence'][batch_itt]
            correspondence_aug = example['correspondence_aug'][batch_itt]
            keypoints = torch.stack([points_list[0][correspondence, :3], points_list[1][correspondence_aug, :3]])

            proposal_feature_list.append(self.interpolate_from_bev_features(keypoints, z, 2, stride).squeeze())

            points_num = 65536
            for i in range(len(points_list)):
                cur_npoints = points_list[i].shape[0]
                if cur_npoints > points_num:  # random sampling is much faster than FPS
                    choice = torch.randperm(cur_npoints)[:points_num]
                else:
                    choice = torch.arange(cur_npoints)
                    extra_choice = torch.randint(high=cur_npoints,
                                                  size=(points_num - cur_npoints,))
                    choice = torch.cat((choice, extra_choice), dim=0)
                    choice = choice[torch.randperm(points_num)]
                points_list[i] = points_list[i][choice]

            points_ds = torch.stack([points_list[0], points_list[1]])[..., :3]
            points_ds_list.append(points_ds)
            points_feature_list.append(self.interpolate_from_bev_features(points_ds, z, 2, stride).transpose(1, 2).contiguous())
            proposal_center_list.append(keypoints)

            count = 2 * (batch_itt + 1)

        # proposal encoding
        proposal_feature_batch = torch.cat(proposal_feature_list, dim=0)  # (2N)xd
        proposal_center_batch = torch.cat(proposal_center_list, dim=0)
        points_ds_batch = torch.cat(points_ds_list, dim=0)
        points_feature_batch = torch.cat(points_feature_list, dim=0)
        proposal_instance, proposal_class = self.embed_neck(type='joint_instance_class_embed', xyz=points_ds_batch.contiguous(), features=points_feature_batch, new_xyz=proposal_center_batch, point_feat=proposal_feature_batch)

        # instance contrast
        proposal_instance = proposal_instance / (torch.norm(proposal_instance, p=2, dim=1, keepdim=True) + 1e-10)
        assert proposal_instance.size(0) % 2 == 0
        N = proposal_instance.size(0) // 2
        s = torch.matmul(proposal_instance, proposal_instance.permute(1, 0))  # (2N)x(2N)
        mask, pos_ind, neg_mask = self._create_buffer(N, s)
        s = torch.masked_select(s, mask == 1).reshape(s.size(0), -1)
        pos = s[pos_ind].unsqueeze(1)  # (2N)x1
        neg = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)
        ret_dict['instance'] = (pos, neg)

        # cluster contrast
        proposal_proj, proposal_pred = self.embed_neck(x=proposal_class, type='cluster')
        proposal_proj = torch.cat([proposal_proj[::2], proposal_proj[1::2]])
        proposal_pred = torch.cat([proposal_pred[::2], proposal_pred[1::2]])
        ret_dict['class'] = (proposal_proj, proposal_pred, self.embed_neck)

        return self.ssl_head(ret_dict)