import numpy as np
import os

from det3d.core.bbox import box_np_ops
from det3d.core.sampler import preprocess as prep
from det3d.builder import build_dbsampler

from det3d.core.input.voxel_generator import VoxelGenerator
from det3d.core.utils.center_utils import (
    draw_umich_gaussian, gaussian_radius
)
from ..registry import PIPELINES

import random
import pyransac3d as pyrsc


def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]


def drop_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds


@PIPELINES.register_module
class Preprocess(object):
    def __init__(self, cfg=None, **kwargs):
        self.shuffle_points = cfg.shuffle_points
        self.min_points_in_gt = cfg.get("min_points_in_gt", -1)

        self.mode = cfg.mode
        self.ssl_mode = cfg.get("ssl_mode", False)

        if self.mode == "train":
            self.global_rotation_noise = cfg.global_rot_noise
            # self.global_translate_noise_std = cfg.global_trans_noise
            self.global_scaling_noise = cfg.global_scale_noise
            self.class_names = cfg.class_names
            if cfg.db_sampler != None:
                self.db_sampler = build_dbsampler(cfg.db_sampler)
            else:
                self.db_sampler = None

        self.no_augmentation = cfg.get('no_augmentation', False)
        self.point_cloud_range = cfg.voxel_cfg['range']

        self.remove_ground = True
        self.remove_ego_points = True

    def __call__(self, res, info):

        res["mode"] = self.mode

        if res["type"] in ["WaymoDataset"]:
            if "combined" in res["lidar"]:
                points = res["lidar"]["combined"]
            else:
                points = res["lidar"]["points"]
            ds_npoints = 120000  # for random dropout

        elif res["type"] in ["NuScenesDataset"]:
            points = res["lidar"]["combined"]
        else:
            raise NotImplementedError

        if self.ssl_mode:

            self.ds_factor = 0.15
            self.npoints = 16384

            if self.remove_ground and self.mode == 'train':
                filename = os.path.splitext(os.path.basename(info['path']))[0] + '.npy'
                ground_point_idx = np.load(os.path.join('data/Waymo/train/lidar_ground', filename))
                ground_point_idx = ground_point_idx[np.random.permutation(ground_point_idx.shape[0])[:int(ground_point_idx.shape[0]*0.7)]]
                ground_mask = np.zeros(len(points), dtype=bool)
                ground_mask[ground_point_idx] = 1

            if self.remove_ego_points and self.mode == 'train':
                center_radius = 3
                ego_mask = (np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius)
                ground_mask = ground_mask | ego_mask

            # mask points by range
            upper_idx = np.sum((points[:, 0:3] <= self.point_cloud_range[3:6]).astype(np.int32), 1) == 3
            lower_idx = np.sum((points[:, 0:3] >= self.point_cloud_range[0:3]).astype(np.int32), 1) == 3
            new_pointidx = (upper_idx) & (lower_idx)
            points = points[new_pointidx, :]
            if self.remove_ground and self.mode == 'train':
                ground_mask = ground_mask[new_pointidx]

            res["lidar"]["points_overlap_org"] = points.copy()

            # ensure the correspondence
            upper_idx = np.sum((points[:, 0:2] <= [ele*0.8 for ele in self.point_cloud_range[3:5]]).astype(np.int32), 1) == 2
            lower_idx = np.sum((points[:, 0:2] >= [ele*0.8 for ele in self.point_cloud_range[0:2]]).astype(np.int32), 1) == 2
            all_idx = np.arange(len(points))
            new_pointidx = (upper_idx) & (lower_idx)
            reserve_idx_all = all_idx[new_pointidx]
            reserve_idx = reserve_idx_all[np.random.permutation(reserve_idx_all.shape[0])[:int(ds_npoints*self.ds_factor)]]
            mask = np.zeros(points.shape[0])
            mask[reserve_idx] = 1
            other_idx = all_idx[(1-mask).astype(bool)]

            if self.mode == 'train':
                aug_list = ['global', 'random', 'local']
                if 'global' in aug_list:
                    _, points = prep.random_flip_both(None, points, kitti_format=False)
                    _, points = prep.global_rotation(None, points, rotation=[-0.3925, 0.3925])
                    _, points = prep.global_scaling_v2(None, points, *[0.95, 1.05])

            points_org = [points, points.copy()]
            if not self.mode == "train":
                res["lidar"]["points"] = points_org
                return res, info

            points_aug = []
            correspondence = [[],[]]
            points_idx = []
            augmentation_list = {'flip':[], 'rot':[], 'scale':[]}
            for i, points in enumerate(points_org):

                if 'random' in aug_list and len(points) > ds_npoints: # drop out
                    other_idx_new = other_idx[np.random.permutation(other_idx.shape[0])[:int(ds_npoints*(1-self.ds_factor))]]
                    newidx = np.concatenate([reserve_idx, other_idx_new])
                    newidx = newidx[np.random.permutation(newidx.shape[0])]
                    points = points[newidx, :]
                    correspondence[i].append(newidx)
                else:
                    newidx = np.arange(0, len(points), dtype=np.int32)
                    correspondence[i].append(newidx)

                if 'local' in aug_list:
                    flip_param, points = prep.random_flip_both_ssl(None, points, kitti_format=False)
                    rot_param, points = prep.global_rotation_ssl(
                        None, points, rotation=self.global_rotation_noise
                    )
                    scale_param, points = prep.global_scaling_v2_ssl(
                        None, points, *self.global_scaling_noise
                    )
                    augmentation_list['flip'].append(flip_param)
                    augmentation_list['rot'].append(rot_param)
                    augmentation_list['scale'].append(scale_param)

                for j in range(len(correspondence[i])):
                    if j == 0:
                        remain = correspondence[i][j]
                    else:
                        remain = remain[correspondence[i][j]]

                points_idx.append(remain)
                points_aug.append(points)

            overlap_idx = np.intersect1d(points_idx[0], points_idx[1])
            overlap_idx = np.intersect1d(overlap_idx, reserve_idx_all)   # smaller area for FPS
            if self.remove_ground:
                overlap_idx = np.intersect1d(overlap_idx, all_idx[ground_mask==0])
            if len(overlap_idx)>self.npoints:  # for batch computation
                newidx = np.random.choice(len(overlap_idx), self.npoints, replace=False)
                overlap_idx = overlap_idx[newidx]
            else:
                return None, None

            res["lidar"]['correspondence'] = np.array(points_idx[0])
            res["lidar"]['correspondence_aug'] = np.array(points_idx[1])
            res["lidar"]["points_overlap_org"] = res["lidar"]["points_overlap_org"][overlap_idx]
            res["lidar"]['overlap'] = overlap_idx

            points_org = points_aug

            res["lidar"]["points"] = points_org
            res["augmentation_list"] = augmentation_list
            return res, info

        if self.mode == "train":
            anno_dict = res["lidar"]["annotations"]

            gt_dict = {
                "gt_boxes": anno_dict["boxes"],
                "gt_names": np.array(anno_dict["names"]).reshape(-1),
            }

        if self.mode == "train" and not self.no_augmentation:
            selected = drop_arrays_by_name(
                gt_dict["gt_names"], ["DontCare", "ignore", "UNKNOWN"]
            )

            _dict_select(gt_dict, selected)

            if self.min_points_in_gt > 0:
                point_counts = box_np_ops.points_count_rbbox(
                    points, gt_dict["gt_boxes"]
                )
                mask = point_counts >= min_points_in_gt
                _dict_select(gt_dict, mask)

            gt_boxes_mask = np.array(
                [n in self.class_names for n in gt_dict["gt_names"]], dtype=np.bool_
            )

            if self.db_sampler:
                sampled_dict = self.db_sampler.sample_all(
                    res["metadata"]["image_prefix"],
                    gt_dict["gt_boxes"],
                    gt_dict["gt_names"],
                    res["metadata"]["num_point_features"],
                    False,
                    gt_group_ids=None,
                    calib=None,
                    road_planes=None
                )

                if sampled_dict is not None:
                    sampled_gt_names = sampled_dict["gt_names"]
                    sampled_gt_boxes = sampled_dict["gt_boxes"]
                    sampled_points = sampled_dict["points"]
                    sampled_gt_masks = sampled_dict["gt_masks"]
                    gt_dict["gt_names"] = np.concatenate(
                        [gt_dict["gt_names"], sampled_gt_names], axis=0
                    )
                    gt_dict["gt_boxes"] = np.concatenate(
                        [gt_dict["gt_boxes"], sampled_gt_boxes]
                    )
                    gt_boxes_mask = np.concatenate(
                        [gt_boxes_mask, sampled_gt_masks], axis=0
                    )

                    points = np.concatenate([sampled_points, points], axis=0)

            _dict_select(gt_dict, gt_boxes_mask)

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_dict["gt_classes"] = gt_classes

            gt_dict["gt_boxes"], points = prep.random_flip_both(gt_dict["gt_boxes"], points)

            gt_dict["gt_boxes"], points = prep.global_rotation(
                gt_dict["gt_boxes"], points, rotation=self.global_rotation_noise
            )
            gt_dict["gt_boxes"], points = prep.global_scaling_v2(
                gt_dict["gt_boxes"], points, *self.global_scaling_noise
            )
        elif self.no_augmentation:
            gt_boxes_mask = np.array(
                [n in self.class_names for n in gt_dict["gt_names"]], dtype=np.bool_
            )
            _dict_select(gt_dict, gt_boxes_mask)

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_dict["gt_classes"] = gt_classes

        if self.shuffle_points:
            np.random.shuffle(points)

        res["lidar"]["points"] = points

        if self.mode == "train":
            res["lidar"]["annotations"] = gt_dict

        return res, info


@PIPELINES.register_module
class Voxelization(object):
    def __init__(self, **kwargs):
        cfg = kwargs.get("cfg", None)
        self.range = cfg.range
        self.voxel_size = cfg.voxel_size
        self.max_points_in_voxel = cfg.max_points_in_voxel
        self.max_voxel_num = [cfg.max_voxel_num, cfg.max_voxel_num] if isinstance(cfg.max_voxel_num,
                                                                                  int) else cfg.max_voxel_num

        self.double_flip = cfg.get('double_flip', False)

        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num[0],
        )
        self.ssl_mode = cfg.get('ssl_mode', 0)
        self.aug_mode = [['local', 'global_drop'], ['local', 'local_drop'], 'local', 'global_drop', 'no_aug'][-1]
        if self.ssl_mode and 'local' in self.aug_mode:
            self.voxel_generator_aug = VoxelGenerator(
                voxel_size=[1.0, 1.0, cfg.range[-1]-cfg.range[2]],
                point_cloud_range=self.range,
                max_num_points=500 if self.max_voxel_num[0]==90000 else 200,
                max_voxels=self.max_voxel_num[0],
            )
            self.sample_num = cfg.get('samples_num', 2048)

    def __call__(self, res, info):

        if self.ssl_mode:

            visual = False
            points1_org, points2_org = res["lidar"]["points"]
            # res["lidar"]["fps_points"] = [points1_org, points2_org]

            if 'local' in self.aug_mode:
                voxels, coordinates, num_points, points_idx = self.voxel_generator_aug.generate(
                    points1_org
                )
                voxel_num = voxels.shape[0]
                points1, points2 = [], []
                points_fps1, points_fps2 = [], []
                for i in range(voxel_num):
                    voxel_points1 = voxels[i][:num_points[i]]
                    mean_voxel_points1 = voxel_points1[:,:3].mean(axis=0)
                    points_fps1.append(mean_voxel_points1)
                    voxel_points2 = points2_org[points_idx[i][:num_points[i]]]
                    mean_voxel_points2 = voxel_points2[:,:3].mean(axis=0)
                    points_fps2.append(mean_voxel_points2)
                    if 'local_drop' in self.aug_mode and random.random() > 0.5:
                        sparse = [10, 5, 2][np.random.randint(0,3)]
                        sparse = num_points[i] // sparse
                        newidx = np.random.choice(len(voxel_points2), sparse, replace=False)
                        if random.random() > 0.5:
                            voxel_points1 = voxel_points1[newidx, :]
                        else:
                            voxel_points2 = voxel_points2[newidx, :]
                    if 'local_aug' in self.aug_mode and random.random() > 0.5:
                        xyz_center = np.expand_dims(np.mean(voxel_points[:, :3], axis=0), 0)
                        voxel_points[:, :3] = voxel_points[:, :3] - xyz_center

                        _, voxel_points = prep.random_flip_both(None, voxel_points)

                        _, voxel_points = prep.global_rotation(
                            None, voxel_points, rotation=[-3.141592653589793, 3.141592653589793]
                        )
                        _, voxel_points = prep.global_scaling_v2(
                            None, voxel_points, *[0.8, 1.25]
                        )
                        voxel_points[:, :3] += xyz_center
                        points2.append(voxel_points2[newidx, :])
                    else:
                        points1.append(voxel_points1)
                        points2.append(voxel_points2)

                res["lidar"]["points"][0], res["lidar"]["points"][1] = np.concatenate(points1, axis=0), np.concatenate(points2, axis=0)
                if voxel_num >= self.sample_num:
                    index = np.random.choice(np.arange(voxel_num), size=self.sample_num, replace=False)
                else:
                    index = np.random.choice(np.arange(voxel_num), size=voxel_num, replace=False)
                    # index_add = np.random.choice(np.arange(voxel_num), size=self.sample_num-voxel_num, replace=False)
                    # list(index).extend(list(index_add))
                res["lidar"]["fps_points"][0], res["lidar"]["fps_points"][1] = np.vstack(points_fps1)[index], np.vstack(points_fps2)[index]

                # res["lidar"]["fps_points"][0], res["lidar"]["fps_points"][1]

                # batch process
                # _, voxels = prep.global_rotation(
                #         None, voxels, rotation=[-3.141592653589793, 3.141592653589793]
                #     )
                # for i in range(voxels.shape[0]):
                #     voxel_points = voxels[i][:num_points[i]]
                #     points_aug.append(voxel_points)
                if visual:
                    import matplotlib.pyplot as plt
                    for i in range(len(res["lidar"]["points"])):
                        points_visual = res["lidar"]["points"][i][:, :3].transpose()
                        fig = plt.figure(figsize=(12, 12))
                        ax = fig.add_subplot(111)
                        dists = np.sqrt(np.sum(points_visual[:2, :] ** 2, axis=0))
                        colors = np.minimum(1, dists / 75 / np.sqrt(2))
                        ax.scatter(points_visual[0, :], -points_visual[1, :], c=colors, s=0.2)  # r,g,b
                        # # ax.scatter(src_node_np.transpose()[0, :], -src_node_np.transpose()[1, :], c=[0, 1, 0], s=1.0)
                        # ax.scatter(points2[0, :], -points2[1, :], c=[1, 0, 0], s=0.2)
                        # # ax.scatter(dst_node[0, :], -dst_node[1, :], c=[0, 1, 0], s=1.0)
                        ax.plot(0, 0, 'x', color='black')
                        ax.axis('off')
                        ax.set_aspect('equal')
                        fig.savefig(
                            '/home/junbo/repository/CenterPoint-Local/output/visual/augmentation/{}_view{}.png'.format(
                                res['metadata']['token'], i), format='png')
                        plt.close(fig)

            if 'drop' in self.aug_mode and len(points1_org) > 65536 and random.random()>0.5:
                idx1, idx2  = random.randint(0,2), random.randint(0,2)
                # sparse1 = [16384, 32768, 65536, len(points1_org)][idx1]
                sparse2 = [16384, 32768, 65536][idx2]
                # newidx1 = np.random.choice(len(points1_org), sparse1, replace=False)
                newidx2 = np.random.choice(len(points1_org), sparse2, replace=False)
                # res["lidar"]["points"][0] = points1_org[newidx1]
                res["lidar"]["points"][1] = points2_org[newidx2]

                # if idx1>=idx2:
                #     res["lidar"]["fps_points"][0] = points1_org[newidx2]
                #     res["lidar"]["fps_points"][1] = points2_org[newidx2]
                # else:
                #     res["lidar"]["fps_points"][0] = points1_org[newidx1]
                #     res["lidar"]["fps_points"][1] = points2_org[newidx1]

        voxel_size = self.voxel_generator.voxel_size
        pc_range = self.voxel_generator.point_cloud_range
        grid_size = self.voxel_generator.grid_size

        if res["mode"] == "train" and res["lidar"]["annotations"] is not None:
            gt_dict = res["lidar"]["annotations"]
            bv_range = pc_range[[0, 1, 3, 4]]
            mask = prep.filter_gt_box_outside_range(gt_dict["gt_boxes"], bv_range)
            _dict_select(gt_dict, mask)

            res["lidar"]["annotations"] = gt_dict
            max_voxels = self.max_voxel_num[0]
        else:
            max_voxels = self.max_voxel_num[1]
        if isinstance(res["lidar"]["points"], list):
            voxels_list, coordinates_list, num_points_list, num_voxels_list = [], [], [], []
            for points in res["lidar"]["points"]:
                voxels, coordinates, num_points, _ = self.voxel_generator.generate(
                    points
                )
                num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
                voxels_list.append(voxels)
                coordinates_list.append(coordinates)
                num_points_list.append(num_points)
                num_voxels_list.append(num_voxels)

            res["lidar"]["voxels"] = dict(
                voxels=voxels_list,
                coordinates=coordinates_list,
                num_points=num_points_list,
                num_voxels=num_voxels_list,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )
        else:
            voxels, coordinates, num_points, _ = self.voxel_generator.generate(
                res["lidar"]["points"], max_voxels=max_voxels
            )
            num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

            res["lidar"]["voxels"] = dict(
                voxels=voxels,
                coordinates=coordinates,
                num_points=num_points,
                num_voxels=num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

        double_flip = self.double_flip and (res["mode"] != 'train')

        if double_flip:
            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["yflip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["yflip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["xflip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["xflip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["double_flip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["double_flip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

        return res, info


def flatten(box):
    return np.concatenate(box, axis=0)


def merge_multi_group_label(gt_classes, num_classes_by_task):
    num_task = len(gt_classes)
    flag = 0

    for i in range(num_task):
        gt_classes[i] += flag
        flag += num_classes_by_task[i]

    return flatten(gt_classes)


@PIPELINES.register_module
class AssignLabel(object):
    def __init__(self, **kwargs):
        """Return CenterNet training labels like heatmap, height, offset"""
        assigner_cfg = kwargs["cfg"]
        self.out_size_factor = assigner_cfg.out_size_factor
        self.tasks = assigner_cfg.target_assigner.tasks
        self.gaussian_overlap = assigner_cfg.gaussian_overlap
        self._max_objs = assigner_cfg.max_objs
        self._min_radius = assigner_cfg.min_radius

    def __call__(self, res, info):

        if res["mode"] == "train" and res["lidar"]["annotations"] is None:
            return res, info

        max_objs = self._max_objs
        class_names_by_task = [t.class_names for t in self.tasks]
        num_classes_by_task = [t.num_class for t in self.tasks]

        # Calculate output featuremap size
        grid_size = res["lidar"]["voxels"]["shape"]
        pc_range = res["lidar"]["voxels"]["range"]
        voxel_size = res["lidar"]["voxels"]["size"]

        feature_map_size = grid_size[:2] // self.out_size_factor
        example = {}

        if res["mode"] == "train":
            gt_dict = res["lidar"]["annotations"]

            # reorganize the gt_dict by tasks
            task_masks = []
            flag = 0
            for class_name in class_names_by_task:
                task_masks.append(
                    [
                        np.where(
                            gt_dict["gt_classes"] == class_name.index(i) + 1 + flag
                        )
                        for i in class_name
                    ]
                )
                flag += len(class_name)

            task_boxes = []
            task_classes = []
            task_names = []
            flag2 = 0
            for idx, mask in enumerate(task_masks):
                task_box = []
                task_class = []
                task_name = []
                for m in mask:
                    task_box.append(gt_dict["gt_boxes"][m])
                    task_class.append(gt_dict["gt_classes"][m] - flag2)
                    task_name.append(gt_dict["gt_names"][m])
                task_boxes.append(np.concatenate(task_box, axis=0))
                task_classes.append(np.concatenate(task_class))
                task_names.append(np.concatenate(task_name))
                flag2 += len(mask)

            for task_box in task_boxes:
                # limit rad to [-pi, pi]
                task_box[:, -1] = box_np_ops.limit_period(
                    task_box[:, -1], offset=0.5, period=np.pi * 2
                )

            # print(gt_dict.keys())
            gt_dict["gt_classes"] = task_classes
            gt_dict["gt_names"] = task_names
            gt_dict["gt_boxes"] = task_boxes

            res["lidar"]["annotations"] = gt_dict

            draw_gaussian = draw_umich_gaussian

            hms, anno_boxs, inds, masks, cats = [], [], [], [], []

            for idx, task in enumerate(self.tasks):
                hm = np.zeros((len(class_names_by_task[idx]), feature_map_size[1], feature_map_size[0]),
                              dtype=np.float32)

                if res['type'] == 'NuScenesDataset':
                    # [reg, hei, dim, vx, vy, rots, rotc]
                    anno_box = np.zeros((max_objs, 10), dtype=np.float32)
                elif res['type'] == 'WaymoDataset':
                    anno_box = np.zeros((max_objs, 10), dtype=np.float32)
                else:
                    raise NotImplementedError("Only Support nuScene for Now!")

                ind = np.zeros((max_objs), dtype=np.int64)
                mask = np.zeros((max_objs), dtype=np.uint8)
                cat = np.zeros((max_objs), dtype=np.int64)

                num_objs = min(gt_dict['gt_boxes'][idx].shape[0], max_objs)

                for k in range(num_objs):
                    cls_id = gt_dict['gt_classes'][idx][k] - 1

                    w, l, h = gt_dict['gt_boxes'][idx][k][3], gt_dict['gt_boxes'][idx][k][4], \
                              gt_dict['gt_boxes'][idx][k][5]
                    w, l = w / voxel_size[0] / self.out_size_factor, l / voxel_size[1] / self.out_size_factor
                    if w > 0 and l > 0:
                        radius = gaussian_radius((l, w), min_overlap=self.gaussian_overlap)
                        radius = max(self._min_radius, int(radius))

                        # be really careful for the coordinate system of your box annotation.
                        x, y, z = gt_dict['gt_boxes'][idx][k][0], gt_dict['gt_boxes'][idx][k][1], \
                                  gt_dict['gt_boxes'][idx][k][2]

                        coor_x, coor_y = (x - pc_range[0]) / voxel_size[0] / self.out_size_factor, \
                                         (y - pc_range[1]) / voxel_size[1] / self.out_size_factor

                        ct = np.array(
                            [coor_x, coor_y], dtype=np.float32)
                        ct_int = ct.astype(np.int32)

                        # throw out not in range objects to avoid out of array area when creating the heatmap
                        if not (0 <= ct_int[0] < feature_map_size[0] and 0 <= ct_int[1] < feature_map_size[1]):
                            continue

                        draw_gaussian(hm[cls_id], ct, radius)

                        new_idx = k
                        x, y = ct_int[0], ct_int[1]

                        cat[new_idx] = cls_id
                        ind[new_idx] = y * feature_map_size[0] + x
                        mask[new_idx] = 1

                        if res['type'] == 'NuScenesDataset':
                            vx, vy = gt_dict['gt_boxes'][idx][k][6:8]
                            rot = gt_dict['gt_boxes'][idx][k][8]
                            anno_box[new_idx] = np.concatenate(
                                (ct - (x, y), z, np.log(gt_dict['gt_boxes'][idx][k][3:6]),
                                 np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)
                        elif res['type'] == 'WaymoDataset':
                            vx, vy = gt_dict['gt_boxes'][idx][k][6:8]
                            rot = gt_dict['gt_boxes'][idx][k][-1]
                            anno_box[new_idx] = np.concatenate(
                                (ct - (x, y), z, np.log(gt_dict['gt_boxes'][idx][k][3:6]),
                                 np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)
                        else:
                            raise NotImplementedError("Only Support Waymo and nuScene for Now")

                hms.append(hm)
                anno_boxs.append(anno_box)
                masks.append(mask)
                inds.append(ind) # coordinate index of each gt
                cats.append(cat)

            # used for two stage code
            boxes = flatten(gt_dict['gt_boxes'])
            classes = merge_multi_group_label(gt_dict['gt_classes'], num_classes_by_task)

            if res["type"] == "NuScenesDataset":
                gt_boxes_and_cls = np.zeros((max_objs, 10), dtype=np.float32)
            elif res['type'] == "WaymoDataset":
                gt_boxes_and_cls = np.zeros((max_objs, 10), dtype=np.float32)
            else:
                raise NotImplementedError()

            boxes_and_cls = np.concatenate((boxes,
                                            classes.reshape(-1, 1).astype(np.float32)), axis=1)
            num_obj = len(boxes_and_cls)
            assert num_obj <= max_objs
            # x, y, z, w, l, h, rotation_y, velocity_x, velocity_y, class_name
            boxes_and_cls = boxes_and_cls[:, [0, 1, 2, 3, 4, 5, 8, 6, 7, 9]]
            gt_boxes_and_cls[:num_obj] = boxes_and_cls

            example.update({'gt_boxes_and_cls': gt_boxes_and_cls})

            example.update({'hm': hms, 'anno_box': anno_boxs, 'ind': inds, 'mask': masks, 'cat': cats})

        res["lidar"]["targets"] = example

        return res, info

