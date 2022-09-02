import itertools
import logging

from det3d.utils.config_tool import get_downsample_factor


# dataset settings
dataset_type = "WaymoDataset"
data_root = "data/Waymo/"
nsweeps = 1

tasks = [
    dict(num_class=1, class_names=['SSL']),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training settings
target_assigner = dict(
    tasks=tasks,
)

voxel_generator = dict(
    range=[-75.2, -75.2, -2, 75.2, 75.2, 4],
    voxel_size=[0.1, 0.1, 0.15],
    max_points_in_voxel=5,
    max_voxel_num=120000,
)

# model settings
model = dict(
    type="VoxelNet",
    pretrained=None,
    reader=dict(
        type="VoxelFeatureExtractorV3",
        num_input_features=5 if nsweeps==1 else 6,
    ),
    backbone=dict(
        type="SpMiddleResNetFHD", num_input_features=5 if nsweeps==1 else 6, ds_factor=8),
    neck=dict(
        type="RPN",
        layer_nums=[5, 5],
        ds_layer_strides=[1, 2],
        ds_num_filters=[128, 256],
        us_layer_strides=[1, 2],
        us_num_filters=[256, 256],
        num_input_features=256,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=None,
    embed_neck=dict(
        type="SelfSupNeck",
        mode="joint_instance_class_embed",
        samples_num=2048,
        embed_layer=[512, 256, 128],
        radii=1.0,
        npoints=16,
        voxel_cfg=voxel_generator
    ) ,
    ssl_head=dict(type='SSLHead', temperature=0.1, tasks=tasks),
)

train_cfg = None
test_cfg = None

train_preprocessor = dict(
    mode="train",
    shuffle_points=False,
    global_rot_noise=[-3.14159265, 3.14159265],
    global_scale_noise=[0.8, 1.25],
    db_sampler=None,
    class_names=class_names,
    ssl_mode=True,
    voxel_cfg=voxel_generator,
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
    ssl_mode=True,
    voxel_cfg=voxel_generator,
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="Reformat"),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="Reformat"),
]

# we ignore the annotations during pre-training
train_anno = data_root+"infos_train_0{}sweeps_filter_zero_gt.pkl".format(nsweeps)
val_anno = data_root+"infos_val_0{}sweeps_filter_zero_gt.pkl".format(nsweeps)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=6,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=train_pipeline,
        ssl_mode=True,
        load_interval=4
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        test_mode=True,
        ann_file=val_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)


optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
lr_config = dict(
    type="one_cycle", lr_max=0.003, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.1,
)

checkpoint_config = dict(interval=10)
log_config = dict(
    interval=20,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type='TensorboardLoggerHook')
    ],
)

total_epochs = 50
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None
resume_from = None
workflow = [('train', 1)]

