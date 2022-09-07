# _ProposalContrast:_ Unsupervised Pre-training for LiDAR-based 3D Object Detection
This repository contains the PyTorch implementation of the ECCV'2022 paper, [*ProposalContrast: Unsupervised Pre-training for LiDAR-based 3D Object Detection*](https://arxiv.org/pdf/2207.12654.pdf). This work addresses the unsupervised pre-training of 3D backbones via proposal-wise contrastive learning in the context of autonomous driving.

## Updates

* [2022.8.31] Code of ProposalContrast is released. We now support the pre-traning of [VoxeNet](https://pdfs.semanticscholar.org/5125/a16039cabc6320c908a4764f32596e018ad3.pdf) and [PillarNet](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lang_PointPillars_Fast_Encoders_for_Object_Detection_From_Point_Clouds_CVPR_2019_paper.pdf) based on the Waymo dataset.



## Abstract
Existing approaches for unsupervised point cloud pre-training are constrained to either scene-level or point/voxel-level instance discrimination. Scene-level methods tend to lose local details that are crucial for recognizing the road objects, while point/voxel-level methods inherently suffer from limited receptive field that is incapable of perceiving large objects or context environments. Considering region-level representations are more suitable for 3D object detection, we devise a new unsupervised point cloud pre-training framework, called ProposalContrast, that learns robust 3D representations by contrasting region proposals. Specifically, with an exhaustive set of region proposals sampled from each point cloud, geometric point relations within each proposal are modeled for creating expressive proposal representations. To better accommodate 3D detection properties, ProposalContrast optimizes with both inter-cluster and inter-proposal separation, i.e., sharpening the discriminativeness of proposal representations across semantic classes and object instances. The generalizability and transferability of ProposalContrast are verified on various 3D detectors (i.e., PV-RCNN, CenterPoint, PointPillars and PointRCNN) and datasets (i.e., KITTI, Waymo and ONCE).

## Citation
If you find our project is helpful for you, please cite:


    @article{yin2022proposal,
      title={ProposalContrast: Unsupervised Pre-training for LiDAR-based 3D Object Detection},
      author={Yin, Junbo and Zhou, Dingfu and Zhang, Liangjun and Fang, Jin and Xu, Cheng-Zhong and Shen, Jianbing and Wang, Wenguan},
      booktitle={ECCV},
      year={2022}
    }
    
## Main Results

#### 3D Detection on Waymo validation set.
* We provide 3D detection results of the fine-tuned [PillarNet](configs/waymo/pp/waymo_centerpoint_pp.py) and [VoxelNet](configs/waymo/voxelnet/waymo_centerpoint_voxelnet_1x.py) following the OpenPCDet learning schedule with 20% data (~32k frames).
* All the models are trained with 8 Tesla V100 GPUs.

| Model                      | Paradigm      | Veh_L2 | Ped_L2 | Cyc_L2 | Overall  |  
|----------------------------|-------------|--------|--------|--------|-------|
| CenterPoint (PillarNet)    | Scratch     | 60.67  | 51.55  | 55.28  | 55.83 |  
| ProposalContrast (PillarNet) | Fine-tuning | 63.03    | 53.16    | 57.31    | 57.83   | 

| Model                       | Paradigm      | Veh_L2 | Ped_L2 | Cyc_L2 | MAPH |  
|-----------------------------|-------------|--------|--------|--------|------|
| CenterPoint (VoxelNet)      | Scratch     | 63.10  | 58.66  | 66.54  | 62.77 |  
| ProposalContrast (VoxelNet) | Fine-tuning | 64.14  | 60.07  | 67.31  | 63.84 | 

#### Data-efficient 3D Detection on Waymo.
* We uniformly downsample the Waymo training data to 5%, 10%, 20%, 50%, and report the results of [CenterPoint (VoxelNet)](configs/waymo/voxelnet/waymo_centerpoint_voxelnet_1x.py) based on 1x learning schedule.
* All the models are trained with 8 Tesla V100 GPUs.

| Model (VoxelNet)            | Paradigm                | Veh_L2 | Ped_L2 | Cyc_L2 | Overall  |  
|-----------------------------|-----------------------|--------|--------|--------|-------|
| CenterPoint      | 5%, Scratch      | 41.56  | 34.34  | 44.46  | 40.12 |  
| ProposalContrast | 5%, Fine-tuning   |  50.66 | 43.69  | 54.46  | 49.60 |   
| CenterPoint      | 10%, Scratch      | 52.59  | 44.28  | 55.97  | 50.95 |  
| ProposalContrast | 10%, Fine-tuning   | 57.43  | 51.26  | 59.77  | 56.15 |   
| CenterPoint      | 20%, Scratch     | 58.43  | 51.02  | 62.29  | 57.25 |  
| ProposalContrast | 20%, Fine-tuning  | 61.54  | 56.05  | 64.14  | 60.58 |   
| CenterPoint      | 50%, Scratch     | 62.87  | 58.20  | 66.60  | 62.56 |  
| ProposalContrast | 50%, Fine-tuning  | 63.96  | 60.16  | 67.49  | 63.87 |   


## Use ProposalContrast

### Installation

Please refer to [INSTALL](docs/INSTALL.md) to build the required libraries. Our project supports both SpConv v1 and SpConv v2.

### Data Preparation
Currently, this repo supports the pre-training and fine-tuning on the Waymo 3D object detection dataset. Please prepare the dataset according to [WAYMO](docs/WAYMO.md).

### Training and Evaluation
 We evaluate the unsupervised pre-training performance of the 3D models in the context of LiDAR-based 3D object detection. The scripts for pre-training, fine-tuning and evluation can be found in [RUN_MODEL](docs/RUN_MODEL.md). We currently support the 3D models like [VoxelNet](configs/waymo/voxelnet/waymo_centerpoint_voxelnet.py) and [PillarNet](configs/waymo/pp/waymo_centerpoint_pp.py).

## License

This project is released under MIT license, as seen in [LICENSE](LICENSE).




## Acknowlegement
Our project is partially supported by the following codebase. We would like to thank for their contributions.

* [CenterPoint](https://github.com/tianweiy/CenterPoint)
* [DeepCluster](https://github.com/facebookresearch/deepcluster)
* [PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
* [PointTransformer](https://github.com/lucidrains/point-transformer-pytorch)
* [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
