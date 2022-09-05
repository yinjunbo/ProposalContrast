## Data Preparation for Running ProposalContrast

### Prerequisite 

- Follow [INSTALL.md](INSTALL.md) to install all required libraries. 
- Tensorflow 
- Waymo-open-dataset devkit

```bash
conda activate proposalcontrast 
pip install waymo-open-dataset-tf-2-4-0==1.3.1 
```

### Prepare data

#### Download Waymo data and organise as follows

```
# For Waymo Dataset         
└── WAYMO_DATASET_ROOT
       ├── tfrecord_training       
       ├── tfrecord_validation   
       ├── tfrecord_testing 
```

Convert the tfrecord data to pickle files.

```bash
# train set 
CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py --tfrecord_path 'WAYMO_DATASET_ROOT/tfrecord_training/segment-*.tfrecord'  --root_path 'WAYMO_DATASET_ROOT/train/'

# validation set 
CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py --tfrecord_path 'WAYMO_DATASET_ROOT/tfrecord_validation/segment-*.tfrecord'  --root_path 'WAYMO_DATASET_ROOT/val/'

# testing set 
CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py --tfrecord_path 'WAYMO_DATASET_ROOT/tfrecord_testing/segment-*.tfrecord'  --root_path 'WAYMO_DATASET_ROOT/test/'
```
Create a symlink to the dataset root 
```bash
mkdir data && cd data
ln -s WAYMO_DATASET_ROOT Waymo
```
Remember to change the WAYMO_DATASET_ROOT to the actual path in your system. 

#### Download Waymo road plane files.
We compute the road plane via the [RANSAC](https://pypi.org/project/pyransac3d/) algorithm in order to reduce the samples on the roads. Please refer to Baidu Cloud <https://pan.baidu.com/s/1ra8GmOR1maG7tM2Kz6Cd8Q> with access code ```eko8``` to download the computed road plane. Please unzip the file and place it at ```~/data/Waymo/train/lidar_ground```.
 
#### Create info files

```bash
# Prepare the pre-training dataset 
python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split train 

# Prepare the training (fine-tuning) dataset, where we align the training samples with OpenPCDet
python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split pcdet_train

# Prepare the validation dataset 
python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split val 

# For data-efficient 3D object detection (optional, for the puerpose of making downsampled gt database)
python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split train --interval=100
python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split train --interval=20
python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split train --interval=10
python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split train --interval=2


```
These scripts can be also found in ```~/ProposalContrast/scripts/create_data.sh```. The final data infos are as follows:
```
└── ProposalContrast
       └── data    
              └── Waymo 
                     ├── tfrecord_training       
                     ├── tfrecord_validation
                     ├── train <-- all training frames and annotations 
                     ├── val   <-- all validation frames and annotations 
                     ├── test   <-- all testing frames and annotations 
                     ├── dbinfos_train_1sweeps_withvelo.pkl
                     ├── gt_database_1sweeps_withvelo/
                     ├── infos_pcdet_train_01sweeps_filter_zero_gt.pkl
                     ├── infos_train_01sweeps_filter_zero_gt.pkl
                     ├── infos_val_01sweeps_filter_zero_gt.pkl
                 

```
