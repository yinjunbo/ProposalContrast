import copy
from pathlib import Path
import pickle

import fire, os

from det3d.datasets.nuscenes import nusc_common as nu_ds
from det3d.datasets.utils.create_gt_database import create_groundtruth_database
from det3d.datasets.waymo import waymo_common as waymo_ds


def waymo_data_prep(root_path, split, interval=1, nsweeps=1):
    '''
    waymo_data_prep
    --root_path=data/Waymo/
    --interval=10
    --split train_D20
    --nsweeps=1
    '''
    dataset_type = ['train', 'pcdet_train', 'val', 'test']
    assert split in dataset_type, "Only support {} now.".format(dataset_type)
    version=None
    if 'pcdet' in split:
        version = 'convert_info_to_pcdet'
        split = split.split('_')[1]

    if interval==1:
        waymo_ds.create_waymo_infos(root_path, version=version, interval=1, split=split, nsweeps=nsweeps)

    if 'train' in split and not version == 'convert_info_to_pcdet':
        info_name = "infos_{}_{:02d}sweeps_filter_zero_gt.pkl".format(version, nsweeps) if version is not None else "infos_train_{:02d}sweeps_filter_zero_gt.pkl".format(nsweeps)
        if interval>1:
            version='D'+str(interval)
        create_groundtruth_database(
            "WAYMO",
            root_path,
            Path(root_path) / info_name,
            used_classes=['VEHICLE', 'CYCLIST', 'PEDESTRIAN'],
            version=version,
            interval=interval,
            nsweeps=nsweeps
        )
    

if __name__ == "__main__":
    fire.Fire()
