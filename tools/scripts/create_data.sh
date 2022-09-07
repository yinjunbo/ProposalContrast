#!/bin/bash

CRTDIR='/home/junbo/repository/ProposalContrast'
export PYTHONPATH=$PYTHONPATH:$CRTDIR

python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split train
python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split val
python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split pcdet_train

#python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split train --interval=20
#python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split train --interval=10
#python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split train --interval=5
#python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split train --interval=2