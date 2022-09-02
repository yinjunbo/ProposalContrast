#!/bin/bash

CRTDIR='/home/junbo/repository/ProposalContrast'

TASK_DESC=$1
PRETRAIN_MODEL=$2

DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`
OUT_DIR=$CRTDIR/output/waymo/finetune/pillar
PILLAR_WORK_DIR=$OUT_DIR/PILLAR_$TASK_DESC\_$DATE_WITH_TIME

if [ ! $TASK_DESC ]
then
 echo "TASK_DESC must be specified."
 echo "Usage: bash tools/scripts/train_voxel_sup.sh task_description pretrain_model(optional)"
 exit $E_ASSERT_FAILED
fi

export PYTHONPATH=$PYTHONPATH:$CRTDIR
PORT=$((8000 + RANDOM %57535))

# train
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=$PORT  ./tools/train.py  configs/waymo/pp/waymo_centerpoint_pp.py --work_dir=$PILLAR_WORK_DIR --pretrained_model=$PRETRAIN_MODEL

# eval
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=$PORT  ./tools/dist_test.py  configs/waymo/pp/waymo_centerpoint_pp.py  --work_dir=$PILLAR_WORK_DIR --checkpoint=$PILLAR_WORK_DIR/latest.pth --speed_test

cd $CRTDIR/waymo-od
bazel-bin/waymo_open_dataset/metrics/tools/compute_detection_metrics_main $PILLAR_WORK_DIR/detection_pred.bin gt.bin 2>&1 |tee $PILLAR_WORK_DIR/eval.log

