#!/bin/bash

CRTDIR='/home/junbo/repository/ProposalContrast'

PRETRAIN_DESC1=$1
FINETUNE_DESC2=$2

DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`
OUT_DIR=$CRTDIR/output/waymo/pretrain

PRE_WORK_DIR=$OUT_DIR/PILLAR_$PRETRAIN_DESC1\_$DATE_WITH_TIME

if [ ! $PRETRAIN_DESC1 ]
then
 echo "PRETRAIN_DESC1 must be specified."
 echo "Usage: tools/scripts/ssl_pretrain_pillar.sh pretrain_description finetune_description"
 exit $E_ASSERT_FAILED
fi

export PYTHONPATH=$PYTHONPATH:$CRTDIR
PORT=$((8000 + RANDOM %57535))

# pre-train
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=$PORT  ./tools/train.py  configs/waymo/pp/ssl_pretrain_pillarnet.py  --work_dir=$PRE_WORK_DIR

# fine-tune
bash tools/scripts/train_pillar_sup.sh $FINETUNE_DESC2 $PRE_WORK_DIR/latest.pth