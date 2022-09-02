## Training ProposalContrast on Waymo

### Training Scripts. 

* Use the following command to pretrain the 3D models. The ```pretrain_description``` and ```finetune_description``` specify the task names. The models will be saved to ```~/ProposalContrast/output/waymo/pretrain```. 
We would recommand distributed training with 8*32G GPUs for better results. 
```bash
# pretrain and fientune VoxelNet
bash tools/scripts/ssl_pretrain_voxel.sh pretrain_description finetune_description
```

```bash
# pretrain and fientune PillarNet
bash tools/scripts/ssl_pretrain_pillar.sh pretrain_description finetune_description
```
Note that the fine-tuning command is also contained in above scripts. 

* Use the following command to obtain the baseline results that are trained from random initialization. TASK_DESC indicates the task name. 
```bash
# train VoxelNet baseline
bash tools/scripts/train_voxel_sup.sh TASK_DESC
```

```bash
# train PillarNet baseline
bash tools/scripts/train_pillar_sup.sh TASK_DESC
```

### Evaluation Scripts.
Use this command to inference a model:
```bash
python ./tools/dist_test.py CONFIG_PATH --work_dir output/waymo/finetune/xxx --checkpoint output/waymo/finetune/xxx/latest.pth --speed_test 
```

This will generate a `my_preds.bin` file in the output directory. You can create submission to Waymo server using waymo-open-dataset code by following the instructions [here](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md).  

If you want to do local evaluation, please follow the waymo instructions [here](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md).

```bash
bazel-bin/waymo_open_dataset/metrics/tools/compute_detection_metrics_main ~/output/waymo/finetune/xxx/detection_pred.bin gt.bin
```
