import os.path as osp

import torch

from ...utils import master_only
from .base import LoggerHook
import numpy as np

class TensorboardLoggerHook(LoggerHook):
    def __init__(self, log_dir=None, interval=10, ignore_last=True, reset_flag=True):
        super(TensorboardLoggerHook, self).__init__(interval, ignore_last, reset_flag)
        self.log_dir = log_dir

    @master_only
    def before_run(self, trainer):
        if torch.__version__ >= "1.1":
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    "the dependencies to use torch.utils.tensorboard "
                    "(applicable to PyTorch 1.1 or higher)"
                )
        else:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    'the dependencies to use torch.utils.tensorboard '
                    '(applicable to PyTorch 1.1 or higher)')

        if self.log_dir is None:
            self.log_dir = osp.join(trainer.work_dir, "tf_logs")
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, trainer):
        for var in trainer.log_buffer.output:
            if var in ["time", "data_time", "loc_loss_elem"]:
                continue
            tag = "{}/{}".format(var, trainer.mode)
            record = trainer.log_buffer.output[var]
            if isinstance(record, str):
                self.writer.add_text(tag, record, trainer.iter)
            else:
                if isinstance(record, list):
                    record = np.mean(record)
                self.writer.add_scalar(
                    tag, record, trainer.iter
                )
            self.writer.add_scalar('learning_rate',
                                   trainer.current_lr()[0], trainer.iter)
    @master_only
    def after_run(self, trainer):
        self.writer.close()
