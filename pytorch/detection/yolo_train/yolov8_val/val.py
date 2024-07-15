# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


"""
    函数
"""


def yolov8_val(dataloader, model, compute_loss, save_dir="./val_runs/train/exp_l", nc=80):
    device = model.device
    model.eval()
    n_batches = len(dataloader)
    desc = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'Box(P', 'R', 'mAP50', 'mAP50-95)')
    TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'
    bar = tqdm(dataloader, desc, n_batches, bar_format=TQDM_BAR_FORMAT)
    for batch_i, batch in enumerate(bar):
        ...


if __name__ == '__main__':
    ...
