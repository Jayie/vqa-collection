import os
import time
import random

import numpy as np
import torch

def spatial_relation(a, b):
    """
    Calculate the spatial relation between 2 bounding boxes, and return the type number of relation.
    The definition of types of spatial relations <a-b> is described in "Exploring Visual Relationship for Image Captioning".
    1: a is inside b
    2: a is coverd by b
    3: a overlaps b (IoU >= 0.5)
    4~11: 
    """
    # get IoU region
    iou = np.array([
        max(a[0], b[0]), max(a[1], b[1]), # (x0, y0)
        min(a[2], b[2]), min(a[3], b[3])  # (x1, y1)
    ])

    if iou == b: return 1, 2 # If IoU == b: b is inside a
    elif iou == a: return 2, 1 # Else if IoU == a: a is covered by b

    # Else if IoU >=0.5: a and b overlap

    # Else: compute the distance and angle between a and b
    # TODO

def get_vocab_list(vocab_path):
    with open(vocab_path, encoding='utf-8') as f:
        vocab_list = f.read().split('\n')
    return vocab_list

def random_seed(seed=10):
    """
    set random seed
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if using multi-GPU
    torch.backends.cudnn.deterministic = True

def set_device():
    """
    set device as 'cuda' if available, otherwise 'cpu'
    """

    # Use cuda if available
    if torch.cuda.is_available():
        device = 'cuda'
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    else:
        device = 'cpu'
    return device


class Logger():
    def __init__(self, exp_name, log_name='log.txt'):
        save_path = os.path.join('checkpoint', exp_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        t = time.strftime("%y%m%d-%H-%M-%S_", time.localtime()) # timestamp
        self.log_file = open(os.path.join(save_path, t + log_name), 'w+')
        self.exp_name = exp_name
    #     self.infos = {}

    # def append(self, key, value):
    #     values = self.infos.setdefault(key, [])
    #     values.append(value)

    # def log(self, extra_msg=''):
    #     msg = [extra_msg]
    #     for key, values in self.infos.items():
    #         msg.append(f'{key} {np.mean(values):.6f}')
    #     self.log_file.write('\n'.join(msg) + '\n')
    #     self.log_file.flush()
    #     self.infos = {}
    #     return msg

    def write(self, msg):
        self.log_file.write(time.strftime("%y%m%d-%H:%M:%S ", time.localtime())) # timestamp
        self.log_file.write(msg+'\n')
        self.log_file.flush()
