import os
import time
import random

import numpy as np
import torch

def get_vocab_list(vocab_path):
    with open(vocab_path) as f:
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

        self.log_file = open(os.path.join(save_path, log_name), 'w+')
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
        self.log_file.write(time.strftime("%y%m%d %H:%M:%S ", time.localtime())) # time
        self.log_file.write(msg+'\n')
        self.log_file.flush()