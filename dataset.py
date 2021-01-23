import os
import json
import time

import numpy as np
from torch.utils.data import Dataset

def set_dataset(load_dataset, feature_path, vocab_list, ans_list, dataset_type='vqa', is_train=False, is_val=False):
    dataset_types = {
        'vqa':VQADataset,
        'vqac': VQACaptionDataset
    }

    if is_train:
        dataset_name = 'train2014'
    elif is_val:
        dataset_name = 'val2014'

    load_dataset = os.path.join(load_dataset, dataset_name)
    feature_path = os.path.join(feature_path, dataset_name)
    return dataset_types[dataset_type](load_dataset, feature_path, vocab_list, ans_list)


class VQADataset(Dataset):
    def __init__(self, load_dataset, feature_path, vocab_list, ans_list):
        t = time.time()
        self.questions = self.load_dataset(load_dataset, 'questions')
        self.answers = self.load_dataset(load_dataset, 'answers')
        
        self.vocab_list = vocab_list
        self.ans_list = ans_list
        self.feature_path = feature_path
        t = time.time() - t
        print(f'ready ({t:.2f} sec).')
    
    def __len__(self):
        return len(self.questions)
    
    def load_dataset(self, path, dataset_type):
        with open(f'{path}_{dataset_type}.json') as f:
            dataset = json.load(f)['data']
        return dataset
    
    def load_answer(self, index):
        answers = self.answers[index]
        output = np.array([0]*len(self.ans_list))
        for key, value in answers.items():
            key = int(key)
            output[key] = value
            
        return np.divide(output, 3).tolist()
            
    def __getitem__(self, index):
        return {
            'img': np.load(self.feature_path + '/' + self.questions[index]['img_file'])['x'],
            'q': np.array(self.questions[index]['q']),
            'a': np.array(self.load_answer(index)),
        }


class VQACaptionDataset(VQADataset):
    def __init__(self, load_dataset, feature_path, vocab_list, ans_list):
        t = time.time()
        print('loading dataset...', end=' ')
        self.questions = self.load_dataset(load_dataset, 'questions')
        self.answers = self.load_dataset(load_dataset, 'answers')
        self.captions = self.load_dataset(load_dataset, 'captions')
        
        self.vocab_list = vocab_list
        self.ans_list = ans_list
        self.feature_path = feature_path
        t = time.time() - t
        print(f'dataset ready ({t:.2f} sec).')
    
    def __len__(self):
        return len(self.questions)
            
    def __getitem__(self, index):
        return {
            'img': np.load(self.feature_path + '/' + self.questions[index]['img_file'])['x'],
            'q': np.array(self.questions[index]['q']),
            'c': np.array(self.captions[index]['c']),
            'l_c': self.captions[index]['l_c'],
            'a': np.array(self.load_answer(index)),
        }