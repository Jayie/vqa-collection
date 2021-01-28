import os
import json
import time

import numpy as np
from torch.utils.data import Dataset

def set_dataset(load_dataset, feature_path, vocab_list, ans_list, dataset_type='vqa', ans_type='', is_train=False, is_val=False):
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
    return dataset_types[dataset_type](load_dataset, feature_path, vocab_list, ans_list, ans_type)


class VQADataset(Dataset):
    def __init__(self, load_dataset, feature_path, vocab_list, ans_list, ans_type=''):
        t = time.time()
        print('loading dataset...', end=' ')

        # If ans_type is not empty: load the question IDs of certain answer type
        self.q_id = None
        if ans_type != '':
            with open(f'{load_dataset}_answer_type.json') as f: self.q_id = json.load(f)[ans_type]
        
        self.questions = self.load_dataset(load_dataset, 'questions')
        self.answers = self.load_dataset(load_dataset, 'answers')
        
        self.vocab_list = vocab_list
        self.ans_list = ans_list
        self.feature_path = feature_path
        t = time.time() - t
        print(f'ready ({t:.2f} sec).')
    
    def __len__(self):
        if self.q_id == None:
            return len(self.questions)
        return len(self.q_id)
    
    def load_dataset(self, path, dataset_type):
        with open(f'{path}_{dataset_type}.json') as f:
            dataset = json.load(f)['data']
        
        if self.q_id == None: return dataset
        output = []
        for i in self.q_id:
            output.append(dataset[i])
        return output
    
    def load_answer(self, index):
        answers = self.answers[index]
        output = np.array([0]*len(self.ans_list))
        for key, value in answers.items():
            key = int(key)
            output[key] = value
            
        return np.divide(output, 3).tolist()
            
    def __getitem__(self, index):
        return {
            'id': index,
            'img': np.load(os.path.join(self.feature_path, self.questions[index]['img_file']))['x'],
            'q': np.array(self.questions[index]['q']),
            'a': np.array(self.load_answer(index)),
        }


class VQACaptionDataset(VQADataset):
    def __init__(self, load_dataset, feature_path, vocab_list, ans_list, ans_type=''):
        t = time.time()
        print('loading dataset...', end=' ')

        # If ans_type is not empty: load the question IDs of certain answer type
        self.q_id = None
        if ans_type != '':
            with open(f'{load_dataset}_answer_type.json') as f: self.q_id = json.load(f)[ans_type]
        
        self.questions = self.load_dataset(load_dataset, 'questions')
        self.answers = self.load_dataset(load_dataset, 'answers')
        self.captions = self.load_dataset(load_dataset, 'captions')
        
        self.vocab_list = vocab_list
        self.ans_list = ans_list
        self.feature_path = feature_path
        t = time.time() - t
        print(f'dataset ready ({t:.2f} sec).')
            
    def __getitem__(self, index):
        return {
            'id': index,
            'img': np.load(os.path.join(self.feature_path, self.questions[index]['img_file']))['x'],
            'q': np.array(self.questions[index]['q']),
            'c': np.array(self.captions[index]['c']),
            'cap_len': self.captions[index]['cap_len'],
            'a': np.array(self.load_answer(index)),
        }