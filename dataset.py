import os
import json
import time
import pickle

import numpy as np
from torch.utils.data import Dataset

from util.relation import relation_graph

#################################################################
# TODO: Rewrite preprocessing.py
#################################################################

def set_dataset(
        load_path: str,
        feature_path: str,
        vocab_list: list,
        ans_list: list,
        caption_id_path: str='',
        graph_path: str='',
        is_train: bool=False,
        is_val: bool=False,
        dataset_type: str='select',
    ):

    dataset_types = {
        'select': VQACaptionDataset,
        'all': VQACaptionAllDataset,
    }

    if is_train: dataset_name = 'train2014'
    elif is_val: dataset_name = 'val2014'

    load_path = os.path.join(load_path, dataset_name)
    feature_path = os.path.join(feature_path, dataset_name)
    graph_path = os.path.join(graph_path, dataset_name) if graph_path != '' else ''
    return dataset_types[dataset_type](
                                        load_path=load_path,
                                        feature_path=feature_path,
                                        dataset_name=dataset_name,
                                        vocab_list=vocab_list,
                                        ans_list=ans_list,
                                        caption_id_path=caption_id_path,
                                        graph_path=graph_path,
                                      )


class VQACaptionDataset(Dataset):
    """For each Q-A pair, select one caption."""
    def __init__(self,
                 load_path: str,
                 feature_path: str,
                 dataset_name: str,
                 vocab_list: list,
                 ans_list: list,
                 caption_id_path: str,
                 graph_path: str='',
    ):
        """
        load_path: path for VQA annotations
        feature_path: path for COCO image features
        dataset_name: train2014 / val2014
        vocab_list: GloVe vocabulary list
        ans_list: answer candidate list
        caption_id_path: path for caption ID for each Q-A pair
        graph_path: path for COCO graph (default = '' i.e. don't use graph)
        """
            
        t = time.time()
        print(f'loading {dataset_name} dataset...', end=' ')
        with open(os.path.join(f'{load_path}_questions.json')) as f:
            self.questions = json.load(f)['data']
        with open(os.path.join(f'{load_path}_answers.json')) as f:
            self.answers = json.load(f)['data']
        with open(os.path.join(f'{load_path}_all_captions.json')) as f:
            self.captions = json.load(f)
        with open(caption_id_path, 'rb') as f:
            self.caption_id = pickle.load(f)
            
        self.feature_path = feature_path
        self.graph_path = graph_path
        self.vocab_list = vocab_list
        self.ans_list = ans_list
        
        t = time.time() - t
        print(f'ready ({t:.2f} sec).')
        
    def __len__(self): return len(self.questions)
    
    def load_answer(self, index):
        answers = self.answers[index]
        output = np.array([0]*len(self.ans_list))
        for key, value in self.answers[index].items():
            output[int(key)] = min(value, 3)
        return np.divide(output, 3)
        
    def __getitem__(self, index):
        img_file = self.questions[index]['img_file']
        img = np.load(os.path.join(self.feature_path, img_file))
        img_id = str(int(img_file[-16:-4]))
        output = {
            'id': index,
            'img': img['x'],
            'q': np.array(self.questions[index]['q']),
            'a': self.load_answer(index),
            'c': np.array(self.captions[img_id]['c'][self.caption_id[index]])
        }
        if self.graph_path != '':
            output['graph'] = np.load(os.path.join(self.graph_path, img_file))['graph']
        return output

class VQACaptionAllDataset(VQACaptionDataset):
    """Q-A pairs with 5 captions"""
    def __init__(self,
                 load_path: str,
                 feature_path: str,
                 dataset_name: str,
                 vocab_list: list,
                 ans_list: list,
                 caption_id_path: str,
                 graph_path: str='',
    ):
        """
        load_path: path for VQA annotations
        feature_path: path for COCO image features
        dataset_name: train2014 / val2014
        vocab_list: GloVe vocabulary list
        ans_list: answer candidate list
        caption_id_path: path for caption ID for each Q-A pair
        graph_path: path for COCO graph (default = '' i.e. don't use graph)
        """

        pass
        # TODO