import os
import json
import random
import argparse

import numpy as np
from tqdm import tqdm
# import nltk
# from nltk.tokenize import RegexpTokenizer
# from nltk.stem import WordNetLemmatizer

import torch
import torch.nn as nn

from util.utils import get_vocab_list
from modules.modules import PretrainedWordEmbedding

############################################
# TODO: Reconstruct
# Implement the caption selection strategy
############################################

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--vqa_path', type=str, default='../annot/VQA_annotations')
    parser.add_argument('--coco_path', type=str, default='../annot/annotations')
    parser.add_argument('--vocab_path', type=str, default='../data/vocab_list.txt')
    parser.add_argument('--ans_vocab_path', type=str, default='../data/answer_candidate.txt')
    parser.add_argument('--feature_path', type=str, default='../COCO_feature_36')
    parser.add_argument('--dataset_type', type=str, default='train2014')
    parser.add_argument('--save_path', type=str, default='../annot')
    parser.add_argument('--c_len', type=int, default=20)
    parser.add_argument('--q_len', type=int, default=10)
    parser.add_argument('--save_q', type=bool, default=False)
    parser.add_argument('--save_a', type=bool, default=False)
    parser.add_argument('--save_c', type=bool, default=False)
    parser.add_argument('--glove_path', type=str, default='')
    
    args = parser.parse_args()    

    return args

def preprocessing(
    vqa_path: str,
    coco_path: str,
    vocab_path: str,
    ans_vocab_path: str,
    feature_path: str,
    dataset_type: str,
    save_path: str = 'annot',
    c_len: int = 20,
    q_len: int = 10,
    save_q: bool = False,
    save_a: bool = False,
    save_c: bool = False,
    glove_path: str = '',
):
    """
    Dataset preprocessing.
    vqa_path: path for original VQA dataset
    coco_path: path for original COCO Captions dataset
    vocab_path: path for vocabulary
    ans_vocab_path: path for answer candidate vocabulary
    feature_path: path for extracted COCO image features
    dataset_type: train2014/val2014 (default = train2014)
    save_path: path for saving preprocessed files (default = annot)
    c_len: the maximal length of captions (default = 20)
    q_len: the maximal length of questions (default = 10)
    save_q/save_a/save_c: save questions/answers/captions or not
    """
    print('q:', save_q)
    print('a:', save_a)
    print('c:', save_c)

    # Check if save_path exist
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Setup tools
    vocab_list = get_vocab_list(vocab_path)
    ans_list = get_vocab_list(ans_vocab_path)
    # lemmatizer = WordNetLemmatizer()
    # tokenizer = RegexpTokenizer(r'\w+')

    # def get_tokens(sentence, is_cap=False):
    #     # turn into lower case, tokenize and remove symbols
    #     words = tokenizer.tokenize(sentence.lower())
    #     words = [lemmatizer.lemmatize(word, pos='n') for word in words]
    #     if is_cap:
    #         words.insert(0, '<start>')
    #         words.append('<end>')

    #     tokens = []
    #     for word in words:
    #         if word in vocab_list: token = vocab_list.index(word)
    #         else: token = vocab_list.index('<oov>')
    #         tokens.append(token)
    #     return ' '.join(words), tokens

    def get_tokens(sentence, is_cap=False):
        sentence = sentence.lower()
        for c in [' \'', '\' ', ' \"', '\" ', '\n']:
            sentence = sentence.replace(c, ' ')
        for c in '.,?':
            sentence = sentence.replace(c, '')
        sentence = sentence.replace('\'s', ' \'s')
        words = [i for i in sentence.split() if len(i) > 0]
        
        if is_cap:
            words.insert(0, '<start>')
            words.append('<end>')

        tokens = []
        for word in words:
            if word in vocab_list: token = vocab_list.index(word)
            else: token = vocab_list.index('<oov>')
            tokens.append(token)
        return ' '.join(words), tokens
        


    def padding(tokens, max_l):
        l = min(len(tokens), max_l)
        if l < max_l:
            tokens.extend([vocab_list.index('<pad>')] * (max_l - l))
        else:
            tokens = tokens[:l]
        return tokens, l

    def save_file(file_name, desc, data_type, data):
        with open(file_name, 'w') as f:
            f.write(json.dumps({'description': desc, 
                                'data_type': data_type,
                                'data': data}))

    #########################################################################
    # Read VQA annotation dataset
    # Keep the questions of certain answer type
    with open(os.path.join(vqa_path, f'v2_mscoco_{dataset_type}_annotations.json')) as f:
        a_json = json.load(f)['annotations']
        print('Load answer json file.')
    
    a_data = []
    ans_type = {'yes/no': [], 'number': [], 'other': []}
    for i in tqdm(range(len(a_json)), desc='answer'):
        ans_type[a_json[i]['answer_type']].append(i)
        
        if save_a:
            image_id = a_json[i]['image_id']
            answers = []
            for a in a_json[i]['answers']:
                answers.append(a['answer'])
            ans_dict = {}
            for a in set(answers):
                if a in ans_list: ans_dict[ans_list.index(a)] = answers.count(a)
            a_data.append(ans_dict)

    if save_a:
        # Save answer dataset
        save_file(file_name=os.path.join(save_path,f'{dataset_type}_answers.json'),
                desc='This is VQA v2.0 answers dataset.',
                data_type=dataset_type, data=a_data
        )
        print('answer dataset saved.')

        # Save IDs for different answer types
        with open(os.path.join(save_path,f'{dataset_type}_answer_type.json'), 'w') as f:
            f.write(json.dumps(ans_type))

    #########################################################################
    # Read VQA question dataset
    # Save the image IDs in order to load corresponding captions
    with open(os.path.join(vqa_path, f'v2_OpenEnded_mscoco_{dataset_type}_questions.json')) as f:
        q_json = json.load(f)['questions']
        print('Load question json file.')

    image_ids = []
    q_data = []
    for i in tqdm(range(len(q_json)), desc='question'):
        image_id = q_json[i]['image_id']
        image_ids.append(image_id)
        
        if save_q:
            words, tokens = get_tokens(q_json[i]['question'])
            tokens, _ = padding(tokens, q_len)
            q_data.append({
                'img_file': f'COCO_{dataset_type}_{str(image_id).zfill(12)}.npz',
                'q_word': words,
                'q': tokens,
            })

    # Save question dataset
    if save_q:
        save_file(file_name=os.path.join(save_path,f'{dataset_type}_questions.json'),
                desc='This is VQA v2.0 questions dataset.',
                data_type=dataset_type, data=q_data
        )
        print('question dataset saved.')

    #########################################################################
    # Read COCO Captions dataset
    with open(os.path.join(coco_path, f'captions_{dataset_type}.json')) as f:
        c_json = json.load(f)['annotations']
        print('Load caption json file.')
    
    # Store captions based on image ID
    captions = {}
    for c in tqdm(c_json):
        image_id = c['image_id']
        if image_id not in captions:
            captions[image_id] = []
        captions[image_id].append(c['caption'])

    if save_c:
        cap_token = {}
        for image_id in tqdm(captions):
            cap_token[image_id] = {'c_word':[], 'c':[], 'cap_len':[]}
            for caption in captions[image_id]:
                words, tokens = get_tokens(caption, is_cap=True)
                tokens, cap_len = padding(tokens, c_len)
                cap_token[image_id]['c_word'].append(words)
                cap_token[image_id]['c'].append(tokens)
                cap_token[image_id]['cap_len'].append(cap_len)
        # Save answer dataset
        save_file(file_name=os.path.join(save_path,f'{dataset_type}_captions.json'),
                desc='This is COCO Captions dataset.',
                data_type=dataset_type, data=c_data
        )
        print('caption dataset saved.')

    return


if __name__ == '__main__':
    args = parse_args()

    preprocessing(
        vqa_path=args.vqa_path,
        coco_path=args.coco_path,
        vocab_path=args.vocab_path,
        ans_vocab_path=args.ans_vocab_path,
        feature_path=args.feature_path,
        dataset_type=args.dataset_type,
        save_path=args.save_path,
        c_len=args.c_len,
        q_len=args.q_len,
        save_q=args.save_q,
        save_a=args.save_a,
        save_c=args.save_c,
        select_c=args.select_c
    )