import os
import json
import argparse

import nltk
import numpy as np
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

from util.utils import get_vocab_list

# Data preprocessing
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
    answer_type: str = ''
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
    answer_type: only save datas with answer type 'yes/no' , 'number', or 'other' (default = '' means save all kinds of answer type)
    """
    # Check if save_path exist
    save_path = os.path.join(save_path, answer_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Setup tools
    vocab_list = get_vocab_list(vocab_path)
    ans_list = get_vocab_list(ans_vocab_path)
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')

    def get_tokens(sentence, is_cap=False):
        # turn into lower case, tokenize and remove symbols
        words = tokenizer.tokenize(sentence.lower())
        words = [lemmatizer.lemmatize(word, pos='n') for word in words]
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
    with open(f'{vqa_path}/v2_mscoco_{dataset_type}_annotations.json') as f:
        a_json = json.load(f)['annotations']
        print('Load answer json file.')
    
    q_id = []
    data = []
    for i in tqdm(range(len(a_json)), desc='answer'):
        # If only select certain answer type
        if answer_type != '' and a_json[i]['answer_type'] != answer_type:
            continue
        
        q_id.append(a_json[i]['question_id'])

        image_id = a_json[i]['image_id']
        answers = []
        for a in a_json[i]['answers']:
            answers.append(a['answer'])
        ans_dict = {}
        for a in set(answers):
            if a in ans_list: ans_dict[ans_list.index(a)] = answers.count(a)
        data.append(ans_dict)

    # Save answer dataset
    save_file(file_name=f'{save_path}/{dataset_type}_answers.json',
              desc='This is VQA v2.0 answers dataset.',
              data_type=dataset_type, data=data
    )
    print('answer dataset saved.')

    #########################################################################
    # Read VQA question dataset
    with open(f'{vqa_path}/v2_OpenEnded_mscoco_{dataset_type}_questions.json') as f:
        q_json = json.load(f)['questions']
        print('Load question json file.')

    image_ids = []
    data = []
    for i in tqdm(range(len(q_json)), desc='question'):
        if q_json[i]['question_id'] not in q_id:
            continue
        
        image_id = q_json[i]['image_id']
        image_ids.append(image_id)
        words, tokens = get_tokens(q_json[i]['question'])
        tokens, _ = padding(tokens, q_len)
        data.append({
            'img_file': f'COCO_{dataset_type}_{str(image_id).zfill(12)}.npz',
            'q_word': words,
            'q': tokens,
        })

    # Save question dataset
    save_file(file_name=f'{save_path}/{dataset_type}_questions.json',
              desc='This is VQA v2.0 questions dataset.',
              data_type=dataset_type, data=data
    )
    print('question dataset saved.')

    #########################################################################
    # Read COCO Captions dataset
    with open(f'{coco_path}/captions_{dataset_type}.json') as f:
        c_json = json.load(f)['annotations']
        print('Load caption json file.')
    
    # Store captions based on image ID
    captions = {}
    for c in tqdm(c_json, desc='Store captions'):
        if c['image_id'] not in captions:
            captions[c['image_id']] = []
        captions[c['image_id']].append(c['caption'])

    data = []    
    for image_id in tqdm(image_ids, desc='caption'):
        all_words = []
        all_tokens = []
        all_cap_lens = []
        for i, caption in enumerate(captions[image_id]):
            words, tokens = get_tokens(caption, is_cap=True)
            tokens, cap_len = padding(tokens, c_len)
            all_words.append(words)
            all_tokens.append(tokens)
            all_cap_lens.append(cap_len)
        data.append({'c_word': all_words, 'c': all_tokens, 'cap_len': all_cap_lens})

    # Save answer dataset
    save_file(file_name=f'{save_path}/{dataset_type}_captions.json',
              desc='This is COCO Captions dataset.',
              data_type=dataset_type, data=data
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
        q_len=args.q_len
    )