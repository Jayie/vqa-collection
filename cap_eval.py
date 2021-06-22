import os
import json
import argparse

from util.utils import Logger

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


def parse_args():
    # set parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_ref', type=str, default='../annot/VQA-E/val2014_captions.json')
    parser.add_argument('--load_path', type=str)
    args = parser.parse_args()
    return args


def score(ref, sample):
    # ref and sample are both dict
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Cider(), "CIDEr"),
        (Rouge(), "ROUGE_L"),
    ]
    final_scores = {}
    for scorer, method in scorers:
        print('computing %s score with COCO-EVAL...' % (scorer.method()))
        score, scores = scorer.compute_score(ref, sample)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

if __name__ == '__main__':
    args = parse_args()
    hypotheses = {}
    index = 0
    print('Load predicted captions:', args.load_path)
    with open(os.path.join('checkpoint', args.load_path, 'decode.txt')) as f:
        predict = f.read().split('\n')
        for s in predict:
            if len(s) != 0:
                hypotheses[index] = [s.replace('<start> ', '')]
                index += 1

    references = {}
    index = 0
    print('Load target captions:', args.load_ref)
    with open(args.load_ref) as f:
        target = json.load(f)
        for s in target['data']:
            references[index] = [s['c_word']]
            index += 1

    result = score(references, hypotheses)
    print('================================================')
    with open(os.path.join('checkpoint', args.load_path, 'eval_result.txt'), 'w') as f:
        for k, v in result.items():
            output = f'{k}: {100*v:.8f} %'
            print(output)
            f.write(output)
            f.write('\n')