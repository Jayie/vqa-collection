import os
import argparse
import json

import torch
from torch.utils.data import DataLoader

from dataset import set_dataset
from util.model import set_model
from train import train
from util.utils import *

def parse_args():
    # set parameters
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('--vocab_path', type=str, default='../data/vocab_list.txt', help='path for vocabulary list')
    parser.add_argument('--ans_path',type=str, default='../data/answer_candidate.txt', help='path for answer candidate list')
    parser.add_argument('--load_path', type=str, default='../annot', help='path for loading dataset')
    parser.add_argument('--feature_path', type=str, default='../../COCO_feature_36', help='path for COCO image features')
    # parser.add_argument('--dataset', type=str, default='train2014', help='dataset name')
    # parser.add_argument('--save_path', type=str, default='checkpoint/test', help='path for saving outputs')
    parser.add_argument('--comment', type=str, default='exp1', help='comment for Tensorboard')
    parser.add_argument('--seed', type=int, default=10, help='random seed')

    # dataset and dataloader settings
    parser.add_argument('--epoches', type=int, default=30, help='the number of epoches')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--shuffle', type=bool, default=False, help='shuffle dataloader or not')

    # model settings
    parser.add_argument('--model', type=str, default='base', help='model type')
    parser.add_argument('--embed_dim', type=int, default=300, help='the dimension of embedding')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='the dimension of hidden layers')
    parser.add_argument('--v_dim', type=int, default=2048, help='the dimension of visual embedding')
    parser.add_argument('--att_fc_dim', type=int, default=1024, help='the dimension of fc layer in the attention module')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--rnn_layer', type=int, default=1, help='the number of RNN layers for question embedding')
    parser.add_argument('--cls_layer', type=int, default=2, help='the number of non-linear layers in the classifier')

    args = parser.parse_args()
    return args

def main():
    # get parameters
    args = parse_args()

    ###### for saving results ######
    save_path = os.path.join('checkpoint', args.comment)
    # prepare logger
    logger = Logger(save_path)
    # save the settings
    with open(os.path.join(save_path, 'param.txt'), 'w') as f:
        for key, value in args.__dict__.items():
            f.write(f'{key}: {value}\n')

    ###### settings ######
    # set random seed
    random_seed(args.seed)
    # set device
    device = set_device()
    # prepare vocabulary list
    vocab_list = get_vocab_list(args.vocab_path)
    # answer candidate list
    ans_list = get_vocab_list(args.ans_path)

    # setup dataset and dataloader
    print('loading train dataset', end='... ')
    train_data = set_dataset(load_dataset=args.load_path, feature_path=args.feature_path, vocab_list=vocab_list, ans_list=ans_list, is_train=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=args.shuffle)
    print('loading valid dataset', end='... ')
    val_data = set_dataset(load_dataset=args.load_path, feature_path=args.feature_path, vocab_list=vocab_list, ans_list=ans_list, is_val=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=args.shuffle)


    # setup model
    model = set_model(args.model)(
        ntoken=len(vocab_list),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        v_dim=args.v_dim,
        att_fc_dim=args.att_fc_dim,
        ans_dim=len(ans_list),
        rnn_layer=args.rnn_layer,
        cls_layer=args.cls_layer,
        dropout=args.dropout
    )
    print('model ready.')

    print('start training.')
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epoches=args.epoches,
        save_path=save_path,
        device=device,
        logger=logger,
        checkpoint=10000,
        max_norm=0.25,
        comment=args.comment,
    )


if __name__ == '__main__':
    main()

