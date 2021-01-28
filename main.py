import os
import argparse
import json
import time
import traceback
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import set_dataset
from train import train, evaluate
from sample import sample_vqa
from util.model import set_model
from util.utils import *

def parse_args():
    # set parameters
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('--vocab_path', type=str, default='../data/vocab_list.txt', help='path for vocabulary list')
    parser.add_argument('--ans_path',type=str, default='../data/answer_candidate.txt', help='path for answer candidate list')
    parser.add_argument('--load_path', type=str, default='../annot', help='path for loading dataset')
    parser.add_argument('--feature_path', type=str, default='../../COCO_feature_36', help='path for COCO image features')
    parser.add_argument('--seed', type=int, default=10, help='random seed')
    parser.add_argument('--device', type=str, default='', help='set device (automatically select if not assign)')
    parser.add_argument('--comment', type=str, default='exp1', help='comment for Tensorboard')

    # dataset and dataloader settings
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--shuffle', type=bool, default=False, help='shuffle dataloader or not')
    parser.add_argument('--c_len', type=int, default=20)

    # model settings
    parser.add_argument('--model', type=str, default='base', help='model type')
    parser.add_argument('--embed_dim', type=int, default=300, help='the dimension of embedding')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='the dimension of hidden layers')
    parser.add_argument('--v_dim', type=int, default=2048, help='the dimension of visual embedding')
    parser.add_argument('--att_fc_dim', type=int, default=1024, help='the dimension of fc layer in the attention module')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--rnn_layer', type=int, default=1, help='the number of RNN layers for question embedding')
    parser.add_argument('--cls_layer', type=int, default=2, help='the number of non-linear layers in the classifier')

    parser.add_argument('--mode', type=str, default='train', help='mode: train/val')
    parser.add_argument('--load_model', type=str, default='', help='path for the trained model to evaluate')
    parser.add_argument('--epoches', type=int, default=30, help='the number of epoches')
    parser.add_argument('--batches', type=int, default=0, help='the number of batches we want to run (default = 0 means run the whole epoch)')
    parser.add_argument('--start_epoch', type=int, default=0, help='the previous epoch number if need to train continuosly')

    args = parser.parse_args()
    return args

def main():
    # get parameters
    args = parse_args()

    ###### settings ######
    # prepare logger
    logger = Logger(args.comment)
    # set random seed
    random_seed(args.seed)
    # set device
    args.device = args.device if args.device != '' else set_device()
    # prepare vocabulary list
    vocab_list = get_vocab_list(args.vocab_path)
    # answer candidate list
    ans_list = get_vocab_list(args.ans_path)
    # save the settings
    logger.write('parameters:')
    save_path = os.path.join('checkpoint', args.comment)
    for key, value in args.__dict__.items():
        logger.write(f'{key}: {value}\n')

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
                    dropout=args.dropout,
                    device=args.device,
                    c_len=args.c_len,
                )
    print('model ready.')

    logger.write('\nmode:', args.mode)
    if args.mode == 'train':
        # setup training and validation datasets
        print('loading train dataset', end='... ')
        train_data = set_dataset(load_dataset=args.load_path, feature_path=args.feature_path, vocab_list=vocab_list, ans_list=ans_list, is_train=True, dataset_type='vqac')
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=args.shuffle)
        print('loading valid dataset', end='... ')
        val_data = set_dataset(load_dataset=args.load_path, feature_path=args.feature_path, vocab_list=vocab_list, ans_list=ans_list, is_val=True, dataset_type='vqac')
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=args.shuffle)

        # if need to train continously, load the previous status of model
        if args.start_epoch != 0:
            model.load_state_dict(torch.load(f'{save_path}/epoch_{args.start_epoch-1}_final.pt'))
            print(f'load parameters: {save_path}/epoch_{args.start_epoch-1}_final.pt')

        print('start training.')
        train(
            model=model,
            model_type=args.model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epoches=args.epoches,
            save_path=save_path,
            device=args.device,
            logger=logger,
            checkpoint=10000,
            max_norm=0.25,
            comment=args.comment,
            start_epoch=args.start_epoch,
            batches = args.batches,
        )

    elif args.mode == 'val':
        # load model
        model.load_state_dict(torch.load(args.load_model))
        print('load parameters:', args.load_model)

        # setup validation dataset
        print('loading valid dataset', end='... ')
        val_data = set_dataset(load_dataset=args.load_path, feature_path=args.feature_path, vocab_list=vocab_list, ans_list=ans_list, is_val=True, dataset_type='vqac')
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=args.shuffle)

        # evaluate
        score, _ = evaluate(
            model=model,
            dataloader=val_loader,
            device=args.device,
            logger=logger
        )

        # Write the results to Tensorboard
        with SummaryWriter() as w:
            w.add_hparams(
                hparam_dict={
                    'name': args.comment,
                    'embed_dim': args.embed_dim,
                    'hidden_dim': args.hidden_dim,
                    'att_fc_dim': args.att_fc_dim,
                    'rnn_layer': args.rnn_layer,
                    'cls_layer': args.cls_layer,
                    'epoches': args.epoches,
                    'dropout': args.dropout,
                },
                metric_dict={
                    'hparam/score': score
                }
            )

    elif args.mode == 'sample_vqa':
        val_data = set_dataset(load_dataset=args.load_path, feature_path=args.feature_path, vocab_list=vocab_list, ans_list=ans_list, is_val=True, dataset_type='vqac')
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
        output = sample_vqa(model, val_loader, ans_list, args.device, logger=logger)
        
        with open(os.path.join(save_path, 'count.json'), 'w') as f:
            f.write(json.dumps(output))

        plt.barh(list(output.keys()), output.values())
        plt.savefig(os.path.join(save_path, 'count.png'))
        plt.show()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        error = traceback.format_exc()
        print(error)
        with open('checkpoint/error.txt', 'w') as f:
            f.write(time.ctime())
            f.write('\n')
            f.write(error)