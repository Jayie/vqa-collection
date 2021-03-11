import os
import argparse
import json
import pickle
import time
import traceback
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import set_dataset
from train import train, evaluate
from sample import sample_vqa
from modules.wrapper import set_model, use_pretrained_embedding
from util.utils import *

def parse_args():
    # set parameters
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('--vocab_path', type=str, default='../data/vocab_list.txt', help='path for vocabulary list')
    parser.add_argument('--ans_path',type=str, default='../data/answer_candidate.txt', help='path for answer candidate list')
    parser.add_argument('--load_path', type=str, default='../annot', help='path for loading dataset')
    parser.add_argument('--feature_path', type=str, default='../../COCO_feature_36', help='path for COCO image features')
    parser.add_argument('--graph_path', type=str, default='../../COCO_graph_36', help='path for COCO spatial relation graphs')
    parser.add_argument('--seed', type=int, default=10, help='random seed')
    parser.add_argument('--device', type=str, default='', help='set device (automatically select if not assign)')
    parser.add_argument('--comment', type=str, default='exp1', help='comment for Tensorboard')

    # dataset and dataloader settings
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle dataloader or not')
    parser.add_argument('--c_len', type=int, default=20)

    # model settings
    parser.add_argument('--model', type=str, default='base', help='model type')
    parser.add_argument('--rnn_type', type=str, default='GRU', help='RNN layer type (GRU or LSTM, default = GRU)')
    parser.add_argument('--att_type', type=str, default='base', help='attention layer type (base/new, default = base')
    parser.add_argument('--embed_dim', type=int, default=300, help='the dimension of embedding')
    parser.add_argument('--hidden_dim', type=int, default=512, help='the dimension of hidden layers (default = 512)')
    parser.add_argument('--v_dim', type=int, default=2048, help='the dimension of visual embedding')
    parser.add_argument('--att_fc_dim', type=int, default=512, help='the dimension of fc layer in the attention module (default = 512)')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--rnn_layer', type=int, default=1, help='the number of RNN layers for question embedding')
    parser.add_argument('--cls_layer', type=int, default=2, help='the number of non-linear layers in the classifier')

    # learning rate scheduler settings
    parser.add_argument('--warm_up', type=int, default=0, help='wram-up epoch number')
    parser.add_argument('--step_size', type=int, default=0, help='step size for learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.25, help='gamma for learning rate scheduler')

    # relation encoder settings
    parser.add_argument('--conv_layer', type=int, default=1, help='the number of GCN layers')
    parser.add_argument('--conv_type', type=str, default='corr', help='GCN type (base/direct/corr, default = corr)')

    parser.add_argument('--mode', type=str, default='train', help='mode: train/val')
    parser.add_argument('--load_model', type=str, default='', help='path for the trained model to evaluate')
    parser.add_argument('--epoches', type=int, default=30, help='the number of epoches')
    parser.add_argument('--batches', type=int, default=0, help='the number of batches we want to run (default = 0 means to run the whole epoch)')
    parser.add_argument('--start_epoch', type=int, default=0, help='the previous epoch number if need to train continuosly')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate')

    # use pre-trained word embedding
    parser.add_argument('--embed_path', type=str, default='', help='path for pre-trained word embedding (default = \'\' means using embedding layer)')

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
    save_path = os.path.join('checkpoint', args.comment)
    with open(os.path.join(save_path, 'param.pkl'), 'wb') as f:
        pickle.dump(args.__dict__, f)
    with open(os.path.join(save_path, 'param.txt'), 'w') as f:
        for key, value in args.__dict__.items():
            f.write(f'{key}: {value}\n')

    # setup model
    model = set_model(  model_type=args.model,
                        ntoken=len(vocab_list),
                        v_dim=args.v_dim,
                        embed_dim=args.embed_dim,
                        hidden_dim=args.hidden_dim,
                        rnn_layer=args.rnn_layer,
                        att_fc_dim=args.att_fc_dim,
                        ans_dim=len(ans_list),
                        cls_layer=args.cls_layer,
                        c_len=args.c_len,
                        dropout=args.dropout,
                        device=args.device,
                        rnn_type=args.rnn_type,
                        att_type=args.att_type,
                        conv_layer=args.conv_layer,
                        conv_type=args.conv_type,
                )
    if args.embed_path != '':
        model = use_pretrained_embedding(model, args.embed_path, args.device)
    print('model ready.')

    if args.mode == 'train':
        # setup training and validation datasets
        print('loading train dataset', end='... ')
        train_data = set_dataset(
            load_dataset=args.load_path,
            feature_path=args.feature_path,
            graph_path=args.graph_path,
            vocab_list=vocab_list,
            ans_list=ans_list,
            is_train=True,
            dataset_type='vqac'
        )
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=args.shuffle)
        print('loading valid dataset', end='... ')
        val_data = set_dataset(
            load_dataset=args.load_path,
            feature_path=args.feature_path,
            graph_path=args.graph_path,
            vocab_list=vocab_list,
            ans_list=ans_list,
            is_val=True,
            dataset_type='vqac'
        )
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=args.shuffle)

        # if need to train continously, load the previous status of model
        score = 0.0
        if args.start_epoch != 0:
            model.load_state_dict(torch.load(f'checkpoint/{args.comment}/best_model.pt'))
            score, _ = evaluate(model, val_loader, args.device)
            print(f'best score: {score:.4f}')

            model.load_state_dict(torch.load(f'{save_path}/epoch_{args.start_epoch-1}_final.pt'))
            print(f'load parameters: {save_path}/epoch_{args.start_epoch-1}_final.pt')

        print('start training.')
        train(
            model=model,
            lr=args.lr,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epoches=args.epoches,
            save_path=save_path,
            device=args.device,
            logger=logger,
            checkpoint=10000,
            max_norm=0.25,
            comment=args.comment+'_train',
            start_epoch=args.start_epoch,
            batches = args.batches,
            best_score = score
        )

    # Evaluate: after training process or for mode 'val'
    if args.mode  == 'train' or args.mode == 'val':
        # load model: if not specified, load the best model
        if args.load_model == '':
            args.load_model = f'checkpoint/{args.comment}/best_model.pt'
        model.load_state_dict(torch.load(args.load_model))
        print('load parameters: ', args.load_model)

        # setup validation dataset
        print('loading valid dataset', end='... ')
        val_data = set_dataset(
            load_dataset=args.load_path,
            feature_path=args.feature_path,
            graph_path=args.graph_path,
            vocab_list=vocab_list,
            ans_list=ans_list,
            is_val=True,
            dataset_type='vqac'
        )
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=args.shuffle)

        # evaluate
        score, _ = evaluate(
            model=model,
            dataloader=val_loader,
            device=args.device,
            logger=logger,
            comment=args.comment+'_val'
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