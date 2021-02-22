import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def instance_bce_with_logits(predict, target):
    """Loss function for VQA prediction"""
    loss = nn.functional.binary_cross_entropy_with_logits(predict, target)
    loss *= target.size(1)
    return loss


def compute_score(predict, target, device):
    """Compute score (according to the VQA evaluation metric)"""
    # get the most possible predicted results for each question
    logits = torch.max(predict, 1)[1].data

    # transfer predicted results into one-hot encoding
    one_hots = torch.zeros(*target.size()).to(device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)

    scores = one_hots * target
    return scores


def ce_for_language_model(predict, target):
    """ Loss function for caption generation"""
    assert predict.dim() == 2
    loss = nn.functional.cross_entropy(predict, target)
    return loss


def set_optim(optim_type: str = 'adamax'):
    optim = {
        'adamax': torch.optim.Adamax,
        'adadelta': torch.optim.Adadelta,
        'adam': torch.optim.Adam
    }
    assert optim_type in optim.keys()
    return optim[optim_type]


def train(  model, lr,
            train_loader, val_loader, num_epoches, save_path, device, logger,
            comment='', checkpoint=10000, max_norm=0.25,
            start_epoch=0, batches=0,
            model_type='base', optim_type='adamax'
    ):
    """
    Train process.
    Input:
        model: the model we want to train
        lr: learning rate
        train_loader/val_loader: training/validation dataloader
        start_epoch/num_epoches: start from the start_epoch (default = 0), and end at the num_epoches
        save_path: path for saving models
        device: device
        logger: logger for writing log file
        comment: comment for Tensorboard Summary Writer (default = '')
        checkpoint: save model status for each N batches (default = 10000)
        max_norm: for clip_grad_norm (default = 0.25)
        batches: only run the first N batches per epoch, if batches = 0 then run the whole epoch (default = 0)
        model_type: the type of model (default = base, i.e. Bottom-Up and Top-Down model)
        optim_type: the type of optimizer (default = adamax)
    """
    # optimizer = torch.optim.Adamax(model.parameters())
    optimizer = set_optim(optim_type)(model.parameters(), lr=lr)
    writer = SummaryWriter(comment=comment)
    best_score = 0
    best_epoch = 0
    if batches == 0: batches = len(train_loader)
    
    # Parallelism
    if torch.cuda.device_count() > 1:
        print('Use', torch.cuda.device_count(), 'GPUs.')
        model = nn.DataParallel(model)
    
    model = model.to(device)
    for epoch in range(start_epoch, num_epoches):
        start = time.time()
        avg_loss = 0
        prev_loss = 0
        
        model.train()
        for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
            if i == batches: break
            target = batch['a'].float().to(device)

            predict = model(batch)
            loss = instance_bce_with_logits(predict, target)
            # TODO: loss functions for captioning
            # TODO: for VQA with Question-relevant Caption, back-prop the gradients only from the most relevant caption

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()

            avg_loss += loss.item()
            score = compute_score(predict, target, device).sum().item()

            # write loss and score to Tensorboard
            writer.add_scalar(f'train/loss', loss.item(), epoch * batches + i)
            writer.add_scalar(f'train/score', score, epoch * batches + i)
            
            if i % checkpoint == 0 and i != 0:
                # save checkpoint
                torch.save(model.state_dict(), f'{save_path}/epoch_{epoch}_batch_{i}.pt')
                t = time.strftime("%H:%M:%S", time.gmtime(time.time()-start))
                logger.write(f'[Batch {i}] loss: {(avg_loss-prev_loss)/checkpoint:.4f} ({t})')
                prev_loss = avg_loss

        # when an epoch is completed
        # save checkpoint
        torch.save(model.state_dict(), f'{save_path}/epoch_{epoch}_final.pt')

        # evaluate
        eval_score, bound = evaluate(model, val_loader, device)
        
        # save log
        avg_loss /= len(train_loader.dataset)
        t = time.strftime("%H:%M:%S", time.gmtime(time.time()-start))
        logger.write(f'[Epoch {epoch}] avg_loss: {avg_loss:.4f} | score: {eval_score:.10f} ({t})')

        # reset average loss
        avg_loss = 0

        # save the best model
        if eval_score > best_score:
            torch.save(model.state_dict(), f'{save_path}/best_model.pt')
            best_score = eval_score
            best_epoch = epoch

        msg = f'[Result] best epoch: {best_epoch}, score: {best_score:.10f} / {bound:.10f}'
        print(msg)
        logger.write(msg)

def evaluate(model, dataloader, device, logger=None, comment=None):
    """
    Evaluate process.
    Input:
        model: the model we want to train
        val_loader: validation dataloader
        device: device
        logger: logger for writing log file, if logger = None then do not write results into log file (default = None)
        comment: comment for Tensorboard Summary Writer, if comment = None then do not write results into Tensorboard (default = None)
    """
    score = 0
    target_score = 0 # the upper bound of score (i.e. the score of ground truth)
    
    if comment: writer = SummaryWriter(comment=comment)
    model = model.to(device)
    model.eval()
    start = time.time()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc='eval')):
            target = batch['a'].float().to(device)
            predict = model(batch)
            batch_score = compute_score(predict, target, device).sum().item()
            score += batch_score
            target_score += target.max(1)[0].sum().item()

            if comment: writer.add_scalar(f'val/score', batch_score, i)
    
    l = len(dataloader.dataset)
    score /= l
    target_score /= l

    if logger:
        # Write to the log file
        t = time.strftime("%H:%M:%S", time.gmtime(time.time()-start))
        logger.write(f'[{t}] evaluate score: {score:.10f} / bound: {target_score:.10f}')
    
    return score, target_score