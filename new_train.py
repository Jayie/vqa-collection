import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.tensorboard import SummaryWriter


def compute_score(predict, target, device):
    """Compute score (according to the VQA evaluation metric)"""
    # get the most possible predicted results for each question
    logits = torch.max(predict, 1)[1].data

    # transfer predicted results into one-hot encoding
    one_hots = torch.zeros(*target.size()).to(device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)

    scores = one_hots * target
    return scores


def instance_bce_with_logits(predict, target):
    """Loss function for VQA prediction"""
    loss = nn.functional.binary_cross_entropy_with_logits(predict, target)
    loss *= target.size(1)
    return loss


def ce_for_language_model(predict, target):
    """ Loss function for caption generation"""
    assert predict.dim() == 2
    loss = nn.functional.cross_entropy(predict, target)
    return loss


def train(  model, lr,
            train_loader, val_loader,
            logger,
            save_path: str,
            num_epoches: int,
            device: str,
            comment: str = '',
            optim_type: str = 'adamax',
            checkpoint: int = 10000,
            start_epoch: int = 0, batches: int = 0,
            max_norm: float = 0.25,
            best_score: float = 0,
): 
    """
    Train process.
    Input:
        model: the model we want to train
        lr: learning rate
        train_loader/val_loader: training/validation dataloader
        logger: logger for writing log file
        save_path: path for saving models
        start_epoch/num_epoches: start from the start_epoch (default = 0), and end at the num_epoches
        device: device
        comment: comment for Tensorboard Summary Writer (default = '')
        optim_type: the type of optimizer (default = adamax)
        checkpoint: save model status for each N batches (default = 10000)
        batches: only run the first N batches per epoch, if batches = 0 then run the whole epoch (default = 0)
        max_norm: for clip_grad_norm (default = 0.25)
        best_score: if load model, get the best score (default = 0)
    """
    writer = SummaryWriter(comment=comment)
    optimizer = torch.optim.Adamax(model.parameters())
    best_score = best_score
    best_epoch = 0
    if batches == 0: batches = len(train_loader)
    for epoch in range(start_epoch, num_epoches):
        start = time.time()
        avg_loss = 0
        prev_loss = 0 # loss at last checkpoint

        model.train()
        for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
            if i == batches: break
            
            predict, caption, _ = model(batch)
            loss = torch.tensor(0, dtype=torch.float).to(device)
            # For VQA
            if predict != None:
                target = batch['a'].float().to(device)
                loss_vqa = instance_bce_with_logits(predict, target)
                loss += loss_vqa

                # write to Tensorboard
                score = compute_score(predict, target, device).sum().item()
                writer.add_scalar('train/vqa/loss', loss_vqa.item(), epoch * batches + i)
                writer.add_scalar('train/vqa/score', score, epoch * batches + i)
                
                # Delete used objects
                predict.detach()
                loss_vqa.detach()
                del predict
                del loss_vqa

            # For captioning
            if caption != None:
                predict = pack_padded_sequence(caption['predict'], caption['decode_len'], batch_first=True).data
                target = pack_padded_sequence(batch['c'].to(device), caption['decode_len'], batch_first=True).data
                loss_cap = ce_for_language_model(predict, target)
                loss += loss_cap

                # Write to Tensorboard
                writer.add_scalar('train/cap/loss', loss_cap.item(), epoch * batches + i)
                
                # Delete used objects
                loss_cap.detach()
                del caption
                del loss_cap
            
            # Back prop.
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()

            avg_loss = loss.item()

            if i % checkpoint == 0 and i != 0:
                torch.save(model.state_dict(), f'{save_path}/epoch_{epoch}_batch_{i}.pt')
                t = time.strftime("%H:%M:%S", time.gmtime(time.time()-start))
                logger.write(f'[Batch {i}] loss: {(avg_loss-prev_loss)/checkpoint:.4f} ({t})')
                prev_loss = avg_loss
            
        # when an epoch is completed
        # save checkpoint
        torch.save(model.state_dict(), f'{save_path}/epoch_{epoch}_final.pt')

        # If there is VQA module: evaluate
        if model.predictor != None:
            # evaluate
            eval_score, bound = evaluate(model, val_loader, device)

            # save log
            avg_loss /= len(batches)
            t = time.strftime("%H:%M:%S", time.gmtime(time.time()-start))
            logger.write(f'[Epoch {epoch}] avg_loss: {avg_loss:.4f} | score: {eval_score:.10f} ({t})')
            writer.add_scalar('train/eval', eval_score, epoch)

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


def evaluate(model, dataloader, device: str, logger = None, comment = None): 
    """
    Evaluate process for VQA.
    Input:
        model: the model we want to train
        dataloader: validation dataloader
        device: device
        logger: logger for writing log file, if logger = None then do not write results into log file (default = None)
        comment: comment for Tensorboard Summary Writer, if comment = None then do not write results into Tensorboard (default = None)
    """
    score = 0
    target_score = 0 # the upper bound of score (i.e. the score of ground truth)
    l = len(dataloader.dataset)

    if comment: writer = SummaryWriter(comment=comment)
    model = model.to(device)
    model.eval()
    start = time.time()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc='evaluate')):
            target = batch['a'].float().to(device)
            predict, _ = model(batch)
            loss = instance_bce_with_logits(predict, target)
            batch_score = compute_score(predict, target, device).sum().item()
            score += batch_score
            target_score += target.max(1)[0].sum().item()

            # write to Tensorboard
            if comment: writer.add_scalar('val/vqa/score', score/l, i)
            
    score /= l
    target_score /= l

    if logger:
        # Write to the log file
        t = time.strftime("%H:%M:%S", time.gmtime(time.time()-start))
        logger.write(f'[{t}] evaluate score: {score:.10f} / bound: {target_score:.10f}')
    
    return score, target_score