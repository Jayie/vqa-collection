import os
import time
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


# def compute_score(predict, target, device, get_label=False):
#     """Compute score (according to the VQA evaluation metric)"""
#     predict = predict.to(device)
#     target = target.to(device)

#     # get the most possible predicted results for each question
#     logits = torch.max(predict, 1)[1].data

#     # transfer predicted results into one-hot encoding
#     one_hots = torch.zeros(*target.size()).to(device)
#     one_hots.scatter_(1, logits.view(-1, 1), 1)

#     scores = one_hots * target
#     if get_label: return scores, logits
#     return scores


# def instance_bce_with_logits(predict, target):
#     """Loss function for VQA prediction"""
#     loss = nn.functional.binary_cross_entropy_with_logits(predict, target)
#     loss *= target.size(1)
#     return loss


# def ce_for_language_model(predict, target):
#     """ Loss function for caption generation"""
#     assert predict.dim() == 2
#     loss = nn.functional.cross_entropy(predict, target)
#     return loss


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
            warm_up: int = 0,
            step_size: int = 0,
            gamma: float = 0.25,
            lr_vqa: float = 0,
            lr_cap: float = 0,
): 
    """
    Train process.
    Input:
        model: the model we want to train
        lr / lr_vqa / lr_cap: learning rate
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
        warm_up: warm-up step for lr scheduler (default = 0)
        step_size: step size for lr scheduler (default = 0 i.e. do not use lr scheduler)
        gamma: gama for lr scheduler (default = 0.25)

    """
    writer = SummaryWriter(comment=comment)
    lr_vqa = max(lr_vqa, lr)
    lr_cap = max(lr_cap, lr)
    params = [{'params': model.encoder.parameters()}]
    if model.predictor: params.append({'params': model.predictor.parameters(), 'lr': lr_vqa})
    if model.generator: params.append({'params': model.generator.parameters(), 'lr': lr_cap})
    optimizer = torch.optim.Adamax(params, lr=lr)
    schedualer = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_score = best_score
    best_epoch = 0
    if batches == 0: batches = len(train_loader)

    # # Data parallelism
    # if torch.cuda.device_count() > 1:
    #     print('Use', torch.cuda.device_count(), 'GPUs.')
    #     model = nn.DataParallel(model)

    for epoch in range(start_epoch, num_epoches):
        start = time.time()
        avg_loss = 0
        prev_loss = 0 # loss at last checkpoint

        model.train()
        for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
            if i == batches: break
            
            loss, writes = model.get_loss(batch)
            for tag, write_value in writes.items():
                writer.add_scalar(tag, write_value, epoch * batches + i)

            ##############################################################################################################################
            # TODO: Back prop. strategy for 'Generating Question Relevant Captions to Aid Visual Question Answering'
            #       Suppose that datas in each batch share the same Q-A pair with different caption,
            #       our goal is to select the most relevant one (which means that the gradient of Q-A pair and the caption are similar)
            ##############################################################################################################################
            
            # Back prop.
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()

            avg_loss += loss.item()

            if i % checkpoint == 0 and i != 0:
                # torch.save(model.state_dict(), f'{save_path}/epoch_{epoch}_batch_{i}.pt')
                t = time.strftime("%H:%M:%S", time.gmtime(time.time()-start))
                logger.write(f'[Batch {i}] loss: {(avg_loss-prev_loss)/checkpoint:.4f} ({t})')
                prev_loss = avg_loss
            
        # when an epoch is completed
        # save checkpoint
        torch.save(model.state_dict(), f'{save_path}/epoch_{epoch}.pt')
 
        # If there is VQA module: evaluate
        if model.predictor != None:
            # evaluate
            eval_score, bound = evaluate(model, val_loader, device)

            # save log
            avg_loss /= batches
            t = time.strftime("%H:%M:%S", time.gmtime(time.time()-start))
            logger.show(f'[Epoch {epoch}] avg_loss: {avg_loss:.4f} | score: {eval_score:.10f} ({t})')
            writer.add_scalar('train/eval', eval_score, epoch)

            # reset average loss
            avg_loss = 0

            # save the best model
            if eval_score > best_score:
                torch.save(model.state_dict(), f'{save_path}/best_model.pt')
                best_score = eval_score
                best_epoch = epoch
            logger.show(f'[Result] best epoch: {best_epoch}, score: {best_score:.10f} / {bound:.10f}')

        # if not warm-up step: scheduler step
        if epoch >= warm_up and step_size != 0:
            schedualer.step()
            lr = [param['lr'] for param in optimizer.param_groups]
            logger.show(f'learning rate: {lr}')


def evaluate(model, dataloader, device: str, logger=None, writer=None, ans_index=None, save_path=None):
    """
    Evaluate process for VQA.
    Input:
        model: the model we want to train
        dataloader: validation dataloader
        device: device
        logger: logger for writing log file, if logger = None then do not write results into log file (default = None)
        writer: writer for Tensorboard Summary Writer, if comment = None then do not write results into Tensorboard (default = None)
    """
    score = 0
    target_score = 0 # the upper bound of score (i.e. the score of ground truth)
    all_score = torch.zeros(len(dataloader), dataloader.batch_size)
    all_label = torch.zeros_like(all_score)
    l = len(dataloader.dataset)

    model.eval()
    start = time.time()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc='evaluate')):
            # predict = model.forward_vqa(batch) # predict = [batch_size, ans_dim]
            # batch_score, label = compute_score(predict, target, device, True)
            batch_score, label, target = model.forward_vqa(batch)
            batch_size = batch_score.size(0)
            score += batch_score.sum().item()
            target_score += target.max(1)[0].sum().item()
            all_score[i,:batch_size] = batch_score.sum(dim=1)
            all_label[i,:batch_size] = label

            # write to Tensorboard
            if writer: writer.add_scalar('val/vqa/score', score/l, i)
            
    score /= l
    target_score /= l

    if logger:
        # Write to the log file
        t = time.strftime("%H:%M:%S", time.gmtime(time.time()-start))
        logger.show(f'[{t}] evaluate score: {score:.10f} / bound: {target_score:.10f}')
    
    all_score = all_score.view(-1)
    all_label = all_label.view(-1)

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(all_label, os.path.join(save_path, 'labels.pt'))
        torch.save(all_score, os.path.join(save_path, 'scores.pt'))

    if ans_index is not None:
        # return metric dictionary
        all_score = all_score.numpy()
        output = {}
        for ans in ans_index:
            output['hparam/'+ans] = all_score[ans_index[ans]].sum() / len(ans_index[ans])
        for i in output:
            logger.write(f'\t{i}: {output[i]:.10f}')
        output['hparam/score'] = score
        return output
    
    return score, target_score