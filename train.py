import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Loss function
def instance_bce_with_logits(predict, target):
    loss = nn.functional.binary_cross_entropy_with_logits(predict, target)
    loss *= target.size(1)
    return loss


# Compute score (according to the VQA evaluation metric)
def compute_score(predict, target, device):
    target = target.to(device)
    
    # get the most possible predicted results for each question
    logits = torch.max(predict, 1)[1].to(device)

    # transfer predicted results into one-hot encoding
    one_hots = torch.zeros(*target.size()).to(device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)

    scores = one_hots * target / 3
    return scores


def train(model, train_loader, val_loader, num_epoches, save_path, device, logger, checkpoint=10000, max_norm=0.25, comment='', start_epoch=0, batches=0):
    optimizer = torch.optim.Adamax(model.parameters())
    writer = SummaryWriter(comment=comment)
    best_score = 0
    best_epoch = 0
    L = len(train_loader.dataset)
    
    model = model.to(device)
    for epoch in range(start_epoch, num_epoches):
        model.train()
        start = time.time()
        avg_loss = 0
        prev_loss = 0
        
        for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
            if batches != 0 and i == batches: break

            v = batch['img'].to(device)
            q = batch['q'].to(device)
            target = batch['a'].float().to(device)
            
            predict, v = model(v, q)
            
            loss = instance_bce_with_logits(predict, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()

            avg_loss += loss.item()
            score = compute_score(predict, target, device).sum().item()

            # write loss and score to Tensorboard
            writer.add_scalar('bottom-up-vqa/loss', loss.item(), epoch * L + i)
            writer.add_scalar('bottom-up-vqa/score', score, epoch * L + i)
            
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

        logger.write(f'[Result] best epoch: {best_epoch}, score: {best_score:.10f} / {bound:.10f}')

def evaluate(model, dataloader, device, logger=None):
    score = 0
    target_score = 0 # the upper bound of score (i.e. the score of ground truth)
    
    model = model.to(device)
    model.eval()
    start = time.time()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc='eval')):
            v = batch['img'].to(device)
            q = batch['q'].to(device)
            target = batch['a'].float().to(device)
            
            predict, _ = model(v, q)
            
            batch_score = compute_score(predict, target, device).sum().item()
            score += batch_score
            
            target_score += target.max(1)[0].sum().item()
    
    l = len(dataloader.dataset)
    score /= l
    target_score /= l

    if logger != None:
        # Write to the log file
        t = time.strftime("%H:%M:%S", time.gmtime(time.time()-start))
        logger.write(f'[{t}] evaluate score: {score:.10f} / bound: {target_score:.10f}')
    
    return score, target_score