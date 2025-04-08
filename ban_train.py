import os
import time

import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
from tqdm import tqdm
import time
import torch.nn.functional as F

import utils1.config as config

def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)
    if reduction == 'mean':
        loss *= labels.size(1)
    return loss

def compute_score_with_logits(logits, labels):
    logits = torch.argmax(logits, 1)
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def train(model, train_loader, eval_loader,args,qid2type,opt=None):

    lr_default = 1e-3 if eval_loader is not None else 7e-4
    lr_decay_step = 2
    lr_decay_rate = .25
    lr_decay_epochs = range(10,20,lr_decay_step) if eval_loader is not None else range(10,20,lr_decay_step)
    gradual_warmup_steps = [0.5 * lr_default, 1.0 * lr_default, 1.5 * lr_default, 2.0 * lr_default]
    saving_epoch = 3
    grad_clip = .25

    output=args.output
    logger = utils.Logger(os.path.join(output, 'log.txt'))

    optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_default) \
    if opt is None else opt

    total_step = 0
    best_eval_score = 0

    # utils.print_model(model, logger)
    logger.write('optim: adamax lr=%.4f, decay_step=%d, decay_rate=%.2f, grad_clip=%.2f' % \
        (lr_default, lr_decay_step, lr_decay_rate, grad_clip))

    torch.autograd.set_detect_anomaly(True)
    num_epochs=args.epochs
    run_eval=args.eval_each_epoch

    print("开始训练ban_network---------------------------------------------------")
   
    for epoch in range(num_epochs):

        total_loss = 0
        train_score = 0

        total_norm = 0
        count_norm = 0

        t = time.time()

        N = len(train_loader.dataset)

        if epoch < len(gradual_warmup_steps):
            optim.param_groups[0]['lr'] = gradual_warmup_steps[epoch]
            logger.write('gradual warmup lr: %.4f' % optim.param_groups[0]['lr'])
        elif epoch in lr_decay_epochs:
            optim.param_groups[0]['lr'] *= lr_decay_rate
            logger.write('decreased lr: %.4f' % optim.param_groups[0]['lr'])
        else:
            logger.write('lr: %.4f' % optim.param_groups[0]['lr'])

        for i, (v, b, q, a, qid, bias, mg, f1, type) in tqdm(enumerate(train_loader), ncols=100, desc="Epoch %d" % (epoch + 1), total=len(train_loader)):
            total_step += 1

            ############################A#############
            v = Variable(v).cuda().requires_grad_()
            b = Variable(b).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()
            #########################################

            pred, att = model(v, b, q, a)
            loss = instance_bce_with_logits(pred, a)
            loss.backward()

            total_norm += nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            count_norm += 1
            optim.step()
            optim.zero_grad()

            # genb.train(True)
            total_loss += loss.item() * q.size(0)

            pred = F.softmax(F.normalize(pred) / config.temp, 1)

            batch_score = compute_score_with_logits(pred, a.data).sum()
            train_score += batch_score

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)

        logger.write('Epoch %d, time: %.2f' % (epoch + 1, time.time() - t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))

        if run_eval:
            model.train(False)
            results = evaluate(model, eval_loader, qid2type)
            results["epoch"] = epoch
            results["step"] = total_step
            results["train_loss"] = total_loss
            results["train_score"] = train_score

            model.train(True)

            eval_score = results["score"]
            bound = results["upper_bound"]
            yn = results['score_yesno']
            other = results['score_other']
            num = results['score_number']
            logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
            logger.write('\tyn score: %.2f other score: %.2f num score: %.2f' % (100 * yn, 100 * other, 100 * num))
            main_eval_score = eval_score

            if main_eval_score > best_eval_score:
                model_path = os.path.join(output, 'model.pth')
                torch.save(model.state_dict(), model_path)
                best_eval_score = main_eval_score

        model_path = os.path.join(output, 'model_final.pth')
        torch.save(model.state_dict(), model_path)
    print('best eval score: %.2f' % (best_eval_score*100))


def evaluate(model, dataloader, qid2type):
    score = 0
    upper_bound = 0
    score_yesno = 0
    score_number = 0
    score_other = 0
    total_yesno = 0
    total_number = 0
    total_other = 0 

    for v, b, q, a, qids, bias, mg, f1, type in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        v = Variable(v, requires_grad=False).cuda()
        b = Variable(b, requires_grad=False).cuda()
        q = Variable(q, requires_grad=False).cuda()
        pred, _ = model(v, b,q,None)

        pred = F.softmax(F.normalize(pred) / config.temp, 1)

        batch_score = compute_score_with_logits(pred, a.cuda()).cpu().numpy().sum(1)
        score += batch_score.sum()
        upper_bound += (a.max(1)[0]).sum()
        qids = qids.detach().cpu().int().numpy()
        for j in range(len(qids)):
            qid = qids[j]
            typ = qid2type[str(qid)]
            if typ == 'yes/no':
                score_yesno += batch_score[j]
                total_yesno += 1
            elif typ == 'other':
                score_other += batch_score[j]
                total_other += 1
            elif typ == 'number':
                score_number += batch_score[j]
                total_number += 1
            else:
                print('Hahahahahahahahahahaha')

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    score_yesno /= total_yesno
    score_other /= total_other
    score_number /= total_number

    results = dict(
        score=score,
        upper_bound=upper_bound,
        score_yesno=score_yesno,
        score_other=score_other,
        score_number=score_number,
    )
    return results
