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

Tensor = torch.cuda.FloatTensor

def compute_supcon_loss(feats, qtype):
    tau = 1.0
    if isinstance(qtype, tuple):
      i = 0
      dic = {}
      for item in qtype:
          if item not in dic:
              dic[item] = i
              i = i + 1
      tau = 1.0
      qtype = torch.tensor([dic[item] for item in qtype]).cuda()
    feats_filt = F.normalize(feats, dim=1)
    targets_r = qtype.reshape(-1, 1)
    targets_c = qtype.reshape(1, -1)
    mask = targets_r == targets_c
    mask = mask.float().cuda()
    feats_sim = torch.exp(torch.matmul(feats_filt, feats_filt.T) / tau)
    negatives = feats_sim*(1.0 - mask)
    negative_sum = torch.sum(negatives)
    positives = torch.log(feats_sim/negative_sum)*mask
    positive_sum = torch.sum(positives)
    positive_sum = positive_sum/torch.sum(mask)

    sup_con_loss = -1*torch.mean(positive_sum)
    return sup_con_loss
    
class DisentangledSupCon(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        # 特征解耦投影层
        self.content_proj = nn.Linear(feat_dim, feat_dim//2)
        self.bias_proj = nn.Linear(feat_dim, feat_dim//2)

        
    def forward(self, features, labels, bias_scores):
        # 特征解耦
        content_feat = F.normalize(self.content_proj(features), dim=1)
        bias_feat = F.normalize(self.bias_proj(features), dim=1)
        
        # 内容空间对比损失
        content_loss = compute_supcon_loss(content_feat, labels)
        
        
        # 偏差空间对抗对比损失（迫使偏差特征与标签无关）
        bias_labels = torch.randint(0, 2, (len(labels),)).cuda()  # 随机生成伪标签
        bias_loss = compute_supcon_loss(bias_feat, bias_labels)  # 最大化偏差特征的混乱度
        
        # # 正交约束
        # orth_loss = torch.mean(torch.abs(torch.sum(content_feat * bias_feat, dim=1)))

        #动态正交约束
        cos_sim = F.cosine_similarity(content_feat, bias_feat, dim=1)
        orth_weight = 0.1 * (1 + torch.sigmoid(cos_sim.mean()*5))  # 相似度越高权重越大
        orth_loss = torch.mean(torch.abs(cos_sim)) * orth_weight
        
        return content_loss + 0.5*bias_loss + orth_loss

    
class CurriculumLoss(nn.Module):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        
    def forward(self, features, labels, margin, epoch):
        """简单的课程学习损失
        Args:
            features: [batch_size, feat_dim]
            labels: 标签
            margin: 频率信息 [batch_size, num_classes]
            epoch: 当前轮次
        """
        # 计算课程进度 (0~1)
        progress = epoch / self.total_epochs
        
        # 动态调整难度系数
        difficulty_ratio = 0.2 + 0.6 * progress  # 从0.2逐渐增加到0.8
        
        # 计算样本相似度
        features = F.normalize(features, dim=1)
        sim_matrix = torch.matmul(features, features.T)
        
        # 创建标签mask
        label_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        
        # 选择困难负样本
        with torch.no_grad():
            neg_mask = 1 - label_mask
            neg_sim = sim_matrix * neg_mask
            
            # 对每个样本选择最困难的负样本
            num_negs = int(difficulty_ratio * (neg_mask.sum(1).max().item()))
            hardest_negs, _ = neg_sim.topk(k=num_negs, dim=1)
        
        # 计算课程学习损失（关注困难负样本）
        curriculum_loss = -torch.log(1 - hardest_negs.mean())

        return curriculum_loss

class BiasAwareNormalization(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        # 偏差信息映射层
        self.bias_proj = nn.Sequential(
            nn.Linear(2274, 512),   # 先降维减少参数量
            nn.ReLU(),
            nn.Linear(512, feat_dim),
            nn.Sigmoid()
        )
        # 可学习的缩放和平移参数
        self.gamma = nn.Parameter(torch.ones(feat_dim))
        self.beta = nn.Parameter(torch.zeros(feat_dim))
        
    def forward(self, x, bias):
        """
        Args:
            x: 输入特征 [batch_size, feat_dim]
            bias: 偏差分数 [batch_size, 2274]
        Returns:
            归一化后的特征 [batch_size, feat_dim]
        """
        # 沿特征维度归一化（dim=-1）
        x_mean = x.mean(dim=-1, keepdim=True)  # [batch_size, 1]
        x_std = x.std(dim=-1, keepdim=True)    # [batch_size, 1]
        x_norm = (x - x_mean) / (x_std + 1e-6)
        
        # 生成偏差感知参数
        gate = self.bias_proj(bias)  # [batch_size, feat_dim]
        
        # 动态调整归一化结果
        return self.gamma * (x_norm * gate) + self.beta
    
def compute_score_with_logits(logits, labels):
    logits = torch.argmax(logits, 1)
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def calc_genb_loss(logits, bias, labels):
    gen_grad = torch.clamp(2 * labels * torch.sigmoid(-2 * labels * bias.detach()), 0, 1)
    loss = F.binary_cross_entropy_with_logits(logits, gen_grad)
    loss *= labels.size(1)
    return loss
    
def train(model, m_model, loss_fn, genb, discriminator, train_loader, eval_loader, args, qid2type):
   
    optim = torch.optim.Adamax([
            {'params': filter(lambda p: p.requires_grad, model.parameters())},
            {'params': m_model.parameters()}
        ], lr=0.001)

    torch.autograd.set_detect_anomaly(True)
    num_epochs=args.epochs
    run_eval=args.eval_each_epoch
    output=args.output

    logger = utils.Logger(os.path.join(output, 'log.txt'))
    total_step = 0
    best_eval_score = 0

    # supcon_loss = DisentangledSupCon(feat_dim=1024).cuda()
    # curriculum = CurriculumLoss(args.epochs).cuda()

    # bias_norm = BiasAwareNormalization(feat_dim=1024).cuda()

    logger.write('start training: seed: %d, batch_size: %d, epochs: %d' % (args.seed, args.batch_size, args.epochs))

    for epoch in range(num_epochs):

        total_loss = 0
        train_score = 0

        t = time.time()
        for i, (v,s, q, a, qid, bias, mg, f1, type) in tqdm(enumerate(train_loader), ncols=100, desc="Epoch %d" % (epoch + 1), total=len(train_loader)):
            total_step += 1

            #########################################
            v = Variable(v).cuda().requires_grad_()
            q = Variable(q).cuda()
            a = Variable(a).cuda()
            s = Variable(s).cuda()

            mg = mg.cuda()
            f1 = f1.cuda()
            bias = bias.cuda()
            gt = torch.argmax(a, 1)

            #########################################

            # get model output
            optim.zero_grad()

            hidden_, pred = model(v, q)
            # hidden, pred_m = m_model(hidden_, pred, mg, epoch, a)
            # dict_args = {'margin': mg, 'bias': bias, 'hidden': hidden, 'epoch': epoch, 'per': f1}

            # ce_loss = -F.log_softmax(pred, dim=-1) * a
            # ce_loss = ce_loss * f1
            # loss = ce_loss.sum(dim=-1).mean() + loss_fn(hidden, a, **dict_args)

            # # 偏差感知归一化
            # # print(bias.shape)
            # hidden_ = bias_norm(hidden_, bias)  # 新增代码

            # # 解耦对比学习
            # content_loss = supcon_loss(hidden_, gt, bias)
            
            # # 计算课程学习损失
            # curr_loss = curriculum(hidden_, gt, mg, epoch)

            # loss = compute_supcon_loss(hidden_, gt) + loss.mean() + 0.3 * content_loss + 0.2 * curr_loss

            loss = F.binary_cross_entropy_with_logits(pred, a);

            loss.backward()


            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            total_loss += loss.item() * q.size(0)

            # pred = F.softmax(F.normalize(pred) / config.temp, 1)
            # pred_m = F.softmax(F.normalize(pred_m), 1)
            # pred = config.alpha * pred_m + (1 - config.alpha) * pred

            batch_score = compute_score_with_logits(pred, a.data).sum()
            train_score += batch_score


        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)

        logger.write('Epoch %d, time: %.2f' % (epoch + 1, time.time() - t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))

        if run_eval:
            model.train(False)
            results = evaluate(model, m_model, eval_loader, qid2type)
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



def evaluate(model, m_model, dataloader, qid2type):
    score = 0
    upper_bound = 0
    score_yesno = 0
    score_number = 0
    score_other = 0
    total_yesno = 0
    total_number = 0
    total_other = 0 

    for v, s, q, a, qids, bias, mg, f1, type in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        v = Variable(v, requires_grad=False).cuda()
        q = Variable(q, requires_grad=False).cuda()
        
        mg = mg.cuda()
        hidden_, pred = model(v, q)
        hidden, pred_m = m_model(hidden_, pred, mg, 0,  a)

        # pred = F.softmax(F.normalize(pred) / config.temp, 1)
        # pred_m = F.softmax(F.normalize(pred_m), 1)
        # pred = config.alpha * pred_m + (1 - config.alpha) * pred

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
