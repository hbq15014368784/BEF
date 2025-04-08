import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from dataset import Dictionary, VQAFeatureDataset
import base_model

from torch.autograd import Variable

from tqdm import tqdm
import numpy as np

import torch.nn.functional as F
import matplotlib.pyplot as plt

import utils1.config as config


def compute_score_with_logits(logits, labels):
    logits = torch.argmax(logits, 1)
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


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
        hidden_, pred_l = model(v, q)
        hidden, pred_m = m_model(hidden_, pred_l, mg, 0,  a)

        pred_l = F.softmax(F.normalize(pred_l) / config.temp, 1)
        pred_m = F.softmax(F.normalize(pred_m), 1)
        pred = config.alpha * pred_m + (1 - config.alpha) * pred_l

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

    return score, upper_bound, score_yesno, score_other, score_number

def visualize_candidate_distribution(model, m_model, dataloader, qid2type, candidate_type='number', save_path='candidate_distribution.png'):
    """
    对指定问题类型（默认 'number' 对应 how many 问题）的候选答案概率分布进行可视化。
    该函数遍历数据集，对于每个样本，如果其问题类型为 candidate_type，
    则将其预测分布累积，最后绘制出所有样本的平均概率分布图。
    """
    distribution = None
    count = 0

    # 设置为评估模式，不计算梯度
    model.eval()
    m_model.eval()
    with torch.no_grad():
        for v, s, q, a, qids, bias, mg, f1, typ in tqdm(dataloader, desc="Visualizing distribution", ncols=100):
            v = v.cuda()
            q = q.cuda()
            mg = mg.cuda()
            hidden_, pred_l = model(v, q)
            hidden, pred_m = m_model(hidden_, pred_l, mg, 0, a)

            pred_l = F.softmax(F.normalize(pred_l) / config.temp, 1)
            pred_m = F.softmax(F.normalize(pred_m), 1)
            pred = config.alpha * pred_m + (1 - config.alpha) * pred_l

            # 遍历当前 batch 中的每个样本
            for i in range(len(qids)):
                # 使用 qid2type 来判断样本类型
                qid = str(qids[i].item() if isinstance(qids[i], torch.Tensor) else qids[i])
                if qid2type.get(qid, None) == candidate_type:
                    # 累计概率分布
                    if distribution is None:
                        distribution = pred[i].cpu().numpy()
                    else:
                        distribution += pred[i].cpu().numpy()
                    count += 1

    if count > 0:
        avg_distribution = distribution / count
    else:
        print("No samples found for question type:", candidate_type)
        return

    # 可视化分布情况
    plt.figure(figsize=(12, 6))
    indices = np.arange(len(avg_distribution))
    plt.bar(indices, avg_distribution, color='skyblue')
    plt.xlabel("候选答案索引" )
    plt.ylabel("平均概率")
    plt.title(f"问题类型 '{candidate_type}' 下候选答案的分布情况")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"候选答案分布图已保存到: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser("Train the BottomUpTopDown model with a de-biasing method")

    # Arguments we added
    parser.add_argument('--cache_features', default=False, help="Cache image features in RAM. Makes things much faster"
                        "especially if the filesystem is slow, but requires at least 48gb of RAM")
    parser.add_argument('--dataset', default='cpv2', choices=["v2", "cpv2", "cpv1"], help="Run on VQA-2.0 instead of VQA-CP 2.0")
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=114514, help='random seed')
    parser.add_argument('--load_path', type=str, default='best_model')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset=args.dataset

    if dataset=='cpv1':
        dictionary = Dictionary.load_from_file('data/dictionary_v1.pkl')
    elif dataset=='cpv2' or dataset=='v2':
        dictionary = Dictionary.load_from_file('data/dictionary.pkl')

    print("Building test dataset...")
    eval_dset = VQAFeatureDataset('val', dictionary, dataset=dataset,
                                cache_image_features=args.cache_features)

    # Build the model using the original constructor
    constructor = 'build_%s' % args.model
    # model = getattr(base_model, constructor)(eval_dset, args.num_hid).cuda()
    model, m_model = getattr(base_model, constructor)(eval_dset, args.num_hid)
    # 打印当前模型的 state_dict 键
    print("Current model keys:")
    print(model.state_dict().keys())

    with open('util/qid2type_%s.json'%args.dataset,'r') as f:
        qid2type=json.load(f)

    ckpt = torch.load(os.path.join(args.load_path, 'model.pth'))
    ckpt_m = torch.load(os.path.join(args.load_path, 'm_model.pth'))
    # model.load_state_dict(ckpt, strict=False)

    # 打印所有键（参数名）
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    print("Keys in state_dict:")
    for key in state_dict.keys():
        print(key)

    model.load_state_dict(ckpt)
    m_model.load_state_dict(ckpt_m)
    print('Loaded Model!')

    model=model.cuda()
    m_model=m_model.cuda()

    model.train(False)
    m_model.train(False)

    batch_size = args.batch_size

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=0)

    visualize_candidate_distribution(model, m_model, eval_loader, qid2type, candidate_type='number', save_path='tsne')

    eval_score, bound, yn, other, num = evaluate(model, m_model, eval_loader, qid2type)


    print('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
    print('\tyn score: %.2f other score: %.2f num score: %.2f' % (100 * yn, 100 * other, 100 * num))

if __name__ == '__main__':
    main()
