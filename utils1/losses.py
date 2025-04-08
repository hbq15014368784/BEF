import torch
from torch import nn
from torch.nn import functional as F

import utils1.config as config


def convert_sigmoid_logits_to_binary_logprobs(logits):
    """Computes log(sigmoid(logits)), log(1-sigmoid(logits))."""
    log_prob = -F.softplus(-logits)
    log_one_minus_prob = -logits + log_prob
    return log_prob, log_one_minus_prob


def cross_entropy_loss(logits, labels, **kwargs):
    """ Modified cross entropy loss. """
    if config.use_cos:
        logits = config.scale * (logits - (1 - kwargs['ldam']))
        # logits = config.scale * (logits - 0.9)
        # logits = config.scale * logits
    f = kwargs['per']
    nll = F.log_softmax(logits, dim=-1)
    loss = -nll * labels
    loss = loss * f
    return loss.sum(dim=-1).mean()


def cross_entropy_loss_arc(logits, labels, **kwargs):
    """ Modified cross entropy loss. """
    f = kwargs['per']
    nll = F.log_softmax(logits, dim=-1)
    loss = -nll * labels
    loss = loss * f

    return loss.sum(dim=-1).mean()

def cross_entropy_loss_weight(logits, labels, **kwargs):
    """ 支持逐样本逐类别加权的交叉熵损失
    
    参数:
        logits: [batch_size, num_classes]
        labels: [batch_size]（类别索引）或 [batch_size, num_classes]（one-hot）
        kwargs['per']: [batch_size, num_classes] 的权重矩阵
    """
    # 确保labels是one-hot格式 [batch_size, num_classes]
    if labels.dim() == 1 or labels.size(1) != logits.size(1):
        labels = F.one_hot(labels.long(), num_classes=logits.size(1)).float()
    
    # 获取3D权重矩阵 [batch_size, num_classes]
    weights = kwargs['per']
    assert weights.shape == logits.shape, f"权重形状{weights.shape}应与logits{logits.shape}一致"
    
    # 计算对数概率 [batch_size, num_classes]
    log_probs = F.log_softmax(logits, dim=-1)
    
    # 计算加权损失 [batch_size, num_classes]
    loss = -log_probs * labels * weights
    
    # 两种聚合方式选其一：
    # return loss.sum() / weights.sum()  # 方法1：加权平均
    return loss.sum(dim=1).mean()    # 方法2：样本平均

def focal_loss_arc(logits, labels, **kwargs):
    """ 焦点损失 (Focal Loss) 版本的损失函数
    
    参数:
        logits: 模型预测的对数几率 [batch_size, num_classes]
        labels: 真实标签的one-hot向量 [batch_size, num_classes]
        kwargs: 其他参数，包括:
            - per: 频率调整因子
            - gamma: 焦点损失的gamma参数(默认为2)
    """
    # 获取频率调整因子
    f = kwargs['per']
    
    # 获取焦点损失的参数，如果没有提供则使用默认值
    gamma = kwargs.get('gamma', 2.0)
    
    # 计算softmax概率
    probs = F.softmax(logits, dim=-1)
    
    # 计算对数概率
    log_probs = F.log_softmax(logits, dim=-1)
    
    # 获取正确类别的概率
    pt = torch.sum(probs * labels, dim=-1)
    
    # 计算焦点权重: (1-pt)^gamma
    focal_weight = (1 - pt) ** gamma
    
    # 计算焦点损失
    loss = -log_probs * labels
    
    # 应用焦点权重和频率调整因子
    loss = loss * focal_weight.unsqueeze(-1) * f
    
    return loss.sum(dim=-1).mean()

class Plain(nn.Module):
    def __init__(self):
        super(Plain, self).__init__()
        self.gamma = 2.0  # 默认gamma值
        self.alpha = 0.25  # 默认alpha值

    def forward(self, logits, labels, **kwargs):
        if config.loss_type == 'ce':
            loss = cross_entropy_loss(logits, labels, **kwargs)
        elif config.loss_type == 'ce_margin':
            loss = cross_entropy_loss_arc(logits, labels, **kwargs)
        elif config.loss_type == 'focal':
            kwargs['gamma'] = self.gamma
            kwargs['alpha'] = self.alpha
            loss = focal_loss_arc(logits, labels, **kwargs)
        elif config.loss_type == 'weighted_ce':
            loss = cross_entropy_loss_weight(logits, labels, **kwargs)
        return loss
