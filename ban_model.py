"""
Bilinear Attention Networks
Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang
https://arxiv.org/abs/1805.07932

This code is written by Jin-Hwa Kim.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import utils
from attention import BiAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc2 import FCNet
from bc import BCNet
from counting import Counter
from FSRU import FSRU


class BanModel(nn.Module):
    def __init__(self, dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, op, glimpse):
        super(BanModel, self).__init__()
        self.dataset = dataset
        self.op = op
        self.glimpse = glimpse
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.c_prj = nn.ModuleList(c_prj)
        self.classifier = classifier
        self.counter = counter
        self.drop = nn.Dropout(.5)
        self.tanh = nn.Tanh()

        self.sp2freq = FSRU(d_model=1024, seq_len=14, dropout=0., mlp_ratio=4., num_filter=2, num_class=2, num_layer=1)

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb.forward_all(w_emb) # [batch, q_len, q_dim]
        boxes = b[:,:,:4].transpose(1,2)

        b_emb = [0] * self.glimpse
        att, logits = self.v_att.forward_all(v, q_emb) # b x g x v x q

        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v, q_emb, att[:,g,:,:]) # b x l x h
            
            atten, _ = logits[:,g,:,:].max(2)
            embed = self.counter(boxes, atten)

            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
            q_emb = q_emb + self.c_prj[g](embed).unsqueeze(1)

        _, _, _, fsru_feat = self.sp2freq(q, v)

        # print(f"q_emb shape: {q_emb.shape}")
        # print(f"fsru_feat: {fsru_feat.shape}")

        fsru_feat = fsru_feat.unsqueeze(1).repeat(1, 14, 1)  # [batch, 14, 512]

        if fsru_feat.size(-1) != q_emb.size(-1):
            fsru_feat = self.fsru_proj(fsru_feat)  # 将512转换为q_emb的维度

        q_emb = q_emb + fsru_feat

        logits = self.classifier(q_emb.sum(1))

        return logits, att

def build_ban(dataset, num_hid, op='c', gamma=4, task='vqa'):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = BiAttention(dataset.v_dim, num_hid, num_hid, gamma)
    b_net = []
    q_prj = []
    c_prj = []
    objects = 10  # minimum number of boxes
    for i in range(gamma):
        b_net.append(BCNet(dataset.v_dim, num_hid, num_hid, None, k=1))
        q_prj.append(FCNet([num_hid, num_hid], '', .2))
        c_prj.append(FCNet([objects + 1, num_hid], 'ReLU', .0))
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, .5)
    counter = Counter(objects)
    return BanModel(dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, op, gamma)
