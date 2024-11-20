import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier, KAN2_0Classifier
from fc import FCNet, MLP, KAN2_0
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable

import torch.nn.init as init
# import torch.fft

from gfnet import GlobalFilter

eps = 1e-12

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        # init.kaiming_normal(m.weight)
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

import math
# class GlobalFilter(nn.Module):
#     def __init__(self, dim, h=6, w=4):
#         super().__init__()
#         self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
#         self.w = w
#         self.h = h
#
#     def forward(self, x, spatial_size=None):
#         B, N, C = x.shape
#         if spatial_size is None:
#             a = b = int(math.sqrt(N))
#         else:
#             a, b = spatial_size
#
#         x = x.view(B, a, b, C)
#
#         x = x.to(torch.float32)
#         x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
#         weight = torch.view_as_complex(self.complex_weight)
#         x = x * weight
#         x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
#
#         x = x.reshape(B, N, C)
#
#         return x

class GlobalFilter(nn.Module):
    def __init__(self, dim=2048, h=6, w=4):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(512, 513, 2, dtype=torch.float32) * 0.02)
    def forward(self, q, v):

        batch_size = q.size(0)
        weight = torch.view_as_complex(self.complex_weight[:batch_size, :, :])

        print("q:"+ str(q.shape))
        q = torch.fft.rfft2(q, dim=-1, norm='ortho')
        v = torch.fft.rfft2(v, dim=-1, norm='ortho')
        # weight = torch.view_as_complex(self.complex_weight)
        q = q * weight
        v = v * weight
        joint_repr = q + v
        joint_repr = torch.fft.irfft2(joint_repr, dim=-1, norm='ortho')
        q = torch.fft.irfft2(q, dim=-1, norm='ortho')
        v = torch.fft.irfft2(v, dim=-1, norm='ortho')

        return joint_repr, q, v

# import random
# from math import sqrt
# class GlobalFilter(nn.Module):
#     def __init__(self, dim, h=14, w=8,
#                  mask_radio=0.1, mask_alpha=0.5,
#                  noise_mode=1,
#                  uncertainty_model=0, perturb_prob=0.5,
#                  uncertainty_factor=1.0,
#                  noise_layer_flag=0, gauss_or_uniform=0, ):
#         super().__init__()
#         self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
#         self.w = w
#         self.h = h
#
#         self.mask_radio = mask_radio
#
#         self.noise_mode = noise_mode
#         self.noise_layer_flag = noise_layer_flag
#
#         self.alpha = mask_alpha
#
#         self.eps = 1e-6
#         self.factor = uncertainty_factor
#         self.uncertainty_model = uncertainty_model
#         self.p = perturb_prob
#         self.gauss_or_uniform = gauss_or_uniform
#
#     def _reparameterize(self, mu, std, epsilon_norm):
#         # epsilon = torch.randn_like(std) * self.factor
#         epsilon = epsilon_norm * self.factor
#         mu_t = mu + epsilon * std
#         return mu_t
#
#     def spectrum_noise(self, img_fft, ratio=1.0, noise_mode=1,
#                        uncertainty_model=0, gauss_or_uniform=0):
#         """Input image size: ndarray of [H, W, C]"""
#         """noise_mode: 1 amplitude; 2: phase 3:both"""
#         """uncertainty_model: 1 batch-wise modeling 2: channel-wise modeling 3:token-wise modeling"""
#         if random.random() > self.p:
#             return img_fft
#         batch_size, h, w, c = img_fft.shape
#
#         img_abs, img_pha = torch.abs(img_fft), torch.angle(img_fft)
#
#         img_abs = torch.fft.fftshift(img_abs, dim=(1))
#
#         h_crop = int(h * sqrt(ratio))
#         w_crop = int(w * sqrt(ratio))
#         h_start = h // 2 - h_crop // 2
#         w_start = 0
#
#         img_abs_ = img_abs.clone()
#         if noise_mode != 0:
#             if uncertainty_model != 0:
#                 if uncertainty_model == 1:
#                     # batch level modeling
#                     miu = torch.mean(img_abs_[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :], dim=(1, 2),
#                                      keepdim=True)
#                     var = torch.var(img_abs_[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :], dim=(1, 2),
#                                     keepdim=True)
#                     sig = (var + self.eps).sqrt()  # Bx1x1xC
#
#                     var_of_miu = torch.var(miu, dim=0, keepdim=True)
#                     var_of_sig = torch.var(sig, dim=0, keepdim=True)
#                     sig_of_miu = (var_of_miu + self.eps).sqrt().repeat(miu.shape[0], 1, 1, 1)
#                     sig_of_sig = (var_of_sig + self.eps).sqrt().repeat(miu.shape[0], 1, 1, 1)  # Bx1x1xC
#
#                     if gauss_or_uniform == 0:
#                         epsilon_norm_miu = torch.randn_like(sig_of_miu)  # N(0,1)
#                         epsilon_norm_sig = torch.randn_like(sig_of_sig)
#
#                         miu_mean = miu
#                         sig_mean = sig
#
#                         beta = self._reparameterize(mu=miu_mean, std=sig_of_miu, epsilon_norm=epsilon_norm_miu)
#                         gamma = self._reparameterize(mu=sig_mean, std=sig_of_sig, epsilon_norm=epsilon_norm_sig)
#                     elif gauss_or_uniform == 1:
#                         epsilon_norm_miu = torch.rand_like(sig_of_miu) * 2 - 1.  # U(-1,1)
#                         epsilon_norm_sig = torch.rand_like(sig_of_sig) * 2 - 1.
#                         beta = self._reparameterize(mu=miu, std=sig_of_miu, epsilon_norm=epsilon_norm_miu)
#                         gamma = self._reparameterize(mu=sig, std=sig_of_sig, epsilon_norm=epsilon_norm_sig)
#                     else:
#                         epsilon_norm_miu = torch.randn_like(sig_of_miu)  # N(0,1)
#                         epsilon_norm_sig = torch.randn_like(sig_of_sig)
#                         beta = self._reparameterize(mu=miu, std=1., epsilon_norm=epsilon_norm_miu)
#                         gamma = self._reparameterize(mu=sig, std=1., epsilon_norm=epsilon_norm_sig)
#
#                     # adjust statistics for each sample
#                     img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :] = gamma * (
#                             img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :] - miu) / sig + beta
#
#                 elif uncertainty_model == 2:
#                     # element level modeling
#                     miu_of_elem = torch.mean(img_abs_[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :], dim=0,
#                                              keepdim=True)
#                     var_of_elem = torch.var(img_abs_[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :], dim=0,
#                                             keepdim=True)
#                     sig_of_elem = (var_of_elem + self.eps).sqrt()  # 1xHxWxC
#
#                     if gauss_or_uniform == 0:
#                         epsilon_sig = torch.randn_like(
#                             img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :])  # BxHxWxC N(0,1)
#                         gamma = epsilon_sig * sig_of_elem * self.factor
#                     elif gauss_or_uniform == 1:
#                         epsilon_sig = torch.rand_like(
#                             img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :]) * 2 - 1.  # U(-1,1)
#                         gamma = epsilon_sig * sig_of_elem * self.factor
#                     else:
#                         epsilon_sig = torch.randn_like(
#                             img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :])  # BxHxWxC N(0,1)
#                         gamma = epsilon_sig * self.factor
#
#                     img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :] = \
#                         img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :] + gamma
#         img_abs = torch.fft.ifftshift(img_abs, dim=(1))  # recover
#         img_mix = img_abs * (np.e ** (1j * img_pha))
#         return img_mix
#
#     def forward(self, x, spatial_size=None):
#         B, N, C = x.shape
#         if spatial_size is None:
#             a = b = int(math.sqrt(N))
#         else:
#             a, b = spatial_size
#
#         x = x.view(B, a, b, C)
#         x = x.to(torch.float32)
#         x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
#
#         if self.training:
#             if self.noise_mode != 0 and self.noise_layer_flag == 1:
#                 x = self.spectrum_noise(x, ratio=self.mask_radio, noise_mode=self.noise_mode,
#                                         uncertainty_model=self.uncertainty_model,
#                                         gauss_or_uniform=self.gauss_or_uniform)
#         weight = torch.view_as_complex(self.complex_weight)
#         x = x * weight
#         x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
#         x = x.reshape(B, N, C)
#         return x

llh_shift = torch.tensor(5.0)
# def get_predictive_entropy_over_concepts(log_likelihoods, semantic_set_ids):
#     """Compute the semantic entropy"""
#     mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))
#     # This is ok because all the models have the same semantic set ids
#     semantic_set_ids = semantic_set_ids[0]
#     entropies = []
#     for row_index in range(mean_across_models.shape[0]):
#         aggregated_likelihoods = []
#         row = mean_across_models[row_index]
#         semantic_set_ids_row = semantic_set_ids[row_index]
#         for semantic_set_id in torch.unique(semantic_set_ids_row):
#             aggregated_likelihoods.append(torch.logsumexp(row[semantic_set_ids_row == semantic_set_id], dim=0))
#         aggregated_likelihoods = torch.tensor(aggregated_likelihoods) - llh_shift
#         entropy = - torch.sum(aggregated_likelihoods, dim=0) / torch.tensor(aggregated_likelihoods.shape[0])
#         entropies.append(entropy)
#
#     return torch.tensor(entropies)


def get_predictive_entropy_over_concepts(log_likelihoods, semantic_set_ids):
    """Compute the semantic entropy with proper normalization"""
    mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))
    semantic_set_ids = semantic_set_ids[0]
    entropies = []

    for row_index in range(mean_across_models.shape[0]):
        aggregated_likelihoods = []
        row = mean_across_models[row_index]
        semantic_set_ids_row = semantic_set_ids[row_index]

        for semantic_set_id in torch.unique(semantic_set_ids_row):
            aggregated_likelihoods.append(
                torch.logsumexp(row[semantic_set_ids_row == semantic_set_id], dim=0)
            )

        aggregated_likelihoods = torch.tensor(aggregated_likelihoods) - llh_shift
        probs = torch.softmax(aggregated_likelihoods, dim=0)

        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        entropies.append(entropy)

    return torch.tensor(entropies)


from sklearn.neighbors import NearestNeighbors


def mutual_information(x, y, k=5):
    """Estimate mutual information between two feature vectors using k-nearest neighbors in PyTorch"""

    # Concatenate x and y to form a combined feature space
    xy = torch.cat((x, y), dim=1)

    # Compute pairwise squared Euclidean distances between all points in the combined feature space
    distances = torch.cdist(xy, xy, p=2)

    # Sort the distances to get the nearest neighbors (ignoring the diagonal as it represents self-distance)
    sorted_distances, _ = torch.sort(distances, dim=1)

    # Get the distances for the k-nearest neighbors (ignoring the first one since it's the distance to itself)
    knn_distances = sorted_distances[:, 1:k + 1]

    joint_entropy = torch.mean(torch.log(knn_distances[:, -1] + 1e-10))  # Adding a small epsilon to avoid log(0)

    x_entropy = torch.mean(torch.log(knn_distances[:, -1] + 1e-10))  # Entropy for X (similar computation)
    y_entropy = torch.mean(torch.log(knn_distances[:, -1] + 1e-10))  # Entropy for Y (same)

    mi = x_entropy + y_entropy - joint_entropy

    return mi

class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.filter = GlobalFilter(dim=2048, h=6, w=4)
        self.device = 'cuda:0'

    def forward(self, v, q):
        """Forward
        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]
        return: logits
        """

        w_emb = self.w_emb(q)
        q_emb, _ = self.q_emb(w_emb)  # [batch, q_dim]

        att = self.v_att(v, q_emb)

        att = nn.functional.softmax(att, 1)

        v_emb = (att * v)  # [batch, nums, v_dim]
        # v_emb = self.filter(v_emb)
        v_emb = v_emb.sum(1)
        # print(v_emb.shape)

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)

        # v_repr = self.filter(v_repr)
        # v_repr = v_repr.sum(1)

        # # q_repr_np = q_repr.detach().cpu().numpy()
        # v_repr_np = v_repr.detach().cpu().numpy()
        #
        # # q_fft = np.fft.fft(q_repr_np, axis=-1)
        # v_fft = np.fft.fft(v_repr_np, axis=-1)
        #
        # # q_fft_abs = np.abs(q_fft)
        # v_fft_abs = np.abs(v_fft)
        #
        # # q_fft_tensor = torch.tensor(q_fft_abs, dtype=q_repr.dtype, device=q_repr.device)
        # v_fft_tensor = torch.tensor(v_fft_abs, dtype=v_repr.dtype, device=v_repr.device)
        # print(v_repr.shape)

        # v_repr_fft = torch.fft.rfft2(v_repr, dim=(0, 1), norm='ortho')
        # weight = torch.view_as_complex(self.complex_weight)
        # # Adjust weight size to match v_repr size if needed
        # if v_repr_fft.shape[0] != weight.shape[0]:
        #     if v_repr_fft.shape[0] < weight.shape[0]:
        #         weight = weight[:v_repr_fft.shape[0], ...]  # Trim weight if v_repr has fewer images
        #     else:
        #         padding = torch.zeros(v_repr_fft.shape[0] - weight.shape[0], weight.shape[1], weight.shape[2]).to(
        #             weight.device)
        #         weight = torch.cat([weight, padding], dim=0)  # Pad weight if v_repr has more images
        # v_repr_fft = v_repr_fft * weight
        # v_repr_fft = torch.fft.irfft2(v_repr_fft, dim=(0, 1), norm='ortho')


        # q_repr = torch.fft.rfft2(q_repr, dim=(0, 1), norm='ortho')
        # weight = torch.view_as_complex(self.complex_weight)
        # # print(v_repr.shape)
        # # print(weight.shape)
        # # Adjust weight size to match v_repr size if needed
        # if q_repr.shape[0] != weight.shape[0]:
        #     if q_repr.shape[0] < weight.shape[0]:
        #         weight = weight[:q_repr.shape[0], ...]  # Trim weight if v_repr has fewer images
        #     else:
        #         padding = torch.zeros(q_repr.shape[0] - weight.shape[0], weight.shape[1], weight.shape[2]).to(
        #             weight.device)
        #         weight = torch.cat([weight, padding], dim=0)  # Pad weight if v_repr has more images
        # q_repr = q_repr * weight
        # # q_repr = torch.fft.irfft2(q_repr, dim=(0, 1), norm='ortho')

        # cma_block = CMA_Block(1024, 512, 1024).cuda()
        # output = cma_block(v_repr, v_repr_fft)

        # joint_repr = q_repr * output

        # joint_repr_fft = q_repr * v_repr_fft

        # joint_repr = joint_repr_fft + joint_repr
        # joint_repr = torch.log(torch.sigmoid(joint_repr) + eps)

        # joint_repr = torch.fft.irfft2(joint_repr, dim=(0, 1), norm='ortho')

        # joint_repr = q_repr * v_repr
        joint_repr, q, v = self.filter(q_repr, v_repr)


        # semantic_loss_q = get_predictive_entropy_over_concepts(joint_repr, q_repr).to(self.device)
        # semantic_loss_v = get_predictive_entropy_over_concepts(joint_repr, v_repr).to(self.device)

        # semantic_loss_vq = get_predictive_entropy_over_concepts(v_repr, q_repr).to(self.device)
        # semantic_loss_qv = get_predictive_entropy_over_concepts(q_repr, v_repr).to(self.device)
        # semantic_loss = semantic_loss_vq.mean() + semantic_loss_qv.mean()

        # similarity_vq = mutual_information(v_repr, q_repr)
        # similarity_qv = mutual_information(q_repr, v_repr)
        #
        # similarity_loss = similarity_vq + similarity_qv

        # similarity_dot = torch.dot(joint_repr, q_repr)
        #
        # if similarity_dot >= 0.7:
        #     similarity_loss = similarity_dot
        # else:
        #     similarity_loss = 0

        logits = self.classifier(joint_repr)

        return _, joint_repr, logits


class GenB(nn.Module):
    def __init__(self, num_hid, dataset):
        super(GenB, self).__init__()
        self.num_hid = num_hid
        self.w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
        self.q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
        self.v_att = NewAttention(dataset.v_dim, self.q_emb.num_hid, num_hid)
        self.q_net = FCNet([self.q_emb.num_hid, num_hid])
        self.v_net = FCNet([dataset.v_dim, num_hid])
        self.classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.generate = nn.Sequential(
            *block(num_hid//8, num_hid//4),
            *block(num_hid//4, num_hid//2),
            *block(num_hid//2, num_hid),
            nn.Linear(num_hid, num_hid*2),
            nn.ReLU(inplace=True)
            )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, v, q, gen=True):
        w_emb = self.w_emb(q)
        q_emb, _ = self.q_emb(w_emb)

        b, c, f = v.shape

        # generate from noise
        if gen==True:
            v_z = Variable(torch.cuda.FloatTensor(np.random.normal(0,1, (b,c, 128))))
            v = self.generate(v_z.view(-1, 128)).view(b,c,f)

        att = self.v_att(v, q_emb)

        att = nn.functional.softmax(att, 1)
        v_emb = (att * v).sum(1)

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)

        joint_repr = v_repr * q_repr

        logits = self.classifier(joint_repr)

        return logits


class Discriminator(nn.Module):
    def __init__(self, num_hid, dataset):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dataset.num_ans_candidates, 1024),
            nn.ReLU(True),
            nn.Linear(num_hid, num_hid//2),
            nn.ReLU(True),
            nn.Linear(num_hid//2, num_hid//4),
            nn.ReLU(True),
            nn.Linear(num_hid//4, 1),
            nn.Sigmoid(),
            )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        return self.net(z)

'''
    
'''
class CMA_Block(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel):  # 2048, 512, 2048
        super(CMA_Block, self).__init__()

        self.conv1 = nn.Conv1d(in_channel, hidden_channel, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channel, hidden_channel, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channel, hidden_channel, kernel_size=1)

        self.scale = hidden_channel ** -0.5

        self.conv4 = nn.Sequential(
            nn.Conv1d(hidden_channel, out_channel, kernel_size=1),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, rgb, freq):
        _, num_features = rgb.size()

        q = self.conv1(rgb.unsqueeze(2)).view(rgb.size(0), -1, 1)  # Q
        k = self.conv2(freq.unsqueeze(2)).view(freq.size(0), -1, 1)  # K
        v = self.conv3(freq.unsqueeze(2)).view(freq.size(0), -1, 1)  # V

        attn = torch.matmul(q, k.transpose(1, 2)) * self.scale
        m = attn.softmax(dim=-1)

        z = torch.matmul(m, v).view(rgb.size(0), -1)
        # print(z.shape)
        output = rgb + self.conv4(z.unsqueeze(2)).squeeze(2)

        return output

def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)

def build_baseline0_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    # q_net = KAN2_0([q_emb.num_hid, num_hid], drop=0.0, degree=3)
    # v_net = KAN2_0([dataset.v_dim, num_hid], drop=0.0, degree=3)

    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)

    # classifier = KAN2_0Classifier(
    #     num_hid, num_hid * 2, dataset.num_ans_candidates, 0, 3)

    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)
