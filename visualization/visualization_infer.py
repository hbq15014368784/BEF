# # -*- coding: utf-8 -*-
#
# import torch
# from torch.utils.data import DataLoader
# from dataset import Dictionary, VQAFeatureDataset
# import base_model
#
# import json
# import os
# from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import numpy as np
#
# from scipy.ndimage import zoom
#
# # 加载数据字典和训练集
# dictionary = Dictionary.load_from_file('data/dictionary.pkl')
# train_dset = VQAFeatureDataset('val', dictionary, dataset='cpv2', cache_image_features=False)
#
# # 构造模型
# constructor = 'build_baseline0_newatt'
# model, m_model = getattr(base_model, constructor)(train_dset, 1024)
#
# # 加载训练好的模型
# model_path = 'logs/genb_rml_all/model.pth'
# model.load_state_dict(torch.load(model_path))
# model.eval()
#
# # 设置数据加载器，批次大小为10
# train_loader = DataLoader(train_dset, batch_size=256, shuffle=False, num_workers=0)
#
# # 定义目标问题ID
# target_qid = '576513003'  # 示例问题ID 4172001
#
# # 加载 JSON 文件
# with open('data/vqacp_v2_test_annotations.json', 'r') as file:
#     data = json.load(file)
#
# # 在数据中搜索对应的图像ID和COCO split
# image_id = None
# coco_split = None
# for entry in data:  # 直接遍历 data 列表
#     if str(entry['question_id']) == target_qid:
#         image_id = entry['image_id']
#         coco_split = entry['coco_split']
#         break
# else:
#     print("Question ID not found.")
#     exit()
#
# # 构造图像路径
# image_root = "data/coco/raw/"
# image_filename = f"COCO_{coco_split}_{int(image_id):012d}.jpg"
# image_path = os.path.join(image_root, coco_split, image_filename)
#
# print(f"Image path for question ID {target_qid}: {image_path}")
#
# # 加载并显示图像
# img = Image.open(image_path)
# plt.imshow(img)
# plt.axis('off')  # 不显示坐标轴
# plt.show()
#
# # # 处理多个样本
# # for v, q, a, qid, bias, mg, f1, types, spatial_features in train_loader:
# #     for idx in range(v.size(0)):  # 遍历批次中的每个样本
# #         if str(qid[idx].item()) == target_qid:
# #             current_v = v[idx:idx+1]  # 当前样本的图像特征
# #             current_q = q[idx:idx+1]  # 当前样本的问题
# #
# #             with torch.no_grad():
# #                 _, logits = model(current_v, current_q)
# #
# #                 attention_weights = model.last_attention_weights
# #
# #                 # 假设注意力权重形状为[1, 36, 1]，去除多余的维度并转换为2D
# #                 attention_weights = attention_weights.squeeze().numpy().reshape(6, 6)  # 假设可以映射到6x6网格
# #
# #                 plt.imshow(attention_weights, cmap='hot', interpolation='nearest')
# #                 plt.colorbar()
# #                 plt.title("Attention Weights Heatmap")
# #                 plt.show()
#
# # 创建与图像尺寸相同的空白热图
# heatmap = np.zeros((img.height, img.width))
#
# # 加载数据和模型
# for v, q, a, qid, bias, mg, f1, types, spatial_features in train_loader:
#     for idx in range(v.size(0)):
#         if str(qid[idx].item()) == target_qid:
#             current_v = v[idx:idx+1]
#             current_q = q[idx:idx+1]
#             current_spatial = spatial_features[idx]
#
#             with torch.no_grad():
#                 _, logits = model(current_v, current_q)
#                 attention_weights = model.last_attention_weights.squeeze().cpu().numpy()
#
#             # 更新热图
#             for bbox, weight in zip(current_spatial.squeeze().numpy(), attention_weights):
#                 x1, y1, x2, y2 = map(int, [bbox[0], bbox[1], bbox[2], bbox[3]])
#                 heatmap[y1:y2, x1:x2] += weight  # 将权重添加到热图的对应区域
#
#
#
#                 # # 查看注意力权重的形状和内容
#                 # print("Attention weights shape:", attention_weights.shape)
#                 # print("Attention weights data:", attention_weights)
#                 #
#                 # attention_weights_2d = attention_weights.reshape((6, 6))  # 将权重重塑为6x6矩阵
#                 #
#                 # plt.figure(figsize=(6, 6))
#                 # plt.imshow(attention_weights_2d, cmap='hot', interpolation='nearest')
#                 # plt.colorbar()
#                 # plt.title('Attention Weights Heatmap')
#                 # plt.show()
#
#
#             # 获取答案的分数
#             answer_probabilities = torch.softmax(logits, dim=1)
#
#             # 选择置信度前十的答案和索引
#             top_probs, top_indices = torch.topk(answer_probabilities, 10, dim=1)
#
#             # 答案的索引和答案对应关系
#             answer_vocab = train_dset.ans2label
#             answer_labels = {v: k for k, v in answer_vocab.items()}  # 创建从索引到答案的映射
#
#             # 将索引转换成答案字符串
#             top_answers = [answer_labels[idx.item()] for idx in top_indices[0]]
#
#             # 正确处理问题文本
#             question_text = ' '.join(
#                 [dictionary.idx2word[idx.item()] if idx.item() < len(dictionary.idx2word) else '[UNK]' for idx in
#                  current_q[0]])
#
#             # 打印问题和前十个答案及其置信度
#             print(f"Question: {question_text}")
#             print("Top answers and their probabilities:")
#             for answer, prob in zip(top_answers, top_probs[0]):
#                 print(f"{answer}: {prob.item():.4f}")
#
#             break  # 找到匹配的问题ID后跳出循环
#
# # 归一化热图
# heatmap = np.clip(heatmap / np.max(heatmap), 0, 1)
#
# # 叠加热图到原始图像
# plt.figure(figsize=(10, 10))
# plt.imshow(img)
# plt.imshow(heatmap, cmap='jet', alpha=0.6)  # alpha 控制热图透明度
# plt.axis('off')
# plt.show()

# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader
from dataset import Dictionary, VQAFeatureDataset
import base_model
import json
import os

# 加载数据字典和训练集
dictionary = Dictionary.load_from_file('data/dictionary.pkl')
train_dset = VQAFeatureDataset('val', dictionary, dataset='cpv2', cache_image_features=False)

# 构造模型
constructor = 'build_baseline0_newatt'
model1, m_model1 = getattr(base_model, constructor)(train_dset, 1024)
model2, m_model2 = getattr(base_model, constructor)(train_dset, 1024)

# 加载训练好的模型
model_path1 = 'logs/genb_rml_all/model.pth'
model_path2 = 'logs/Updn_1/model.pth'
model1.load_state_dict(torch.load(model_path1))
model2.load_state_dict(torch.load(model_path2))
model1.eval()
model2.eval()

# 设置数据加载器，批次大小为256
train_loader = DataLoader(train_dset, batch_size=256, shuffle=False, num_workers=0)

# 定义目标问题ID
target_qid = '497498004'  # 示例问题ID

# 加载 JSON 文件
with open('data/vqacp_v2_test_annotations.json', 'r') as file:
    data = json.load(file)

# 在数据中搜索对应的图像ID和COCO split
image_id = None
coco_split = None
for entry in data:
    if str(entry['question_id']) == target_qid:
        image_id = entry['image_id']
        coco_split = entry['coco_split']
        break
else:
    print("Question ID not found.")
    exit()

# 构造图像路径
image_root = "data/coco/raw/"
image_filename = f"COCO_{coco_split}_{int(image_id):012d}.jpg"
image_path = os.path.join(image_root, coco_split, image_filename)
print(f"Image path for question ID {target_qid}: {image_path}")

# 优化样本查找过程
for v, q, a, qid, bias, mg, f1, types, spatial_features in train_loader:
    for idx in range(v.size(0)):
        if str(qid[idx].item()) == target_qid:
            current_v = v[idx:idx+1]
            current_q = q[idx:idx+1]
            current_spatial = spatial_features[idx]
            break
# 使用模型1进行推理
with torch.no_grad():
    _, logits1 = model1(current_v, current_q)

# 使用模型2进行推理
with torch.no_grad():
    _, logits2 = model2(current_v, current_q)

# 获取答案的分数
answer_probabilities1 = torch.softmax(logits1, dim=1)
answer_probabilities2 = torch.softmax(logits2, dim=1)

# 选择置信度最高的答案和索引
top_prob1, top_index1 = torch.topk(answer_probabilities1, 1, dim=1)
top_prob2, top_index2 = torch.topk(answer_probabilities2, 1, dim=1)

# 答案的索引和答案对应关系
answer_vocab = train_dset.ans2label
answer_labels = {v: k for k, v in answer_vocab.items()}  # 创建从索引到答案的映射

# 将索引转换成答案字符串
top_answer1 = answer_labels[top_index1.item()]
top_answer2 = answer_labels[top_index2.item()]

# 正确处理问题文本
question_text = ' '.join(
    [dictionary.idx2word[idx.item()] if idx.item() < len(dictionary.idx2word) else '[UNK]' for idx in current_q[0]])

# 打印问题和两个模型的最高置信度答案及其置信度
print(f"Question: {question_text}")
print(f"Model bef Top Answer: {top_answer1} with probability {top_prob1.item():.4f}")
print(f"Model updn Top Answer: {top_answer2} with probability {top_prob2.item():.4f}")



