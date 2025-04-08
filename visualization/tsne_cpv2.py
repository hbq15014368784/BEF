import torch
from torch.utils.data import DataLoader
from dataset import Dictionary, VQAFeatureDataset
import base_model
import json
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 加载数据字典和训练集
dictionary = Dictionary.load_from_file('data/dictionary.pkl')
train_dset = VQAFeatureDataset('val', dictionary, dataset='cpv2', cache_image_features=False)

# 构造模型
constructor = 'build_baseline0_newatt'
model, m_model = getattr(base_model, constructor)(train_dset, 1024)

# 加载训练好的模型
model_path = 'logs/0319_updn_rmlVQA_deloss1_0.2curloss_config0.8/model.pth'
# model_path = 'logs/genb_rml_all/model.pth'
model.load_state_dict(torch.load(model_path))
model.eval()

# 加载问题类型映射
with open('util/qid2type_cpv2.json', 'r') as f:
    qid2type = json.load(f)

# 数据加载器
train_loader = DataLoader(train_dset, 512, shuffle=True, num_workers=0)

# 感兴趣的问题类型
interested_question_types = {'are','do','where are the','are they','what is', 'how many', 'is there','is it'}

print("Start to extract features")
# 提取特征
features = []
question_types = []

with torch.no_grad():
    for v, s, q, a, qid, bias, mg, f1, types in train_loader:  # 假设 `type` 变量包含所有样本的问题类型
        for idx in range(len(types)):  # 遍历批次中的每个样本
            current_type = types[idx]  # 获取当前样本的问题类型
            if current_type in interested_question_types:
                answer_features, _ = model(v[idx:idx+1], q[idx:idx+1])  # 只处理当前样本
                features.extend(answer_features.cpu().numpy())  # 保存特征
                question_types.append(current_type)  # 保存问题类型
                print("Processed:", current_type)

features = np.array(features)
question_types = np.array(question_types)
# # 加载特征和问题类型
# features = np.load('saved_features.npy')
# question_types = np.load('question_types.npy')

# 保存特征和问题类型
np.save('saved_features_genb.npy', features)
np.save('question_types_genb.npy', question_types)
print("Features and question types saved.")

# # 使用部分数据进行 t-SNE
# sample_size = 3000  # 根据需要调整
# if len(features) > sample_size:
#     indices = np.random.choice(len(features), sample_size, replace=False)
#     sampled_features = features[indices]
#     sampled_types = question_types[indices]
# else:
#     sampled_features = features
#     sampled_types = question_types
#
# print("Start to perform PCA and t-SNE")
# # 使用 PCA 预处理
# pca = PCA(n_components=50)
# reduced_features = pca.fit_transform(sampled_features)
#
# # 使用 t-SNE 进行降维
# tsne = TSNE(n_components=2, random_state=42)
# tsne_features = tsne.fit_transform(reduced_features)
#
# # 可视化
# plt.figure(figsize=(12, 10))
# unique_types = set(sampled_types)
# for q_type in unique_types:
#     indices = [i for i, t in enumerate(sampled_types) if t == q_type]
#     plt.scatter(tsne_features[indices, 0], tsne_features[indices, 1], label=q_type, alpha=0.7)
#
# plt.legend()
# plt.title("t-SNE visualization of Answer Embeddings by Question Type")
# plt.xlabel("t-SNE Dimension 1")
# plt.ylabel("t-SNE Dimension 2")
#
# # 保存图像到文件
# plt.savefig('tsne_visualization_genb_rml.png')  # 您可以更改文件名和格式
#
# plt.show()

