import torch
from torch.utils.data import DataLoader
from dataset import Dictionary, VQAFeatureDataset
import base_model
import json
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

print("Start to extract features")


# 加载特征和问题类型
features = np.load('tsne/saved_features_fefd.npy')
question_types = np.load('tsne/question_types_fefd.npy')

# 使用部分数据进行 t-SNE
sample_size = 10000  # 根据需要调整
if len(features) > sample_size:
    indices = np.random.choice(len(features), sample_size, replace=False)
    sampled_features = features[indices]
    sampled_types = question_types[indices]
else:
    sampled_features = features
    sampled_types = question_types

print("Start to perform PCA and t-SNE")
# 使用 PCA 预处理
pca = PCA(n_components=50)
reduced_features = pca.fit_transform(sampled_features)

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=42)
tsne_features = tsne.fit_transform(reduced_features)

# 可视化
plt.figure(figsize=(12, 10))
unique_types = set(sampled_types)
for q_type in unique_types:
    indices = [i for i, t in enumerate(sampled_types) if t == q_type]
    plt.scatter(tsne_features[indices, 0], tsne_features[indices, 1], label=q_type, alpha=0.7)

plt.legend()
plt.title("t-SNE visualization of Answer Embeddings by Question Type")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

# 保存图像到文件
plt.savefig('tsne_visualization_genb.png')  # 您可以更改文件名和格式

plt.show()

