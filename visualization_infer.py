# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader
from dataset import Dictionary, VQAFeatureDataset
import base_model
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# 加载数据字典和训练集
dictionary = Dictionary.load_from_file('data/dictionary.pkl')
train_dset = VQAFeatureDataset('val', dictionary, dataset='cpv2', cache_image_features=False)

# 构造模型
constructor = 'build_baseline0_newatt'
model, m_model = getattr(base_model, constructor)(train_dset, 1024)

# 加载训练好的模型
root_path = '/home/hbq/genb_main/logs/0319_updn_rmlVQA_deloss1_0.2curloss_config0.8'
# root_path = '/home/hbq/genb_main/logs/0310_updn_fsru'
model_path = os.path.join(root_path, 'model.pth')
model.load_state_dict(torch.load(model_path))
model.eval()

# 设置数据加载器，批次大小为10
train_loader = DataLoader(train_dset, batch_size=512, shuffle=False, num_workers=0)

# 定义目标问题ID
target_qid = '348403007'
# case1 530386002
# case2 528018003


# 加载 JSON 文件
with open('data/vqacp_v2_test_annotations.json', 'r') as file:
    data = json.load(file)

# 在数据中搜索对应的图像ID、COCO split 和正确答案
image_id = None
coco_split = None
correct_answer = None
for entry in data:
    if str(entry['question_id']) == target_qid:
        image_id = entry['image_id']
        coco_split = entry['coco_split']
        correct_answer = entry['multiple_choice_answer']  # 获取正确答案
        break
else:
    print("Question ID not found.")
    exit()

# 构造图像路径
image_root = "data/coco/raw/"
image_filename = f"COCO_{coco_split}_{int(image_id):012d}.jpg"
image_path = os.path.join(image_root, coco_split, image_filename)

# 加载并显示图像
img = Image.open(image_path)
plt.imshow(img)
plt.axis('off')

# 创建画布用于绘制注意力框
ax = plt.gca()

# 获取答案推理（核心部分）
answer_vocab = train_dset.ans2label
answer_labels = {v: k for k, v in answer_vocab.items()}  # 索引->答案映射

# 加载数据和模型
for v, s, q, a, qid, bias, mg, f1, type in train_loader:
    for idx in range(v.size(0)):
        if str(qid[idx].item()) == target_qid:
            current_v = v[idx:idx + 1]
            current_q = q[idx:idx + 1]
            current_spatial = s[idx]

            with torch.no_grad():
                hidden_, pred_l = model(current_v, current_q)

                # 获取注意力权重
                attention_weights = model.last_attention_weights.squeeze().cpu().numpy()

                # 将bbox和权重打包成列表
                bboxes_weights = list(zip(current_spatial.squeeze().numpy(), attention_weights))

                # 混合预测结果
                pred_l = torch.softmax(pred_l, 1)

                # 获取top 10答案
                top_probs, top_indices = torch.topk(pred_l, 5, dim=1)

                # 转换为可读格式
                top_answers = [answer_labels[idx.item()] for idx in top_indices[0]]

                # 构建问题文本
                question_text = ' '.join(
                    [dictionary.idx2word[idx.item()] if idx.item() < len(dictionary.idx2word) else '[UNK]'
                     for idx in current_q[0]]
                )

                # 输出结果
                print(f"\nQuestion: {question_text}")
                print(f"Correct Answer: {correct_answer}")  # 输出正确答案
                print("Top answers:")
                for answer, prob in zip(top_answers, top_probs[0]):
                    print(f"{answer}: {prob.item():.4f}")

            # 选择top 5的注意力区域
            top_bboxes = sorted(bboxes_weights, key=lambda x: -x[1])[:2]

            # 绘制注意力框
            for bbox, weight in top_bboxes:
                x1, y1, x2, y2 = map(int, [bbox[0], bbox[1], bbox[2], bbox[3]])

                # 计算标签位置（框底居中）
                label_x = (x1 + x2) / 2
                label_y = y2 + 15  # 箱底下方15像素

                # 创建矩形框
                rect = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    fill=False,
                    edgecolor='red',
                )
                ax.add_patch(rect)

                # # 添加文本标签
                # plt.text(
                #     label_x, label_y,
                #     f'{weight:.2f}',
                #     ha='center',  # 居中对齐
                #     va='bottom',  # 底部对齐
                #     fontsize=12,
                #     color='navy',  # 深蓝色文字
                # )

            # 添加图例
            plt.legend(loc='upper left')

            # 显示图像和注意力框
            plt.show()
            break
    else:
        continue
    break  # 数据加载器循环退出