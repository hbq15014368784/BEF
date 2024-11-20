# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

# 数据
train_data = [5, 1, 6, 4, 3, 2]  # 从上到下的数据
test_data = [0, 5, 1, 6, 4, 3, 2]  # 从上到下的数据

# 创建图形和轴对象
fig, ax = plt.subplots(figsize=(8, 10))

# 计算每个数据在各自数据集中的比例
train_total = sum(train_data)
test_total = sum(test_data)

# 为每个数据点创建不同的高度位置
train_heights = np.cumsum([0] + [val/train_total for val in train_data[:-1]])
test_heights = np.cumsum([0] + [val/test_total for val in test_data[:-1]])

# 设置颜色
colors = ['#E8E8E8', '#FFB6C1', '#90EE90', '#FFB6C1', '#ADD8E6', '#DDA0DD', '#F0E68C']

# 绘制Train数据
for i, (value, height) in enumerate(zip(train_data, train_heights)):
    height_ratio = value/train_total  # 计算每个块的高度比例
    ax.bar(0, height_ratio, bottom=height, color=colors[i], edgecolor='white', alpha=0.8)
    # 添加数值标签在每个块的中间
    ax.text(0, height + height_ratio/2, str(value), ha='center', va='center')

# 绘制Test数据
for i, (value, height) in enumerate(zip(test_data, test_heights)):
    height_ratio = value/test_total  # 计算每个块的高度比例
    ax.bar(1, height_ratio, bottom=height, color=colors[i], edgecolor='white', alpha=0.8)
    # 添加数值标签在每个块的中间
    ax.text(1, height + height_ratio/2, str(value), ha='center', va='center')

# 设置x轴标签
plt.xticks([0, 1], ['Train', 'Test'])

# 移除边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# 移除y轴刻度
ax.set_yticks([])

# 设置标题
plt.title('How many', pad=20)

# 调整布局
plt.tight_layout()

# 保存图形
plt.savefig('distribution.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()