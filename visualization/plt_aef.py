# -*- coding: utf-8 -*-
# import matplotlib.pyplot as plt
#
#
# alpha = [0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
# results = [59.17, 60.97, 61.57, 61.49, 61.11, 61.08]
#
#
# plt.plot(alpha, results, marker='o', linestyle='-', color='r', markerfacecolor='b')
#
#
# plt.title('Performance on VQA CPv2 Dataset')
# plt.xlabel('α')
# plt.ylabel('Result')
#
# plt.legend(['VQA CPv2 Results'])
#
#
# plt.show()


import matplotlib.pyplot as plt
import numpy as np

# 数据
alpha = [0.1,0.2,0.3, 0.4, 0.5, 0.6,0.7, 0.8,0.9, 1.0]
results = [60.08,59.17,60.12, 60.97, 61.57, 61.49,61.06, 61.11,60.98, 61.08]

# 设置柱的宽度
bar_width = 0.05

# 找到最高的柱的索引
max_index = results.index(max(results))

# 绘制柱状图
colors = ['lightblue' if i != max_index else 'lightcoral' for i in range(len(results))]
plt.bar(alpha, results, width=bar_width, color=colors)

# 设置y轴范围
plt.ylim(59, max(results) + 1)

# 添加标题和标签
plt.title('Performance on VQA CPv2 Dataset')
plt.xlabel('α')
plt.ylabel('Result')

# 添加数值标签
for i, value in enumerate(results):
    plt.text(alpha[i], value + 0.02, f'{value}', ha='center', va='bottom')

# 添加图例
# plt.legend(['VQA CPv2 Results'])
legend_general = plt.Line2D([0], [0], color='lightblue', lw=4, label='VQA CPv2 Results')
legend_best = plt.Line2D([0], [0], color='lightcoral', lw=4, label='Best Result')
plt.legend(handles=[legend_general, legend_best])

# 显示网格
plt.grid(True, linestyle='--', alpha=0.7)

# 显示图表
plt.show()


