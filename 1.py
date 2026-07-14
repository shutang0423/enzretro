import matplotlib.pyplot as plt
import numpy as np

# ==================== 解决中文显示问题 ====================
# 方法1：使用系统支持的中文字体（推荐）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 如果上面方法不行，可以尝试指定字体路径（Windows）
# import matplotlib
# matplotlib.font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# ==================== 图1：不同图编码器的性能对比 ====================
fig1, axes1 = plt.subplots(1, 2, figsize=(15, 6))

# 图1(a)：损失对比
encoders = ['GAT', 'GCN', 'GIN']
loss_data = {
    'Action Loss': [0.1956, 0.2435, 0.2571],
    'Source Loss': [1.1680, 1.2470, 1.2062],
    'Target Loss': [0.6518, 0.7024, 0.6659],
    'Label Loss': [0.2014, 0.2190, 0.2128],
}

x = np.arange(len(encoders))
width = 0.2
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

for i, (loss_name, values) in enumerate(loss_data.items()):
    offset = width * i
    bars = axes1[0].bar(x + offset, values, width, label=loss_name, color=colors[i], edgecolor='black', linewidth=0.5)
    # 添加数值标签
    for bar, val in zip(bars, values):
        axes1[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                      f'{val:.4f}', ha='center', va='bottom', fontsize=8)

axes1[0].set_xlabel('Graph Encoder', fontsize=12)
axes1[0].set_ylabel('Loss Value', fontsize=12)
axes1[0].set_title('Loss Comparison of Different Graph Encoders', fontsize=14, fontweight='bold')
axes1[0].set_xticks(x + width * 1.5, encoders)
axes1[0].legend(loc='upper right')
axes1[0].grid(axis='y', alpha=0.3, linestyle='--')

# 图1(b)：准确率对比
acc_data = {
    'Action Acc': [0.9297, 0.9181, 0.9041],
    'Source Acc': [0.6535, 0.6386, 0.6407],
    'Target Acc': [0.8187, 0.8073, 0.8056],
    'Label Seq Acc': [0.7653, 0.7610, 0.7372],
    'Edit Exact Acc': [0.6096, 0.5851, 0.5694],
}

colors_acc = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#2C5F2D']

for i, (acc_name, values) in enumerate(acc_data.items()):
    offset = width * i
    bars = axes1[1].bar(x + offset, values, width, label=acc_name, color=colors_acc[i], edgecolor='black', linewidth=0.5)
    # 添加数值标签
    for bar, val in zip(bars, values):
        axes1[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                      f'{val:.4f}', ha='center', va='bottom', fontsize=8)

axes1[1].set_xlabel('Graph Encoder', fontsize=12)
axes1[1].set_ylabel('Accuracy', fontsize=12)
axes1[1].set_title('Accuracy Comparison of Different Graph Encoders', fontsize=14, fontweight='bold')
axes1[1].set_xticks(x + width * 2, encoders)
axes1[1].legend(loc='lower right')
axes1[1].grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('fig_encoder_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# ==================== 图2：不同损失策略的性能对比 (GAT编码器) ====================
fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))

strategies = ['Equal\nWeighting', 'Manual\nWeighting', 'Uncertainty\nWeighting']

# 图2(a)：损失对比
loss_data_strategy = {
    'Action Loss': [0.1956, 0.2466, 0.2195],
    'Source Loss': [1.1680, 1.1812, 1.1931],
    'Target Loss': [0.6518, 0.6460, 0.6542],
    'Label Loss': [0.2014, 0.2091, 0.1981],
}

x2 = np.arange(len(strategies))
width2 = 0.2

for i, (loss_name, values) in enumerate(loss_data_strategy.items()):
    offset = width2 * i
    bars = axes2[0].bar(x2 + offset, values, width2, label=loss_name, color=colors[i], edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, values):
        axes2[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                      f'{val:.4f}', ha='center', va='bottom', fontsize=8)

axes2[0].set_xlabel('Loss Strategy', fontsize=12)
axes2[0].set_ylabel('Loss Value', fontsize=12)
axes2[0].set_title('Loss Comparison of Different Loss Strategies (GAT)', fontsize=14, fontweight='bold')
axes2[0].set_xticks(x2 + width2 * 1.5, strategies)
axes2[0].legend(loc='upper right')
axes2[0].grid(axis='y', alpha=0.3, linestyle='--')

# 图2(b)：准确率对比
acc_data_strategy = {
    'Action Acc': [0.9297, 0.9110, 0.9215],
    'Source Acc': [0.6535, 0.6512, 0.6387],
    'Target Acc': [0.8187, 0.8192, 0.8123],
    'Label Seq Acc': [0.7653, 0.7585, 0.7628],
    'Edit Exact Acc': [0.6096, 0.5897, 0.5852],
}

for i, (acc_name, values) in enumerate(acc_data_strategy.items()):
    offset = width2 * i
    bars = axes2[1].bar(x2 + offset, values, width2, label=acc_name, color=colors_acc[i], edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, values):
        axes2[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                      f'{val:.4f}', ha='center', va='bottom', fontsize=8)

axes2[1].set_xlabel('Loss Strategy', fontsize=12)
axes2[1].set_ylabel('Accuracy', fontsize=12)
axes2[1].set_title('Accuracy Comparison of Different Loss Strategies (GAT)', fontsize=14, fontweight='bold')
axes2[1].set_xticks(x2 + width2 * 2, strategies)
axes2[1].legend(loc='lower right')
axes2[1].grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('fig_strategy_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# ==================== 图3：热力图形式展示性能对比 ====================
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))

# 图3(a)：图编码器准确率热力图
encoder_acc_matrix = np.array([
    [0.9297, 0.6535, 0.8187, 0.7653, 0.6096],
    [0.9181, 0.6386, 0.8073, 0.7610, 0.5851],
    [0.9041, 0.6407, 0.8056, 0.7372, 0.5694],
])

metric_names = ['Action\nAcc', 'Source\nAcc', 'Target\nAcc', 'Label Seq\nAcc', 'Edit Exact\nAcc']
encoder_names = ['GAT', 'GCN', 'GIN']

im1 = axes3[0].imshow(encoder_acc_matrix, cmap='YlOrRd', aspect='auto', vmin=0.5, vmax=1.0)
axes3[0].set_xticks(np.arange(len(metric_names)), metric_names, fontsize=10)
axes3[0].set_yticks(np.arange(len(encoder_names)), encoder_names, fontsize=11)
axes3[0].set_title('Accuracy Heatmap of Graph Encoders', fontsize=14, fontweight='bold')

# 添加数值标签
for i in range(len(encoder_names)):
    for j in range(len(metric_names)):
        axes3[0].text(j, i, f'{encoder_acc_matrix[i, j]:.4f}', 
                      ha='center', va='center', fontsize=9, color='black' if encoder_acc_matrix[i, j] < 0.8 else 'white')

plt.colorbar(im1, ax=axes3[0], fraction=0.046, pad=0.04)

# 图3(b)：损失策略准确率热力图
strategy_acc_matrix = np.array([
    [0.9297, 0.6535, 0.8187, 0.7653, 0.6096],
    [0.9110, 0.6512, 0.8192, 0.7585, 0.5897],
    [0.9215, 0.6387, 0.8123, 0.7628, 0.5852],
])

strategy_names = ['Equal\nWeighting', 'Manual\nWeighting', 'Uncertainty\nWeighting']

im2 = axes3[1].imshow(strategy_acc_matrix, cmap='YlOrRd', aspect='auto', vmin=0.5, vmax=1.0)
axes3[1].set_xticks(np.arange(len(metric_names)), metric_names, fontsize=10)
axes3[1].set_yticks(np.arange(len(strategy_names)), strategy_names, fontsize=10)
axes3[1].set_title('Accuracy Heatmap of Loss Strategies (GAT)', fontsize=14, fontweight='bold')

for i in range(len(strategy_names)):
    for j in range(len(metric_names)):
        axes3[1].text(j, i, f'{strategy_acc_matrix[i, j]:.4f}', 
                      ha='center', va='center', fontsize=9, color='black' if strategy_acc_matrix[i, j] < 0.8 else 'white')

plt.colorbar(im2, ax=axes3[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('fig_heatmap_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# ==================== 图4：分组柱状图（更清晰展示edit_exact_acc） ====================
fig4, ax4 = plt.subplots(figsize=(10, 6))

# 编辑精确准确率对比
edit_exact_data = {
    'GAT': 0.6096,
    'GCN': 0.5851,
    'GIN': 0.5694,
}

strategy_edit_exact = {
    'Equal Weighting': 0.6096,
    'Manual Weighting': 0.5897,
    'Uncertainty Weighting': 0.5852,
}

x_pos = np.arange(2)
width_big = 0.35

# 左侧：图编码器对比
bars1 = ax4.bar(x_pos[0] - width_big/2, list(edit_exact_data.values()), width_big, 
                label='Graph Encoders', color='#2E86AB', edgecolor='black')
# 右侧：损失策略对比
bars2 = ax4.bar(x_pos[1] - width_big/2, list(strategy_edit_exact.values()), width_big,
                label='Loss Strategies (GAT)', color='#A23B72', edgecolor='black')

# 添加数值标签
for bar, val in zip(bars1, edit_exact_data.values()):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
for bar, val in zip(bars2, strategy_edit_exact.values()):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax4.set_ylabel('Edit Exact Accuracy', fontsize=12)
ax4.set_title('Comparison of Edit Exact Accuracy', fontsize=14, fontweight='bold')
ax4.set_xticks(x_pos, ['Graph Encoder\nComparison', 'Loss Strategy\nComparison (GAT)'])
ax4.set_ylim(0.5, 0.65)
ax4.legend(loc='upper left')
ax4.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('fig_edit_exact_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("All figures have been generated successfully!")
print("Generated files:")
print("  - fig_encoder_comparison.png")
print("  - fig_strategy_comparison.png")
print("  - fig_heatmap_comparison.png")
print("  - fig_edit_exact_comparison.png")