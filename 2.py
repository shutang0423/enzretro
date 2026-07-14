import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.signal import savgol_filter
import os

# ==================== 解决中文显示问题 ====================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置文件路径 ====================
data_dir = '/root/autodl-tmp/enzretro-rl-2/1'  # 请修改为实际路径
file_path = os.path.join(data_dir, 'lr.json')

# ==================== 读取数据函数 ====================
def read_lr_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    # 数据格式: [timestamp, step, lr_value]
    timestamps = [item[0] for item in data]
    steps = [item[1] for item in data]
    lr_values = [item[2] for item in data]
    return steps, lr_values

# ==================== 平滑函数 ====================
def smooth_data(values):
    window_size = min(51, len(values) // 10 * 2 + 1)
    if window_size < 5:
        window_size = 5
    if window_size % 2 == 0:
        window_size += 1
    if window_size >= 5 and len(values) >= window_size:
        return savgol_filter(values, window_size, 3)
    else:
        return values

# ==================== 读取数据 ====================
if not os.path.exists(file_path):
    print(f"文件不存在: {file_path}")
    exit()

steps, lr_values = read_lr_data(file_path)

print("=" * 50)
print("学习率数据统计")
print("=" * 50)
print(f"数据点数量: {len(lr_values)}")
print(f"训练步数范围: {min(steps)} ~ {max(steps)}")
print(f"学习率范围: {min(lr_values):.8f} ~ {max(lr_values):.8f}")
print(f"初始学习率: {lr_values[0]:.8f}")
print(f"最终学习率: {lr_values[-1]:.8f}")

# ==================== 生成学习率曲线图 ====================

# 图1：原始 + 平滑曲线
fig1, ax1 = plt.subplots(figsize=(12, 6))

# 平滑
lr_smooth = smooth_data(lr_values)

ax1.plot(steps, lr_values, 'b-', linewidth=0.5, alpha=0.3, label='Raw Learning Rate')
ax1.plot(steps, lr_smooth, 'r-', linewidth=2, label='Smoothed Learning Rate (S-G filter)')

ax1.set_xlabel('Training Steps', fontsize=12)
ax1.set_ylabel('Learning Rate', fontsize=12)
ax1.set_title('Learning Rate Curve (Savitzky-Golay Smoothed)', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3, linestyle='--')

# 标注初始学习率和最终学习率
ax1.annotate(f'Start: {lr_values[0]:.2e}', 
             xy=(steps[0], lr_values[0]), 
             xytext=(steps[0] + 2000, lr_values[0] + 0.0001),
             arrowprops=dict(arrowstyle='->', color='green'),
             fontsize=10, color='green')
ax1.annotate(f'End: {lr_values[-1]:.2e}', 
             xy=(steps[-1], lr_values[-1]), 
             xytext=(steps[-1] - 15000, lr_values[-1] + 0.00005),
             arrowprops=dict(arrowstyle='->', color='green'),
             fontsize=10, color='green')

plt.tight_layout()
save_path1 = os.path.join(data_dir, 'lr_smooth.png')
plt.savefig(save_path1, dpi=300, bbox_inches='tight')
plt.close()
print(f"\n已保存: {save_path1}")

# 图2：仅平滑曲线（论文推荐）
fig2, ax2 = plt.subplots(figsize=(12, 6))

ax2.plot(steps, lr_smooth, '#2E86AB', linewidth=2)
ax2.set_xlabel('Training Steps', fontsize=12)
ax2.set_ylabel('Learning Rate', fontsize=12)
ax2.set_title('Learning Rate Curve', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')

# 标注关键阶段
# Warmup阶段（前10%步数）
warmup_steps = int(len(steps) * 0.1)
if warmup_steps > 0:
    warmup_end_step = steps[warmup_steps]
    warmup_end_lr = lr_smooth[warmup_steps]
    ax2.axvline(x=warmup_end_step, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Warmup End')
    ax2.annotate('Warmup End', xy=(warmup_end_step, warmup_end_lr), 
                 xytext=(warmup_end_step + 2000, warmup_end_lr + 0.00005),
                 arrowprops=dict(arrowstyle='->', color='orange'),
                 fontsize=9, color='orange')

max_lr_step = steps[np.argmax(lr_smooth)]
max_lr = np.max(lr_smooth)
ax2.scatter([max_lr_step], [max_lr], color='red', s=50, zorder=5, label=f'Peak: {max_lr:.2e}')
ax2.annotate(f'Peak LR: {max_lr:.2e}', 
             xy=(max_lr_step, max_lr), 
             xytext=(max_lr_step + 2000, max_lr + 0.00002),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=9, color='red')

ax2.legend(loc='upper right')
plt.tight_layout()
save_path2 = os.path.join(data_dir, 'lr_clean.png')
plt.savefig(save_path2, dpi=300, bbox_inches='tight')
plt.close()
print(f"已保存: {save_path2}")

# 图3：学习率 + 总损失对比图（可选）
loss_file = os.path.join(data_dir, 'loss_total.json')
if os.path.exists(loss_file):
    with open(loss_file, 'r') as f:
        loss_data = json.load(f)
    loss_steps = [item[1] for item in loss_data]
    loss_values = [item[2] for item in loss_data]
    loss_smooth = smooth_data(loss_values)
    
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    
    # 双y轴
    ax3.set_xlabel('Training Steps', fontsize=12)
    ax3.set_ylabel('Learning Rate', color='#2E86AB', fontsize=12)
    ax3.plot(steps, lr_smooth, color='#2E86AB', linewidth=2, label='Learning Rate')
    ax3.tick_params(axis='y', labelcolor='#2E86AB')
    
    ax4 = ax3.twinx()
    ax4.set_ylabel('Total Loss', color='#C73E1D', fontsize=12)
    ax4.plot(loss_steps, loss_smooth, color='#C73E1D', linewidth=2, label='Total Loss')
    ax4.tick_params(axis='y', labelcolor='#C73E1D')
    
    ax3.set_title('Learning Rate and Total Loss Curves', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # 添加图例
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax4.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    save_path3 = os.path.join(data_dir, 'lr_loss_combined.png')
    plt.savefig(save_path3, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {save_path3}")

print("\n" + "=" * 50)
print("学习率图片生成完成!")
print("=" * 50)
print("\n生成的文件:")
print(f"  - {save_path1} (原始+平滑)")
print(f"  - {save_path2} (仅平滑曲线，论文推荐)")
if os.path.exists(loss_file):
    print(f"  - {save_path3} (学习率+总损失对比)")