"""
Generate figures for the Kakeya Ray Tracing paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Load results
results_dir = Path(__file__).parent.parent / 'results'
figures_dir = Path(__file__).parent.parent / 'paper' / 'figures'
figures_dir.mkdir(exist_ok=True)

with open(results_dir / 'benchmark_results.json', 'r') as f:
    results = json.load(f)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#76b900', '#0070c0', '#ff6b6b']  # NVIDIA green, blue, red

# Figure 1: Load Imbalance Comparison
fig, ax = plt.subplots(figsize=(10, 6))

scenes = [r['scene'].replace('_', ' ').title() for r in results]
uniform_imbalance = [r['methods']['uniform']['imbalance'] for r in results]
kakeya_imbalance = [r['methods']['kakeya_fixed']['imbalance'] for r in results]

x = np.arange(len(scenes))
width = 0.35

bars1 = ax.bar(x - width/2, uniform_imbalance, width, label='Uniform Tiling', color=colors[2], alpha=0.8)
bars2 = ax.bar(x + width/2, kakeya_imbalance, width, label='Kakeya Partitioning', color=colors[0], alpha=0.8)

ax.set_xlabel('Scene Configuration', fontsize=12, fontweight='bold')
ax.set_ylabel('Load Imbalance (lower is better)', fontsize=12, fontweight='bold')
ax.set_title('Load Imbalance: Uniform vs Kakeya Partitioning', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(scenes, rotation=15, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}x',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(figures_dir / 'load_imbalance.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved load_imbalance.png")

# Figure 2: GPU Utilization
fig, ax = plt.subplots(figsize=(10, 6))

uniform_util = [r['methods']['uniform']['utilization'] for r in results]
kakeya_util = [r['methods']['kakeya_fixed']['utilization'] for r in results]

bars1 = ax.bar(x - width/2, uniform_util, width, label='Uniform Tiling', color=colors[2], alpha=0.8)
bars2 = ax.bar(x + width/2, kakeya_util, width, label='Kakeya Partitioning', color=colors[0], alpha=0.8)

ax.set_xlabel('Scene Configuration', fontsize=12, fontweight='bold')
ax.set_ylabel('GPU Utilization % (higher is better)', fontsize=12, fontweight='bold')
ax.set_title('GPU Utilization: Uniform vs Kakeya Partitioning', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(scenes, rotation=15, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 110)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(figures_dir / 'gpu_utilization.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved gpu_utilization.png")

# Figure 3: Partition Overhead vs Render Time
fig, ax = plt.subplots(figsize=(10, 6))

render_times = [r['render_time'] for r in results]
partition_times = [r['methods']['kakeya_fixed']['partition_time'] * 1000 for r in results]  # ms

ax.plot(render_times, partition_times, 'o-', color=colors[0], linewidth=2, markersize=8, label='Partition Overhead')

# Add annotations
for i, (rt, pt) in enumerate(zip(render_times, partition_times)):
    ax.annotate(f'{pt:.1f}ms', (rt, pt), textcoords="offset points", xytext=(0,10),
                ha='center', fontsize=9)

ax.set_xlabel('Render Time (seconds)', fontsize=12, fontweight='bold')
ax.set_ylabel('Partition Overhead (milliseconds)', fontsize=12, fontweight='bold')
ax.set_title('Partition Overhead vs Render Time', fontsize=14, fontweight='bold', pad=20)
ax.grid(alpha=0.3)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(figures_dir / 'partition_overhead.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved partition_overhead.png")

# Figure 4: Speedup Analysis
fig, ax = plt.subplots(figsize=(10, 6))

speedups = [r['methods']['kakeya_fixed']['speedup'] for r in results]

bars = ax.bar(scenes, speedups, color=colors[0], alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_xlabel('Scene Configuration', fontsize=12, fontweight='bold')
ax.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
ax.set_title('Estimated Speedup with Kakeya Partitioning', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(scenes, rotation=15, ha='right')
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Baseline (1.0x)')
ax.legend(fontsize=11)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}x',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(figures_dir / 'speedup.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved speedup.png")

# Figure 5: Workload Distribution Heatmap (synthetic example)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Create synthetic workload with high variance
np.random.seed(42)
workload = np.random.poisson(5, (100, 100))
# Add some hotspots
workload[20:40, 30:50] = 20
workload[60:80, 60:80] = 25

# Uniform partition (4x4 grid)
uniform_grid = np.zeros((100, 100), dtype=int)
for i in range(4):
    for j in range(4):
        uniform_grid[i*25:(i+1)*25, j*25:(j+1)*25] = i*4 + j

# Kakeya partition (simplified - workload-aware regions)
kakeya_grid = np.zeros((100, 100), dtype=int)
# Hotspots get their own cells
kakeya_grid[20:40, 30:50] = 1
kakeya_grid[60:80, 60:80] = 2
# Rest distributed
kakeya_grid[:20, :] = 3
kakeya_grid[80:, :] = 4
kakeya_grid[40:60, :30] = 5
kakeya_grid[40:60, 80:] = 6

im0 = axes[0].imshow(workload, cmap='hot', aspect='equal')
axes[0].set_title('Workload Distribution', fontsize=11, fontweight='bold')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
plt.colorbar(im0, ax=axes[0], label='Workload')

im1 = axes[1].imshow(uniform_grid, cmap='tab20', aspect='equal')
axes[1].set_title('Uniform Partition (16 cells)', fontsize=11, fontweight='bold')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')

im2 = axes[2].imshow(kakeya_grid, cmap='tab20', aspect='equal')
axes[2].set_title('Kakeya Partition (6 cells)', fontsize=11, fontweight='bold')
axes[2].set_xlabel('X')
axes[2].set_ylabel('Y')

plt.suptitle('Partitioning Strategy Comparison', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(figures_dir / 'partitioning_strategies.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved partitioning_strategies.png")

print("\nAll figures generated successfully!")
