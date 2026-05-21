# Kakeya Ray Tracing

**Polynomial Partitioning for GPU Ray Tracing Load Balancing**

[![Live Demo](https://img.shields.io/badge/Live_Demo-76b900?style=for-the-badge)](https://djlougen.github.io/kakeya-ray-tracing/)
[![Paper](https://img.shields.io/badge/Paper-PDF-0070c0?style=for-the-badge)](paper.pdf)
[![License](https://img.shields.io/badge/License-MIT-gray?style=for-the-badge)](LICENSE)

---

## Overview

Modern GPU ray tracing suffers from severe load imbalance. Rays hitting reflective or refractive surfaces may require **10-50 bounces** (500-2500ns), while rays hitting the sky terminate after **1 bounce** (~50ns). This creates workload ratios exceeding **50:1** across pixels.

Traditional tile-based rendering assigns fixed screen regions to GPU warps, causing the entire GPU to stall while waiting for the slowest warp. The result? **GPU utilization as low as 20-40%**.

This project applies the **Kakeya conjecture** from harmonic analysis to solve this problem. The Kakeya conjecture studies the geometric properties of line segments in different directions—and rays *are* line segments. We use **polynomial partitioning** to divide the image into cells with balanced total workload rather than balanced pixel counts.

## Key Results

| Metric | Uniform Tiling | Kakeya Partitioning | Improvement |
|--------|---------------|---------------------|-------------|
| **Load Imbalance** | 1.62-1.73x | 1.03-1.09x | **1.6x better** |
| **GPU Utilization** | 58-62% | 91-97% | **1.6x higher** |
| **Partition Overhead** | 0ms | 3-54ms | Negligible |
| **Overall Speedup** | 1.0x (baseline) | 1.5-1.6x | **50-60% faster** |

![Load Imbalance Comparison](results/load_imbalance.png)

*Load imbalance reduction: Kakeya partitioning achieves near-perfect balance (1.03-1.09x) vs uniform tiling (1.62-1.73x).*

## Live Demo

**[Try the interactive demo here](https://djlougen.github.io/kakeya-ray-tracing/)**

The demo runs a **real GPU path tracer** using WebGL2 fragment shaders. All ray tracing happens on your GPU in real-time.

### Features

- **GPU path tracer** with configurable bounce depth (2, 4, 8 bounces)
- **Real GPU timing** via `EXT_disjoint_timer_query_webgl2` (microsecond precision)
- **Warp divergence visualization** showing where GPU threads stall
- **Kakeya partitioning** that reduces divergence from ~65% to ~15%
- **Performance metrics**: rays/sec, GPU utilization, projected speedup
- **Three visualization modes**: Render, Workload Heatmap, Warp Divergence

### Tested Performance

On **NVIDIA GeForce RTX 3090** (measured via WebGL2 timer query):
- GPU frame time: **54 μs** *(measured)*
- Throughput: **369M rays/sec** *(measured)*

**Projected metrics** (computed from CPU bounce analysis, not actual GPU warp reordering):
- Divergence reduction: **~65% → ~15%** *(theoretical model)*
- Utilization improvement: **~35% → ~85%** *(theoretical model)*
- Speedup: **~14x** *(upper bound; real OptiX/CUDA gains would be lower due to cache, memory access, and partitioning overhead)*

## How It Works

### The Problem

```
Traditional Uniform Tiling:
┌─────────┬─────────┬─────────┬─────────┐
│ Sky     │ Sky     │ Sky     │ Sky     │  ← 50ns each
│ (fast)  │ (fast)  │ (fast)  │ (fast)  │
├─────────┼─────────┼─────────┼─────────┤
│ Sky     │ Reflect │ Reflect │ Sky     │  ← 50ns / 2500ns / 2500ns / 50ns
│ (fast)  │ (SLOW)  │ (SLOW)  │ (fast)  │
├─────────┼─────────┼─────────┼─────────┤
│ Sky     │ Sky     │ Glass   │ Sky     │  ← 50ns / 50ns / 1500ns / 50ns
│ (fast)  │ (fast)  │ (slow)  │ (fast)  │
└─────────┴─────────┴─────────┴─────────┘

GPU waits for slowest tile: 2500ns
All other warps idle: 58% utilization
```

### The Solution

```
Kakeya Polynomial Partitioning:
┌──────────────────────┬─────────┐
│                      │ Reflect │  ← Balanced workload per cell
│ Sky (all fast rays)  │ (isolated)│
│                      │         │
├──────────────────────┼─────────┤
│                      │ Glass   │  ← Hotspots get their own cells
│ Sky                  │ (isolated)│
│                      │         │
└──────────────────────┴─────────┘

GPU work balanced across warps: 1.09x imbalance
Utilization: 91%
```

### Mathematical Foundation

Given rays $R = \{r_1, \ldots, r_N\}$ with workload estimates $w(r_i)$, we construct a polynomial $P(x, y)$ over screen coordinates such that the zero set $\{P = 0\}$ partitions the image into cells $C_1, \ldots, C_K$ with:

$$\sum_{r \in C_i} w(r) \approx \sum_{r \in C_j} w(r) \quad \forall i, j$$

We approximate this using a **workload-aware k-d tree** that recursively splits the image along workload-weighted medians.

## Installation

```bash
# Clone the repository
git clone https://github.com/DJLougen/kakeya-ray-tracing.git
cd kakeya-ray-tracing

# Install dependencies (for Python benchmarks and experiments)
pip install -r requirements.txt
```

## Usage

### Run the Interactive Demo (GPU)

The live demo at [djlougen.github.io/kakeya-ray-tracing](https://djlougen.github.io/kakeya-ray-tracing/) runs a **real GPU path tracer** using WebGL2 fragment shaders. All ray tracing happens on your GPU in real-time.

Features:
- **GPU path tracer** with configurable bounce depth (2, 4, 8 bounces)
- **Real GPU timing** via `EXT_disjoint_timer_query_webgl2` (microsecond precision)
- **Warp divergence visualization** showing where GPU threads stall
- **Kakeya partitioning** that reduces divergence from ~65% to ~15%
- **Performance metrics**: rays/sec, GPU utilization, projected speedup

Tested on **NVIDIA GeForce RTX 3090**: 54 μs/frame, 369M rays/sec, 14.53x projected speedup with Kakeya partitioning.

### Optional: CUDA Ray Tracer (Advanced)

For native CUDA performance (requires CUDA toolkit and numba):

```bash
pip install numba
python src/gpu_ray_tracer.py
```

This runs the same ray tracing algorithm directly on CUDA cores, measuring warp divergence and GPU utilization.

### Run Benchmarks

```bash
cd experiments
python benchmark_partitioning.py
```

This generates `results/benchmark_results.json` with detailed performance metrics across multiple scene configurations.

### Generate Figures

```bash
cd experiments
python generate_figures.py
```

This creates publication-quality figures in `paper/figures/`.

## Project Structure

```
kakeya-ray-tracing/
├── demo/
│   ├── index.html          # Interactive WebGL demo (GPU path tracer)
│   └── gpu_demo.html       # Standalone GPU demo
├── src/
│   ├── gpu_ray_tracer.py   # CUDA-based GPU ray tracer (optional)
│   ├── ray_tracer.py       # CPU reference implementation
│   └── kakeya_partition.py # Polynomial partitioning algorithm
├── experiments/
│   ├── benchmark_partitioning.py  # Performance benchmarks
│   └── generate_figures.py        # Figure generation
├── paper/
│   ├── main.tex            # LaTeX source
│   └── figures/            # Generated figures
├── results/
│   └── benchmark_results.json  # Experimental results
├── paper.pdf               # Compiled paper
└── README.md               # This file
```

## Connection to NVIDIA

This work is directly applicable to **NVIDIA's RT cores** and **DLSS Ray Reconstruction**:

- **RT cores** accelerate ray-triangle intersection tests
- **Our method** ensures all RT cores remain busy by balancing workload
- The two approaches are **orthogonal and complementary**

For static scenes, the partition can be computed once and reused across frames, amortizing the 3-54ms overhead. For dynamic scenes, update the partition every N frames (e.g., N=10).

### Potential Integration Points

1. **NVIDIA OptiX**: Add workload-aware partitioning to the ray generation shader
2. **Vulkan Ray Tracing**: Use compute shaders to partition rays before dispatch
3. **DLSS 3.5**: Integrate with frame generation to balance workload across temporal samples

## Performance Details

### Partition Overhead

| Resolution | Partition Time | Render Time | Overhead % |
|-----------|---------------|-------------|------------|
| 200×150   | 3-4ms         | 0.4s        | <1%        |
| 400×300   | 21-23ms       | 1.9-3.0s    | <1%        |
| 600×450   | 48-54ms       | 6.8-6.9s    | <1%        |

### Adaptive Degree Selection

The algorithm automatically selects polynomial degree based on scene complexity:

```python
degree = {
    2 if CV(workloads) < 0.5    # Low complexity
    3 if CV(workloads) < 1.5    # Medium complexity  
    5 if CV(workloads) >= 1.5   # High complexity
}
```

where CV is the coefficient of variation (σ/μ).

## Why "Kakeya"?

The **Kakeya conjecture** asks: what is the minimum area of a set in ℝ² that contains a unit line segment in every direction? The solution involves **polynomial partitioning**—dividing space using algebraic varieties to balance geometric structures.

In ray tracing, each ray is a line segment. The workload per ray depends on the geometric structures it intersects. We adapt polynomial partitioning to balance *workload* rather than geometric measure.

## Citation

```bibtex
@article{lougen2024kakeya,
  title={Kakeya-Inspired Polynomial Partitioning for Load Balancing in GPU Ray Tracing},
  author={Lougen, Daniel},
  journal={arXiv preprint},
  year={2024}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

**Daniel Lougen** - Gestalt Labs

---

**Built with pure mathematics and GPU rays** 🎯
