"""
Benchmarking experiments for Kakeya polynomial partitioning in ray tracing.

This script compares:
1. Naive uniform tiling (baseline)
2. Kakeya workload-aware partitioning
3. Adaptive degree partitioning

Metrics:
- Load imbalance (max/avg workload ratio)
- GPU utilization estimate
- Partition overhead
- Effective speedup
"""

import numpy as np
import time
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from ray_tracer import RayTracer
from kakeya_partition import KakeyaPartitioner, AdaptiveDegreePartitioner


def compute_load_balance(cells, total_work):
    """Compute load balance metrics."""
    if not cells:
        return {'imbalance': 1.0, 'utilization': 100.0}
    
    cell_workloads = [cell.workload for cell in cells]
    max_work = max(cell_workloads)
    avg_work = sum(cell_workloads) / len(cell_workloads)
    imbalance = max_work / avg_work if avg_work > 0 else 1.0
    utilization = min(100.0, (1.0 / imbalance) * 100.0)
    
    return {
        'imbalance': float(imbalance),
        'utilization': float(utilization),
        'max_workload': float(max_work),
        'avg_workload': float(avg_work),
        'num_cells': len(cells)
    }


def uniform_tiling(workloads, width, height, num_tiles=16):
    """Baseline: uniform grid tiling."""
    # Simple grid split
    tile_w = width // int(np.sqrt(num_tiles))
    tile_h = height // int(np.sqrt(num_tiles))
    
    cells = []
    for ty in range(int(np.sqrt(num_tiles))):
        for tx in range(int(np.sqrt(num_tiles))):
            x0 = tx * tile_w
            y0 = ty * tile_h
            x1 = min(x0 + tile_w, width)
            y1 = min(y0 + tile_h, height)
            
            # Extract workload for this tile
            tile_work = workloads[y0:y1, x0:x1].sum()
            
            # Create cell
            rays = []
            for y in range(y0, y1):
                for x in range(x0, x1):
                    rays.append(y * width + x)
            
            cells.append(type('Cell', (), {
                'rays': rays,
                'workload': tile_work,
                'bounds': (x0, y0, x1, y1)
            })())
    
    return cells, 0.0  # No partition overhead


def benchmark_scene(name, tracer, max_bounces, target_cells=16):
    """Benchmark a single scene configuration."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"Resolution: {tracer.width}x{tracer.height}, Bounces: {max_bounces}")
    print(f"{'='*60}")
    
    # Render scene
    print("Rendering scene...")
    image, workloads, render_time = tracer.render(max_bounces=max_bounces)
    work_2d = workloads.reshape(tracer.height, tracer.width)
    total_work = workloads.sum()
    
    print(f"Render time: {render_time:.3f}s")
    print(f"Total workload: {total_work:.0f}")
    print(f"Workload range: [{workloads.min():.1f}, {workloads.max():.1f}]")
    
    results = {
        'scene': name,
        'resolution': f"{tracer.width}x{tracer.height}",
        'max_bounces': max_bounces,
        'render_time': render_time,
        'total_workload': float(total_work),
        'methods': {}
    }
    
    # Method 1: Uniform tiling (baseline)
    print("\n[1/3] Uniform tiling (baseline)...")
    start = time.time()
    cells, _ = uniform_tiling(work_2d, tracer.width, tracer.height, target_cells)
    uniform_time = time.time() - start
    
    metrics = compute_load_balance(cells, total_work)
    metrics['partition_time'] = uniform_time
    
    print(f"  Partition time: {uniform_time*1000:.2f}ms")
    print(f"  Load imbalance: {metrics['imbalance']:.2f}x")
    print(f"  GPU utilization: {metrics['utilization']:.1f}%")
    
    results['methods']['uniform'] = metrics
    
    # Method 2: Kakeya partitioning (fixed degree)
    print("\n[2/3] Kakeya partitioning (degree=3)...")
    partitioner = KakeyaPartitioner(degree=3)
    cells, partition_time = partitioner.partition(
        workloads, tracer.width, tracer.height, target_cells
    )
    
    metrics = compute_load_balance(cells, total_work)
    metrics['partition_time'] = partition_time
    metrics['degree'] = 3
    
    print(f"  Partition time: {partition_time*1000:.2f}ms")
    print(f"  Load imbalance: {metrics['imbalance']:.2f}x")
    print(f"  GPU utilization: {metrics['utilization']:.1f}%")
    
    results['methods']['kakeya_fixed'] = metrics
    
    # Method 3: Adaptive degree partitioning
    print("\n[3/3] Adaptive degree partitioning...")
    adaptive = AdaptiveDegreePartitioner()
    cells, partition_time, selected_degree = adaptive.adaptive_partition(
        workloads, tracer.width, tracer.height, target_cells
    )
    
    metrics = compute_load_balance(cells, total_work)
    metrics['partition_time'] = partition_time
    metrics['degree'] = selected_degree
    
    print(f"  Partition time: {partition_time*1000:.2f}ms")
    print(f"  Selected degree: {selected_degree}")
    print(f"  Load imbalance: {metrics['imbalance']:.2f}x")
    print(f"  GPU utilization: {metrics['utilization']:.1f}%")
    
    results['methods']['kakeya_adaptive'] = metrics
    
    # Compute speedup estimates
    baseline_time = render_time  # Assume render time dominates
    for method_name in ['uniform', 'kakeya_fixed', 'kakeya_adaptive']:
        method = results['methods'][method_name]
        imbalance = method['imbalance']
        partition_time = method['partition_time']
        
        # Effective time = (render_time * imbalance) + partition_time
        # Speedup = baseline_time / effective_time
        effective_time = (render_time / imbalance) + partition_time
        speedup = baseline_time / effective_time if effective_time > 0 else 1.0
        
        method['speedup'] = float(speedup)
        method['effective_time'] = float(effective_time)
    
    return results


def main():
    """Run all benchmark experiments."""
    print("="*60)
    print("Kakeya Ray Tracing Partitioning Benchmarks")
    print("="*60)
    
    # Create results directory
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # Test configurations
    configs = [
        # (name, width, height, max_bounces)
        ('low_res_simple', 200, 150, 3),
        ('low_res_complex', 200, 150, 5),
        ('medium_res_simple', 400, 300, 3),
        ('medium_res_complex', 400, 300, 5),
        ('high_res_simple', 600, 450, 3),
        ('high_res_complex', 600, 450, 5),
    ]
    
    all_results = []
    
    for name, width, height, bounces in configs:
        tracer = RayTracer(width, height)
        results = benchmark_scene(name, tracer, bounces, target_cells=16)
        all_results.append(results)
    
    # Save results
    output_file = results_dir / 'benchmark_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"\n{'Scene':<25} {'Uniform':<15} {'Kakeya':<15} {'Adaptive':<15}")
    print("-"*70)
    
    for result in all_results:
        scene = result['scene']
        uniform = result['methods']['uniform']
        kakeya = result['methods']['kakeya_fixed']
        adaptive = result['methods']['kakeya_adaptive']
        
        print(f"{scene:<25} "
              f"{uniform['imbalance']:>5.2f}x/{uniform['utilization']:>4.0f}% "
              f"{kakeya['imbalance']:>5.2f}x/{kakeya['utilization']:>4.0f}% "
              f"{adaptive['imbalance']:>5.2f}x/{adaptive['utilization']:>4.0f}%")
    
    print("\nLegend: Imbalance (lower=better) / Utilization (higher=better)")


if __name__ == '__main__':
    main()
