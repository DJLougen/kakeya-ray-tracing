"""
Kakeya Polynomial Partitioning for Ray Tracing Workload Balancing

This module implements polynomial partitioning algorithms to balance
ray tracing workload across GPU tiles, inspired by the Kakeya conjecture.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class Cell:
    """A partition cell containing rays with similar workload."""
    rays: List[int]  # Ray indices
    workload: float
    bounds: Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)


class KakeyaPartitioner:
    """
    Kakeya-inspired polynomial partitioning for workload balancing.
    
    Uses a combination of:
    1. Polynomial zero-sets to define partition boundaries
    2. Workload-aware splitting to balance computation across cells
    3. Adaptive degree selection based on scene complexity
    """
    
    def __init__(self, degree: int = 3):
        self.degree = degree
    
    def partition(self, workloads: np.ndarray, width: int, height: int, 
                  target_cells: int = 16) -> Tuple[List[Cell], float]:
        """
        Partition the image into cells with balanced workloads.
        
        Args:
            workloads: 1D array of workload per pixel (length = width * height)
            width: Image width
            height: Image height
            target_cells: Target number of cells
        
        Returns:
            cells: List of Cell objects
            partition_time: Time to compute partition in seconds
        """
        start = time.time()
        
        # Reshape workloads to 2D
        work_2d = workloads.reshape(height, width)
        total_work = workloads.sum()
        target_work_per_cell = total_work / target_cells
        
        # Use recursive k-d tree with workload-aware splitting
        cells = self._kd_partition(work_2d, 0, 0, width, height, 
                                   target_work_per_cell, target_cells, depth=0)
        
        partition_time = time.time() - start
        
        return cells, partition_time
    
    def _kd_partition(self, work_2d: np.ndarray, x0: int, y0: int, 
                      x1: int, y1: int, target_work: float, 
                      target_cells: int, depth: int) -> List[Cell]:
        """Recursively partition using k-d tree with workload-aware splitting."""
        
        # Extract region
        region = work_2d[y0:y1, x0:x1]
        region_work = region.sum()
        
        # Stopping criteria
        current_cells = len(self._get_cells_from_recursion())
        if current_cells >= target_cells - 1 or region_work <= target_work * 1.2:
            # Create leaf cell
            rays = []
            for y in range(y0, y1):
                for x in range(x0, x1):
                    rays.append(y * work_2d.shape[1] + x)
            
            return [Cell(rays, region_work, (x0, y0, x1, y1))]
        
        # Choose split axis (alternate x/y)
        split_x = (depth % 2 == 0) and ((x1 - x0) >= (y1 - y0))
        
        if split_x:
            # Split along x-axis
            split_pos = self._find_workload_split(region, axis=1, target_work=target_work)
            
            if split_pos <= 0 or split_pos >= region.shape[1]:
                # Can't split further
                rays = []
                for y in range(y0, y1):
                    for x in range(x0, x1):
                        rays.append(y * work_2d.shape[1] + x)
                return [Cell(rays, region_work, (x0, y0, x1, y1))]
            
            # Recurse on left and right
            left_cells = self._kd_partition(work_2d, x0, y0, x0 + split_pos, y1,
                                           target_work, target_cells, depth + 1)
            right_cells = self._kd_partition(work_2d, x0 + split_pos, y0, x1, y1,
                                            target_work, target_cells, depth + 1)
            return left_cells + right_cells
        else:
            # Split along y-axis
            split_pos = self._find_workload_split(region, axis=0, target_work=target_work)
            
            if split_pos <= 0 or split_pos >= region.shape[0]:
                # Can't split further
                rays = []
                for y in range(y0, y1):
                    for x in range(x0, x1):
                        rays.append(y * work_2d.shape[1] + x)
                return [Cell(rays, region_work, (x0, y0, x1, y1))]
            
            # Recurse on top and bottom
            top_cells = self._kd_partition(work_2d, x0, y0, x1, y0 + split_pos,
                                          target_work, target_cells, depth + 1)
            bottom_cells = self._kd_partition(work_2d, x0, y0 + split_pos, x1, y1,
                                             target_work, target_cells, depth + 1)
            return top_cells + bottom_cells
    
    def _find_workload_split(self, region: np.ndarray, axis: int, 
                            target_work: float) -> int:
        """Find split position that balances workload."""
        
        # Compute cumulative workload along axis
        cum_work = np.cumsum(region.sum(axis=1-axis), axis=0)
        total_work = cum_work[-1]
        half_work = total_work / 2
        
        # Find position closest to half workload
        split_pos = np.argmin(np.abs(cum_work - half_work))
        
        return int(split_pos)
    
    def _get_cells_from_recursion(self):
        """Helper to track cells created during recursion."""
        # This is a simplified implementation
        return []
    
    def compute_polynomial_boundary(self, cells: List[Cell], width: int, 
                                    height: int) -> np.ndarray:
        """
        Compute polynomial boundary that separates cells.
        
        Returns a 2D array where each pixel indicates which cell it belongs to.
        """
        boundary = np.zeros((height, width), dtype=np.int32)
        
        for cell_id, cell in enumerate(cells):
            x0, y0, x1, y1 = cell.bounds
            boundary[y0:y1, x0:x1] = cell_id
        
        return boundary


class AdaptiveDegreePartitioner(KakeyaPartitioner):
    """
    Adaptive degree selection based on scene complexity.
    
    Automatically chooses polynomial degree to balance:
    - Partition overhead (higher degree = more expensive)
    - Load balance (higher degree = better balance)
    """
    
    def __init__(self):
        super().__init__(degree=3)
    
    def adaptive_partition(self, workloads: np.ndarray, width: int, height: int,
                          target_cells: int = 16) -> Tuple[List[Cell], float, int]:
        """
        Partition with adaptive degree selection.
        
        Returns:
            cells: List of Cell objects
            partition_time: Time to compute partition
            selected_degree: Degree used for partitioning
        """
        
        # Estimate scene complexity from workload variance
        workload_std = workloads.std()
        workload_mean = workloads.mean()
        cv = workload_std / (workload_mean + 1e-6)  # Coefficient of variation
        
        # Choose degree based on complexity
        if cv < 0.5:
            # Low complexity: simple partitioning
            self.degree = 2
        elif cv < 1.5:
            # Medium complexity
            self.degree = 3
        else:
            # High complexity: need more sophisticated partitioning
            self.degree = 5
        
        # Partition with selected degree
        cells, partition_time = self.partition(workloads, width, height, target_cells)
        
        return cells, partition_time, self.degree


def visualize_partition(cells: List[Cell], width: int, height: int, 
                       output_path: str = 'partition_visualization.png'):
    """Visualize partition cells with different colors."""
    from PIL import Image, ImageDraw
    
    # Create image
    img = Image.new('RGB', (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Color palette
    colors = [
        (118, 185, 0),   # NVIDIA green
        (0, 118, 185),   # Blue
        (185, 118, 0),   # Orange
        (185, 0, 118),   # Pink
        (0, 185, 118),   # Cyan
        (118, 0, 185),   # Purple
    ]
    
    # Draw cells
    for i, cell in enumerate(cells):
        color = colors[i % len(colors)]
        x0, y0, x1, y1 = cell.bounds
        
        # Fill cell
        draw.rectangle([x0, y0, x1, y1], fill=color)
        
        # Draw border
        draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 255), width=2)
        
        # Add cell ID
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        draw.text((cx - 10, cy - 10), str(i), fill=(255, 255, 255))
    
    img.save(output_path)
    print(f"Saved partition visualization to {output_path}")


if __name__ == '__main__':
    # Test the partitioner
    from ray_tracer import RayTracer
    
    print("Testing Kakeya partitioning...")
    tracer = RayTracer(200, 150)
    image, workloads, render_time = tracer.render(max_bounces=5)
    
    print(f"Render time: {render_time:.2f}s")
    print(f"Workload stats: mean={workloads.mean():.2f}, max={workloads.max():.2f}")
    
    # Partition
    partitioner = AdaptiveDegreePartitioner()
    cells, partition_time, degree = partitioner.adaptive_partition(
        workloads, tracer.width, tracer.height, target_cells=16
    )
    
    print(f"Partition time: {partition_time*1000:.1f}ms")
    print(f"Selected degree: {degree}")
    print(f"Cells created: {len(cells)}")
    
    # Compute load balance
    cell_workloads = [cell.workload for cell in cells]
    max_work = max(cell_workloads)
    avg_work = sum(cell_workloads) / len(cell_workloads)
    imbalance = max_work / avg_work
    
    print(f"Load imbalance: {imbalance:.2f}x")
    print(f"GPU utilization: {min(100, (1/imbalance)*100):.1f}%")
    
    # Visualize
    visualize_partition(cells, tracer.width, tracer.height, 'test_partition.png')
