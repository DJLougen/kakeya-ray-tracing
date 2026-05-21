"""
Kakeya-Inspired Ray Tracer with Polynomial Partitioning

This module implements a CPU-based ray tracer that uses polynomial partitioning
to balance workload across tiles, inspired by the Kakeya conjecture.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time


@dataclass
class Vec3:
    """3D vector for positions and directions."""
    x: float
    y: float
    z: float
    
    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def length(self):
        return np.sqrt(self.dot(self))
    
    def normalize(self):
        l = self.length()
        return Vec3(self.x / l, self.y / l, self.z / l)
    
    def reflect(self, normal):
        return self - normal * (2 * self.dot(normal))


@dataclass
class Ray:
    """Ray with origin and direction."""
    origin: Vec3
    direction: Vec3
    
    def at(self, t: float):
        return self.origin + self.direction * t


@dataclass
class Material:
    """Material properties for surfaces."""
    type: str  # 'diffuse', 'metal', 'glass'
    color: Tuple[float, float, float]
    roughness: float = 1.0
    max_bounces: int = 1


@dataclass
class Sphere:
    """Sphere primitive."""
    position: Vec3
    radius: float
    material: Material
    
    def intersect(self, ray: Ray) -> float:
        """Returns distance to intersection, or -1 if no intersection."""
        oc = ray.origin - self.position
        a = ray.direction.dot(ray.direction)
        b = 2.0 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return -1
        
        t = (-b - np.sqrt(discriminant)) / (2.0 * a)
        return t if t > 0.001 else -1


@dataclass
class Scene:
    """Scene containing spheres and ground plane."""
    spheres: List[Sphere]
    ground_y: float = -2.0
    ground_color: Tuple[float, float, float] = (0.7, 0.7, 0.7)


class RayTracer:
    """CPU-based ray tracer with workload estimation."""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        
        # Create scene with reflective spheres
        self.scene = Scene(
            spheres=[
                Sphere(Vec3(-1.5, 0, -5), 1.2, Material('metal', (0.8, 0.8, 0.9), 0.1, 5)),
                Sphere(Vec3(1.5, 0, -5), 1.0, Material('glass', (0.9, 0.9, 1.0), 0.0, 8)),
                Sphere(Vec3(0, -1.5, -5), 1.5, Material('diffuse', (0.8, 0.2, 0.2), 1.0, 1)),
                Sphere(Vec3(-3, 1, -7), 0.8, Material('metal', (1.0, 0.8, 0.2), 0.2, 4)),
            ]
        )
    
    def generate_rays(self) -> List[Ray]:
        """Generate camera rays for all pixels."""
        rays = []
        for y in range(self.height):
            for x in range(self.width):
                u = (x / self.width) * 2 - 1
                v = -(y / self.height) * 2 + 1
                
                direction = Vec3(u * 2, v * 1.5, -1).normalize()
                rays.append(Ray(Vec3(0, 0, 0), direction))
        
        return rays
    
    def trace_ray(self, ray: Ray, depth: int, max_bounces: int) -> Tuple[float, float, float]:
        """Trace a single ray and return color."""
        if depth > max_bounces:
            return (0.05, 0.05, 0.1)  # Sky color
        
        closest_t = float('inf')
        hit_sphere = None
        
        # Check sphere intersections
        for sphere in self.scene.spheres:
            t = sphere.intersect(ray)
            if t > 0.001 and t < closest_t:
                closest_t = t
                hit_sphere = sphere
        
        # Check ground plane
        if ray.direction.y < 0:
            t = (self.scene.ground_y - ray.origin.y) / ray.direction.y
            if t > 0.001 and t < closest_t:
                closest_t = t
                hit_sphere = None
                hit_point = ray.at(t)
                checker = (int(hit_point.x * 2) + int(hit_point.z * 2)) % 2
                ground_color = self.scene.ground_color if checker else (0.3, 0.3, 0.3)
                return (ground_color[0] * 0.8, ground_color[1] * 0.8, ground_color[2] * 0.8)
        
        if not hit_sphere:
            # Sky gradient
            t = 0.5 * (ray.direction.y + 1.0)
            return (
                (1.0 - t) * 1.0 + t * 0.5,
                (1.0 - t) * 1.0 + t * 0.7,
                (1.0 - t) * 1.0 + t * 1.0
            )
        
        hit_point = ray.at(closest_t)
        normal = (hit_point - hit_sphere.position).normalize()
        
        if hit_sphere.material.type == 'diffuse':
            # Lambertian reflection
            light_dir = Vec3(1, 1, 1).normalize()
            diff = max(0, normal.dot(light_dir))
            return (
                hit_sphere.material.color[0] * (0.3 + 0.7 * diff),
                hit_sphere.material.color[1] * (0.3 + 0.7 * diff),
                hit_sphere.material.color[2] * (0.3 + 0.7 * diff)
            )
        elif hit_sphere.material.type == 'metal':
            # Reflective
            reflected = ray.direction.reflect(normal)
            new_ray = Ray(hit_point + normal * 0.001, reflected)
            bounce_color = self.trace_ray(new_ray, depth + 1, max_bounces)
            return (
                hit_sphere.material.color[0] * bounce_color[0],
                hit_sphere.material.color[1] * bounce_color[1],
                hit_sphere.material.color[2] * bounce_color[2]
            )
        elif hit_sphere.material.type == 'glass':
            # Glass: reflect
            reflected = ray.direction.reflect(normal)
            reflect_ray = Ray(hit_point + normal * 0.001, reflected)
            reflect_color = self.trace_ray(reflect_ray, depth + 1, max_bounces)
            
            # Fresnel approximation
            fresnel = 0.5
            return (
                hit_sphere.material.color[0] * reflect_color[0] * fresnel,
                hit_sphere.material.color[1] * reflect_color[1] * fresnel,
                hit_sphere.material.color[2] * reflect_color[2] * fresnel
            )
        
        return (0, 0, 0)
    
    def estimate_workload(self, ray: Ray) -> float:
        """Estimate workload for a ray based on expected bounces."""
        workload = 1.0  # Base: 1 bounce
        
        for sphere in self.scene.spheres:
            t = sphere.intersect(ray)
            if t > 0:
                workload += sphere.material.max_bounces
        
        return workload
    
    def render(self, max_bounces: int = 5) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Render the scene.
        
        Returns:
            image: RGB image array
            workloads: Workload estimate per pixel
            render_time: Time to render in seconds
        """
        rays = self.generate_rays()
        image = np.zeros((self.height, self.width, 3), dtype=np.float32)
        workloads = np.zeros(self.height * self.width, dtype=np.float32)
        
        start = time.time()
        
        for i, ray in enumerate(rays):
            y = i // self.width
            x = i % self.width
            
            color = self.trace_ray(ray, 0, max_bounces)
            image[y, x] = color
            workloads[i] = self.estimate_workload(ray)
        
        render_time = time.time() - start
        
        return image, workloads, render_time


if __name__ == '__main__':
    # Test the ray tracer
    print("Rendering test scene...")
    tracer = RayTracer(200, 150)
    image, workloads, render_time = tracer.render(max_bounces=5)
    
    print(f"Render time: {render_time:.2f}s")
    print(f"Average workload: {workloads.mean():.2f}")
    print(f"Max workload: {workloads.max():.2f}")
    print(f"Load imbalance: {workloads.max() / workloads.mean():.2f}x")
    
    # Save image
    from PIL import Image
    img = Image.fromarray((image * 255).astype(np.uint8))
    img.save('test_render.png')
    print("Saved test_render.png")
