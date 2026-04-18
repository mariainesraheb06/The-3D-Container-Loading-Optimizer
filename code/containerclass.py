from dataclasses import dataclass, field
from typing import List, Tuple
from __future__ import annotations


@dataclass
class Container:
    def __init__(self, length, width, height):
        self.length = length
        self.width = width
        self.height = height
        
        self.boxes = []  
        # Each element: (box, x, y, z, l, w, h)

        self.used_volume = 0

    # -------------------------------
    # Check if a box can be placed
    # -------------------------------
    def can_place(self, box, x, y, z, l, w, h):
        # Check boundaries
        if (x + l > self.length or
            y + w > self.width or
            z + h > self.height):
            return False

        # Check overlap with existing boxes
        for (b, bx, by, bz, bl, bw, bh) in self.boxes:
            if not (x + l <= bx or bx + bl <= x or
                    y + w <= by or by + bw <= y or
                    z + h <= bz or bz + bh <= z):
                return False

        return True

    # -------------------------------
    # Add box if possible
    # -------------------------------
    def add_box(self, box, position, orientation):
        x, y, z = position
        l, w, h = orientation

        if self.can_place(box, x, y, z, l, w, h):
            self.boxes.append((box, x, y, z, l, w, h))
            self.used_volume += l * w * h
            return True

        return False

    # Remove a box
    def remove_box(self, index):
        b, x, y, z, l, w, h = self.boxes.pop(index)
        self.used_volume -= l * w * h

    # Fitness (for GA / heuristics)
    # -------------------------------
    def utilization(self):
        total_volume = self.length * self.width * self.height
        return self.used_volume / total_volume


    # Copy (IMPORTANT for algorithms)
    # -------------------------------
    def copy(self):
        new_container = Container(self.length, self.width, self.height)
        new_container.boxes = self.boxes.copy()
        new_container.used_volume = self.used_volume
        return new_container


    # Get all possible orientations
    # -------------------------------
    @staticmethod
    def get_orientations(box):
        l, w, h = box
        return list(set([
            (l, w, h), (l, h, w),
            (w, l, h), (w, h, l),
            (h, l, w), (h, w, l)
        ]))
    # -------------------------
    # DEBUG
    # -------------------------
    def summary(self):
        print("------ Container Summary ------")
        print(f"Dimensions: {self.length} x {self.width} x {self.height} cm")
        print(f"Boxes placed: {len(self.boxes)}")
        print(f"Used volume: {self.used_volume():.2f} cm³")
        print(f"Utilization: {self.utilization():.2f}%")
        print("--------------------------------")

    def __repr__(self):
        return (
            f"Container({self.length}x{self.width}x{self.height} cm, "
            f"utilization={self.utilization():.2f}%)"
        )