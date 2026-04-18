from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class Box:
    """Represents a single box to be packed."""
    id: int
    length: float
    width: float
    height: float
    weight_kg: float
    fragile: bool = False
    # Position inside container (set during packing)
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    placed: bool = False

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

    def get_dimensions(self) -> Tuple[float, float, float]:
        return (self.length, self.width, self.height)

    def get_orientations(self) -> List[Tuple[float, float, float]]:
        """
        Returns all valid orientations (L, W, H permutations).
        Fragile boxes cannot be flipped upside down:
        height must remain as the vertical axis.
        """
        l, w, h = self.length, self.width, self.height
        if self.fragile:
            # Only rotate in the horizontal plane — height stays fixed
            return [
                (l, w, h),
                (w, l, h),
            ]
        else:
            # All 6 orientations allowed
            return [
                (l, w, h), (l, h, w),
                (w, l, h), (w, h, l),
                (h, l, w), (h, w, l),
            ]

    def __repr__(self):
        return (f'Box(id={self.id}, {self.length}x{self.width}x{self.height}cm, '
                f'{self.weight_kg}kg, fragile={self.fragile})')

@dataclass
class Container:
    """
    Standard ISO 20ft shipping container.
    Internal dimensions (cm): 589 x 235 x 239
    """
    length: float = 589.0   # cm
    width: float  = 235.0   # cm
    height: float = 239.0   # cm

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

    def can_fit(self, box: Box, x: float, y: float, z: float,
                orientation: Tuple[float,float,float]) -> bool:
        """Check if a box fits at position (x,y,z) with given orientation."""
        bl, bw, bh = orientation
        return (
            x + bl <= self.length and
            y + bw <= self.width  and
            z + bh <= self.height
        )

    def get_dimensions(self) -> Tuple[float, float, float]:
        return (self.length, self.width, self.height)

    def __repr__(self):
        return (f'Container({self.length}x{self.width}x{self.height}cm, '
                f'volume={self.volume/1e6:.2f}m³)')

@dataclass                          #python will directly generate its constructor __init__
class CLOProblem :
    container: Container
    seq_boxes: List[Box]
    #def __init__(self, container, seq_boxes):
     #   self.container = container
      #  self.seq_boxes = seq_boxes

    def __post_init__(self):
        """
        To check wether :
                        every box do fit in the container
                        all boxes might fit
        """
        total_box_volume=0

        for box in  self.seq_boxes:
            total_box_volume += box.volume

            if not self.fits_in_container(box, self.container):
                raise ValueError(f"Box {box.get_dimensions()} can not fit in"
                                f"Container {self.container.get_dimensions()} even with rotations!")
        
        container_volume = self.container.volume
        if total_box_volume > container_volume :
            print(  f"Warning: Total box volume ({total_box_volume}) exceeds "
                    f"container volume ({container_volume})")
        self._total_box_volume = total_box_volume
        self._container_volume = container_volume

    @property
    def total_box_volume(self):     
        return self._total_box_volume
    
    @property
    def container_volume(self):    
        return self._container_volume

    @staticmethod
    def fits_in_container(box, container):
        box_dims = sorted(box.get_dimensions())
        container_dims = sorted(container.get_dimensions())
        return all(box_dims[i]<=container_dims[i] for i in range(3))

    def get_difficulty(self) -> str:           #additional !
        """To estimate the problem difficulty"""
        ratio = self.total_box_volume / self.container_volume
        if ratio < 0.5:
            return "Easy"
        elif ratio < 0.8:
            return "Medium"
        else:
            return "Hard"

    def __repr__ (self) -> str :
        return f"""
                    Problem : {self.get_difficulty()}
                    Container: {self.container.get_dims()}
                    Boxes : {len(self.seq_boxes)}
                    Total box volume: {self._total_box_volume:.2f}
                    Container volume: {self._container_volume:.2f}
                """

class SpaceManager:
    def init(self, container_space):
        # Start with one big empty container
        self.spaces = [container_space]

    def find_best_placement(self, box):
        best_score = None
        best_space = None
        best_dims = None

        for space in self.spaces:
            fits, dims = space.get_best_fit(box)

            if fits:
                l, w, h = dims

                # Heuristic: minimize leftover edges
                score = (space.l - l) + (space.w - w) + (space.h - h)

                if best_score is None or score < best_score:
                    best_score = score
                    best_space = space
                    best_dims = dims

        return best_space, best_dims
    def place_box(self, space, dims):
        if space not in self.spaces:
            return

        # Remove used space
        self.spaces.remove(space)

        # Split into new spaces
        new_spaces = space.split(*dims)

        # Add and clean
        self.spaces.extend(new_spaces)
        self.clean_spaces()

    def clean_spaces(self):
        cleaned = []

        for s in self.spaces:
            # Remove zero or negative spaces
            if s.l <= 0 or s.w <= 0 or s.h <= 0:
                continue

            # Remove contained spaces
            if any(other != s and self.contains(other, s) for other in self.spaces):
                continue

            cleaned.append(s)

        self.spaces = cleaned

    def contains(self, a, b):
        return (
            a.x <= b.x and
            a.y <= b.y and
            a.z <= b.z and
            a.x + a.l >= b.x + b.l and
            a.y + a.w >= b.y + b.w and
            a.z + a.h >= b.z + b.h
        )

    def print_spaces(self):
        for s in self.spaces:
            print(s)