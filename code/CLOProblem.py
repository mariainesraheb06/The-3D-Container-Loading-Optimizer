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


class Space:
    """Represents a free rectangular region inside the container."""

    def __init__(self, x, y, z, l, w, h):
        self.x = x   # X position (left)
        self.y = y   # Y position (bottom / floor)
        self.z = z   # Z position (front)
        self.l = l   # Length (X dimension)
        self.w = w   # Width  (Z dimension)
        self.h = h   # Height (Y dimension)

    def volume(self):
        return self.l * self.w * self.h

    def get_fit(self, box, strategy="first"):
        """
        Try all 6 orientations of `box` and return (fits, best_dims).
        `best_dims` is the (l, w, h) tuple that fits with least wasted space.
        Returns (False, None) if no orientation fits.
        """
        bl, bw, bh = box.l, box.w, box.h

        # Generate valid orientations (skip duplicates for cubes/square faces)
        orientations = set()
        for perm in [
            (bl, bw, bh), (bl, bh, bw),
            (bw, bl, bh), (bw, bh, bl),
            (bh, bl, bw), (bh, bw, bl),
        ]:
            # Fragile boxes: never place with original H as L or W
            if getattr(box, "fragile", False) and perm[1] == bh:
                continue
            orientations.add(perm)

        best_dims  = None
        best_score = None

        for dims in orientations:
            l, w, h = dims
            if l <= self.l and w <= self.w and h <= self.h:
                if strategy == "first":
                    return True, dims          # return immediately on first fit
                # For "best": minimise leftover volume in this space
                score = (self.l - l) * self.w * self.h \
                      + (self.w - w) * l * self.h \
                      + (self.h - h) * l * w
                if best_score is None or score < best_score:
                    best_score = score
                    best_dims  = dims

        if best_dims:
            return True, best_dims
        return False, None

    def split(self, l, w, h):
        """
        After placing a box of size (l, w, h) at the origin of this space,
        split the remaining free volume into up to 3 non-overlapping sub-spaces.

        Classic 3-way split (guillotine):

                +---+-------+
                | B |  R1   |   R1: right of box  (full height & depth)
                +---+       |
                |   |       |
                +---+-------+
                |   R2      |   R2: in front of box (full width, remaining depth)
                +---+-------+
                    R3          R3: above box (box footprint, remaining height)
        """
        new_spaces = []

        # Right of box  (x-axis remainder)
        if self.l - l > 0:
            new_spaces.append(Space(
                self.x + l, self.y, self.z,
                self.l - l, self.w, self.h
            ))

        # In front of box  (z-axis remainder)
        if self.w - w > 0:
            new_spaces.append(Space(
                self.x, self.y, self.z + w,
                l, self.w - w, self.h          # only as wide as the box (X)
            ))

        # Above box  (y-axis remainder)
        if self.h - h > 0:
            new_spaces.append(Space(
                self.x, self.y + h, self.z,
                l, w, self.h - h               # only over the box footprint
            ))

        return new_spaces

    def __repr__(self):
        return (f"Space(pos=({self.x},{self.y},{self.z}) "
                f"size={self.l}×{self.w}×{self.h} "
                f"vol={self.volume()})")


class SpaceManager:
    """
    Tracks the collection of free (empty) rectangular spaces inside the container
    and handles placement, splitting, and cleanup of those spaces.
    """

    def __init__(self, container_space: Space):
        self.container = container_space        # keep original for reset()
        self.spaces    = [container_space]


    def reset(self):
        """Reinitialise to a fully empty container (useful between GA runs)."""
        s = self.container
        self.spaces = [Space(s.x, s.y, s.z, s.l, s.w, s.h)]

    def find_placement(self, box, strategy="best"):
        """
        Search for a free space that can hold `box`.

        Parameters
        ----------
        box      : object with attributes l, w, h (and optionally `fragile`)
        strategy : "first"  – return the first space that fits (fast, greedy)
                   "best"   – return the space that wastes the least volume
                   "bottom" – among fitting spaces prefer lowest Y then X then Z
                              (gravity / stability heuristic)

        Returns
        -------
        (space, dims) or (None, None)
        """
        best_space = None
        best_dims  = None
        best_score = None

        # Sort spaces bottom-left-front for stability-aware strategies
        search_order = self.get_sorted_spaces()

        for space in search_order:
            fits, dims = space.get_fit(box, strategy="best" if strategy != "first" else "first")

            if not fits:
                continue

            if strategy == "first":
                return space, dims

            l, w, h = dims

            if strategy == "best":
                # Minimise leftover gap in the chosen space
                score = (space.l - l) + (space.w - w) + (space.h - h)

            elif strategy == "bottom":
                # Prefer lowest Y (floor proximity), then leftmost X, then front Z
                score = (space.y * 1_000_000) + (space.x * 1_000) + space.z

            else:
                raise ValueError(f"Unknown strategy '{strategy}'")

            if best_score is None or score < best_score:
                best_score = score
                best_space = space
                best_dims  = dims

        return best_space, best_dims

    def place_box(self, space: Space, dims: tuple):
        """
        Mark `space` as occupied by a box of size `dims`,
        split the remainder into new free spaces, and clean up.

        Parameters
        ----------
        space : the Space object returned by find_placement()
        dims  : (l, w, h) tuple – the orientation chosen for the box
        """
        if space not in self.spaces:
            return

        self.spaces.remove(space)

        new_spaces = space.split(*dims)
        self.spaces.extend(new_spaces)
        self.clean_spaces()


    def free_volume(self):
        """Sum of all remaining free space volumes (may overlap – use carefully)."""
        return sum(s.volume() for s in self.spaces)

    def container_volume(self):
        return self.container.volume()

    def utilization(self, packed_volume: float) -> float:
        """
        Return packing utilisation as a percentage.

        Parameters
        ----------
        packed_volume : total volume of all boxes successfully placed so far
        """
        cv = self.container_volume()
        if cv == 0:
            return 0.0
        return (packed_volume / cv) * 100.0

    def get_sorted_spaces(self):
        """
        Return spaces sorted bottom-left-front:
          primary   → lowest Y (floor first  → gravity)
          secondary → lowest X (left first   → tighter packing)
          tertiary  → lowest Z (front first  → easier loading)
        """
        return sorted(self.spaces, key=lambda s: (s.y, s.x, s.z))

    def clean_spaces(self):
        """
        Remove:
          1. Zero/negative-volume spaces
          2. Spaces fully contained inside a larger space
        """
        # Step 1 – remove degenerate spaces
        valid = [s for s in self.spaces if s.l > 0 and s.w > 0 and s.h > 0]

        # Step 2 – remove fully-contained spaces (keep larger ones)
        cleaned = []
        for s in valid:
            dominated = any(
                other is not s and self._contains(other, s)
                for other in valid
            )
            if not dominated:
                cleaned.append(s)

        self.spaces = cleaned

    def merge_spaces(self):
        """
        Optional post-clean step: merge pairs of spaces that share two
        dimensions and are adjacent along the third.  Reduces fragmentation.

        Only run this after a full pack cycle – it is O(n²) and expensive
        to call on every placement.
        """
        merged = True
        while merged:
            merged = False
            result = []
            used   = set()

            for i, a in enumerate(self.spaces):
                if i in used:
                    continue
                for j, b in enumerate(self.spaces):
                    if j <= i or j in used:
                        continue
                    m = self._try_merge(a, b)
                    if m is not None:
                        result.append(m)
                        used.add(i)
                        used.add(j)
                        merged = True
                        break
                if i not in used:
                    result.append(a)

            self.spaces = result
    @staticmethod
    def _contains(a: Space, b: Space) -> bool:
        """Return True if space `a` fully contains space `b`."""
        return (
            a.x <= b.x and a.y <= b.y and a.z <= b.z and
            a.x + a.l >= b.x + b.l and
            a.y + a.h >= b.y + b.h and
            a.z + a.w >= b.z + b.w
        )

    @staticmethod
    def _try_merge(a: Space, b: Space):
        """
        Merge two spaces if they are adjacent along exactly one axis and
        share the same dimensions on the other two axes.
        Returns a merged Space or None.
        """
        # Adjacent along X
        if (a.y == b.y and a.z == b.z and
                a.h == b.h and a.w == b.w):
            if a.x + a.l == b.x:
                return Space(a.x, a.y, a.z, a.l + b.l, a.w, a.h)
            if b.x + b.l == a.x:
                return Space(b.x, b.y, b.z, a.l + b.l, a.w, a.h)

        # Adjacent along Y (height)
        if (a.x == b.x and a.z == b.z and
                a.l == b.l and a.w == b.w):
            if a.y + a.h == b.y:
                return Space(a.x, a.y, a.z, a.l, a.w, a.h + b.h)
            if b.y + b.h == a.y:
                return Space(b.x, b.y, b.z, a.l, a.w, a.h + b.h)

        # Adjacent along Z (depth)
        if (a.x == b.x and a.y == b.y and
                a.l == b.l and a.h == b.h):
            if a.z + a.w == b.z:
                return Space(a.x, a.y, a.z, a.l, a.w + b.w, a.h)
            if b.z + b.w == a.z:
                return Space(b.x, b.y, b.z, a.l, a.w + b.w, a.h)

        return None

    def print_spaces(self):
        print(f"SpaceManager — {len(self.spaces)} free spaces "
              f"(container vol={self.container_volume()}):")
        for s in self.get_sorted_spaces():
            print(f"  {s}")