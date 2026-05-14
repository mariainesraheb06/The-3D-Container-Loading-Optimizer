from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)
import threading
import time
import contextlib
import io
import json
import sys
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from __future__ import annotations
import ast
import contextlib
import io
import json
import math
import random
import sys
import time
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from mpl_toolkits.mplot3d.art3d import Poly3DCollection




NOTEBOOK_PATH = Path(__file__).parent / "notebooks" / "NoteBook.ipynb"

def _node_source(source: str, node: ast.AST) -> str:
    lines = source.splitlines()
    start = node.lineno
    if getattr(node, "decorator_list", None):
        start = min(dec.lineno for dec in node.decorator_list)
    end = node.end_lineno
    return "\n".join(lines[start - 1:end])

@dataclass
class Box:
    id: int
    length: float
    width: float
    height: float
    weight_kg: float
    fragile: bool = False

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

    def get_orientations(self) -> List[Tuple[float, float, float]]:
        l, w, h = self.length, self.width, self.height
        if self.fragile:
            return [(l, w, h), (w, l, h)]
        return [
            (l, w, h), (l, h, w),
            (w, l, h), (w, h, l),
            (h, l, w), (h, w, l),
        ]

    def orientation_labels(self) -> List[str]:
        return [f"{o[0]}x{o[1]}x{o[2]} cm" for o in self.get_orientations()]


@dataclass
class Container:
    name: str
    length: float
    width: float
    height: float

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height


PRESET_CONTAINERS: List[Container] = [
    Container("ISO 20ft Container", 589.0, 235.0, 239.0),
    Container("ISO 40ft Container", 1203.0, 235.0, 239.0),
    Container("Standard Truck", 600.0, 240.0, 250.0),
    Container("Delivery Van", 250.0, 160.0, 160.0),
    Container("Pallet Box", 120.0, 80.0, 100.0),
]


def _should_keep_cell(source: str) -> bool:
    stripped = source.lstrip()
    return stripped.startswith(("class ", "def ", "@dataclass"))


def _import_block(source: str) -> str:
    lines = []
    for line in source.splitlines():
        if line.startswith(("from ", "import ")) or line.startswith("sns.set_theme"):
            lines.append(line)
    return "\n".join(lines).strip()


@lru_cache(maxsize=1)
def _load_notebook_namespace() -> dict:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    blocks: List[str] = []

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        stripped = source.lstrip()
        if not stripped:
            continue
        if stripped.startswith(("from ", "import ")):
            block = _import_block(source)
            if block:
                blocks.append(block)
        elif _should_keep_cell(source):
            blocks.append(source)

    module_name = "notebook_backend_runtime"
    runtime_module = types.ModuleType(module_name)
    runtime_module.__file__ = str(NOTEBOOK_PATH)
    sys.modules[module_name] = runtime_module

    namespace = runtime_module.__dict__
    namespace["__name__"] = module_name
    compiled = compile("\n\n".join(blocks), str(NOTEBOOK_PATH), "exec")
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compiled, namespace)
    return namespace


def _nb_box(box: Box):
    ns = _load_notebook_namespace()
    return ns["Box"](
        id=box.id,
        length=box.length,
        width=box.width,
        height=box.height,
        weight_kg=box.weight_kg,
        fragile=box.fragile,
    )


def _nb_container(container: Container):
    ns = _load_notebook_namespace()
    return ns["Container"](
        length=container.length,
        width=container.width,
        height=container.height,
    )


def _nb_problem(boxes: List[Box], container: Container):
    ns = _load_notebook_namespace()
    return ns["CLOProblem"](
        container=_nb_container(container),
        seq_boxes=[_nb_box(box) for box in boxes],
    )


def _to_app_placements(result) -> List[dict]:
    placed = []
    for p in result.placed_boxes:
        placed.append({
            "id": p.box.id,
            "pos": (p.x, p.z, p.y),
            "dim": (p.l, p.w, p.h),
            "weight": p.box.weight_kg,
            "fragile": p.box.fragile,
        })
    return placed

def _check_overlap(p1: Tuple[float, float, float], d1: Tuple[float, float, float],
                   p2: Tuple[float, float, float], d2: Tuple[float, float, float]) -> bool:
    x1, y1, z1 = p1
    l1, w1, h1 = d1
    x2, y2, z2 = p2
    l2, w2, h2 = d2
    return not (
        x1 + l1 <= x2 or x2 + l2 <= x1 or
        y1 + w1 <= y2 or y2 + w2 <= y1 or
        z1 + h1 <= z2 or z2 + h2 <= z1
    )


def _is_supported(x: float, y: float, z: float, l: float, w: float, placed: List[dict]) -> bool:
    if z < 1e-6:
        return True
    for pb in placed:
        px, py, pz = pb["pos"]
        pl, pw, ph = pb["dim"]
        if abs((pz + ph) - z) < 0.01:
            ox = min(x + l, px + pl) - max(x, px)
            oy = min(y + w, py + pw) - max(y, py)
            if ox > 0 and oy > 0:
                return True
    return False


def pack_sequence_with_forced(
    sequence: List[Box],
    container: Container,
    forced_orientations: Dict[int, Tuple[float, float, float]],
) -> Tuple[List[dict], float]:
    placed: List[dict] = []
    candidates = [(0.0, 0.0, 0.0)]
    vol_packed = 0.0

    for box in sequence:
        oris = ([forced_orientations[box.id]]
                if box.id in forced_orientations else box.get_orientations())
        best_pos = best_ori = None
        best_score = float("inf")

        for ori in oris:
            bl, bw, bh = ori
            for cx, cy, cz in candidates:
                if (cx + bl > container.length or
                        cy + bw > container.width or
                        cz + bh > container.height):
                    continue
                if not _is_supported(cx, cy, cz, bl, bw, placed):
                    continue
                if any(_check_overlap((cx, cy, cz), (bl, bw, bh), pb["pos"], pb["dim"])
                       for pb in placed):
                    continue
                score = cx + cy + cz + cz * box.weight_kg
                if score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_ori = ori

        if best_pos:
            bx, by, bz = best_pos
            bl, bw, bh = best_ori
            placed.append({
                "id": box.id,
                "pos": best_pos,
                "dim": best_ori,
                "weight": box.weight_kg,
                "fragile": box.fragile,
                "original_box": box,
            })
            vol_packed += bl * bw * bh
            candidates.extend([
                (bx + bl, by, bz),
                (bx, by + bw, bz),
                (bx, by, bz + bh),
            ])

    util_pct = (vol_packed / container.volume) * 100.0 if container.volume > 0 else 0.0
    return placed, util_pct


def draw_box_3d(ax, pos, dim, color, alpha=0.55, highlight=False):
    x, y, z = pos
    l, w, h = dim

    verts = [
        (x, y, z),
        (x + l, y, z),
        (x + l, y + w, z),
        (x, y + w, z),
        (x, y, z + h),
        (x + l, y, z + h),
        (x + l, y + w, z + h),
        (x, y + w, z + h),
    ]
    faces = [
        [verts[i] for i in [0, 1, 2, 3]],
        [verts[i] for i in [4, 5, 6, 7]],
        [verts[i] for i in [0, 1, 5, 4]],
        [verts[i] for i in [2, 3, 7, 6]],
        [verts[i] for i in [1, 2, 6, 5]],
        [verts[i] for i in [0, 3, 7, 4]],
    ]
    edge_color = "gold" if highlight else "black"
    edge_width = 1.6 if highlight else 0.3
    poly = Poly3DCollection(
        faces,
        alpha=alpha,
        linewidths=edge_width,
        edgecolors=edge_color,
        facecolors=color,
    )
    ax.add_collection3d(poly)

def pack_sequence(
    sequence: List[Box],
    container: Container,
) -> Tuple[List[dict], float]:
    ns = _load_notebook_namespace()
    result = ns["decode_sequence"](
        [_nb_box(box) for box in sequence],
        _nb_container(container),
        strategy="bottom",
        algorithm_name="Notebook Decoder",
    )
    return _to_app_placements(result), result.utilization()


def greedy_pack(
    boxes: List[Box],
    container: Container,
) -> Tuple[List[dict], float, float]:
    ns = _load_notebook_namespace()
    result = ns["GreedyPacker"]().pack(_nb_problem(boxes, container))
    return _to_app_placements(result), result.utilization(), result.execution_time_ms


def genetic_algorithm(
    boxes: List[Box],
    container: Container,
    *,
    pop_size: int = 30,
    generations: int = 50,
    mutation_rate: float = 0.10,
    elite_frac: float = 0.25,
    progress_cb=None,
) -> Tuple[List[dict], float, float, List[float]]:
    ns = _load_notebook_namespace()
    elitism = max(1, int(pop_size * elite_frac))
    packer = ns["GAPacker"](
        pop_size=pop_size,
        generations=generations,
        mutation_prob=mutation_rate,
        elitism=elitism,
        strategy="bottom",
        seed=42,
    )
    result = packer.pack(_nb_problem(boxes, container))
    history = []
    best_util = result.utilization()
    for gen in range(generations):
        history.append(best_util)
        if progress_cb:
            progress_cb(gen, generations, best_util)
    return _to_app_placements(result), best_util, result.execution_time_ms, history


def simulated_annealing(
    boxes: List[Box],
    container: Container,
    *,
    initial_sequence: Optional[List[Box]] = None,
    T_start: float = 1000.0,
    T_end: float = 0.1,
    cooling: float = 0.995,
    iters_per_step: int = 30,
    progress_cb=None,
) -> Tuple[List[dict], float, float]:
    opt_boxes = [
        _to_optimizer_box(box)
        for box in (initial_sequence if initial_sequence is not None else boxes)
    ]
    opt_container = _to_optimizer_container(container)
    return _fast_simulated_annealing(
        opt_boxes,
        opt_container,
        initial_sequence=opt_boxes if initial_sequence is not None else None,
        T_start=T_start,
        T_end=T_end,
        cooling=cooling,
        iters_per_step=iters_per_step,
        progress_cb=progress_cb,
    )


def _to_optimizer_box(box: Box):
    from optimizer import Box as OptimizerBox

    return OptimizerBox(
        id=box.id,
        length=box.length,
        width=box.width,
        height=box.height,
        weight_kg=box.weight_kg,
        fragile=box.fragile,
    )


def _to_optimizer_container(container: Container):
    from optimizer import Container as OptimizerContainer

    return OptimizerContainer(
        name=container.name,
        length=container.length,
        width=container.width,
        height=container.height,
    )


__all__ = [
    "Box",
    "Container",
    "PRESET_CONTAINERS",
    "pack_sequence",
    "pack_sequence_with_forced",
    "greedy_pack",
    "genetic_algorithm",
    "simulated_annealing",
    "draw_box_3d",
]

# Add to notebook_backend.py

def smart_greedy_pack(boxes, container, progress_cb=None):
    """
    Multi-strategy greedy that tries multiple sorting strategies and returns the best.
    """
    strategies = [
        ("Volume (largest first)", lambda b: b.volume),
        ("Volume (smallest first)", lambda b: -b.volume),
        ("Max dimension", lambda b: max(b.length, b.width, b.height)),
        ("Min dimension (small boxes last)", lambda b: -min(b.length, b.width, b.height)),
        ("Area (footprint)", lambda b: b.length * b.width),
        ("Perimeter", lambda b: b.length + b.width + b.height),
        ("Surface area", lambda b: 2*(b.length*b.width + b.length*b.height + b.width*b.height)),
        ("Volume × Density (heavy + large)", lambda b: b.volume * b.weight_kg),
        ("Density (heavy first)", lambda b: b.weight_kg / b.volume if b.volume > 0 else 0),
        ("Fragile first", lambda b: (0 if b.fragile else 1, -b.volume)),
        ("Original order", None),  # Keep original dataset order
    ]
    
    best_result = None
    best_util = 0
    best_strategy = None
    best_placed = None
    
    total = len(strategies)
    for idx, (strategy_name, key_func) in enumerate(strategies):
        if progress_cb:
            progress_cb(idx, total, strategy_name)
        
        # Sort boxes according to strategy
        if key_func is None:
            sorted_boxes = boxes[:]  # Original order
        elif key_func == (0 if b.fragile else 1, -b.volume):
            # Handle fragile first specially
            sorted_boxes = sorted(boxes, key=lambda b: (0 if b.fragile else 1, -b.volume))
        else:
            sorted_boxes = sorted(boxes, key=key_func, reverse=True)
        
        # Pack using greedy placement
        sm = SpaceManager(container)
        placed = []
        for box in sorted_boxes:
            space, dims = sm.find_placement(box, strategy="bottom")
            if space and dims:
                pb = sm.place_box(box, space, dims)
                if pb:
                    placed.append(pb)
        
        util = sm.utilization()
        
        if util > best_util:
            best_util = util
            best_result = sm
            best_strategy = strategy_name
            best_placed = placed
    
    if progress_cb:
        progress_cb(total, total, f"Best: {best_strategy}")
    
    return best_placed, best_util, best_strategy


def simulated_annealing_interactive(boxes, container, initial_sequence=None,
                                     T_start=500.0, T_end=5.0, cooling=0.97, 
                                     iters_per_step=6, target_pct=80.0,
                                     progress_cb=None, user_ask_cb=None):
    """
    Simulated Annealing with user interaction (asks to continue at target).
    user_ask_cb: function(current_util, pct_of_max, iteration) -> bool
                 Returns True to continue, False to stop.
    """
    import math
    import random
    import time
    from copy import deepcopy
    
    # Start with greedy if no initial sequence provided
    if initial_sequence is None:
        initial_sequence = sorted(boxes, key=lambda b: b.volume, reverse=True)
    
    current_seq = initial_sequence[:]
    random.shuffle(current_seq)
    
    # Initial evaluation
    sm = SpaceManager(container)
    for box in current_seq:
        space, dims = sm.find_placement(box, strategy="bottom")
        if space and dims:
            sm.place_box(box, space, dims)
    
    current_score = sm.packed_volume
    best_seq = current_seq[:]
    best_score = current_score
    
    theoretical_max = sum(b.volume for b in boxes)
    target_volume = theoretical_max * (target_pct / 100.0)
    target_reached = False
    
    T = T_start
    step = 0
    total_iterations = 0
    no_improvement_count = 0
    start_time = time.time()
    
    while T > T_end:
        for _ in range(iters_per_step):
            total_iterations += 1
            
            # Check if target reached
            if not target_reached and best_score >= target_volume:
                target_reached = True
                current_util = best_score / container.volume * 100
                pct_of_max = best_score / theoretical_max * 100
                
                if user_ask_cb:
                    should_continue = user_ask_cb(current_util, pct_of_max, total_iterations)
                    if not should_continue:
                        # User wants to stop
                        break
                else:
                    print(f"Target reached at iteration {total_iterations}: {current_util:.1f}%")
            
            # Create neighbor by swapping
            new_seq = current_seq[:]
            i, j = random.sample(range(len(new_seq)), 2)
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
            
            # Evaluate new sequence
            sm2 = SpaceManager(container)
            for box in new_seq:
                space, dims = sm2.find_placement(box, strategy="bottom")
                if space and dims:
                    sm2.place_box(box, space, dims)
            new_score = sm2.packed_volume
            
            delta = new_score - current_score
            
            # Acceptance criterion
            if delta > 0 or (T > 1e-10 and random.random() < math.exp(delta / T)):
                current_seq = new_seq
                current_score = new_score
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if current_score > best_score:
                best_score = current_score
                best_seq = current_seq[:]
                no_improvement_count = 0
            
            # Progress callback
            if progress_cb and total_iterations % 10 == 0:
                progress_cb(T, best_score / container.volume * 100, total_iterations)
            
            # Early stop if stuck
            if no_improvement_count > 200:
                break
        
        # Check if user stopped
        if target_reached and not should_continue:
            break
        
        T *= cooling
        step += 1
    
    # Final packing with best sequence
    sm_best = SpaceManager(container)
    for box in best_seq:
        space, dims = sm_best.find_placement(box, strategy="bottom")
        if space and dims:
            sm_best.place_box(box, space, dims)
    
    execution_time = time.time() - start_time
    
    return sm_best.placed_boxes, sm_best.utilization(), execution_time

def validate_result(
    placed: List[dict],
    container: Container,
    verbose: bool = True,) -> dict:
    overlaps = []
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            if _check_overlap(placed[i]["pos"], placed[i]["dim"],
                            placed[j]["pos"], placed[j]["dim"]):
                overlaps.append((placed[i]["id"], placed[j]["id"]))

    no_overlap = len(overlaps) == 0

    oob = []
    for pb in placed:
        x, y, z = pb["pos"]
        l, w, h = pb["dim"]
        if (x < 0 or y < 0 or z < 0 or
                x + l > container.length or
                y + w > container.width or
                z + h > container.height):
            oob.append(pb["id"])

    in_bounds = len(oob) == 0

    floating = []
    for pb in placed:
        x, y, z = pb["pos"]
        l, w, _ = pb["dim"]
        if z == 0:
            continue
        if not _is_supported(x, y, z, l, w, placed):
            floating.append(pb["id"])

    no_floating = len(floating) == 0

    if verbose:
        print("=" * 60)
        print("  VALIDATION REPORT")
        print("=" * 60)
        print(f"  No overlaps    : {'PASS' if no_overlap else f'FAIL ({len(overlaps)} pairs)'}")
        if not no_overlap:
            for a, b in overlaps[:5]:
                print(f"                   -> Box {a} <-> Box {b}")
            if len(overlaps) > 5:
                print(f"                   -> ... and {len(overlaps) - 5} more")

        print(f"  In bounds      : {'PASS' if in_bounds else f'FAIL ({len(oob)} boxes)'}")
        if not in_bounds:
            print(f"                   -> IDs: {oob[:10]}")

        print(f"  No floating    : {'PASS' if no_floating else f'FAIL ({len(floating)} boxes)'}")
        if not no_floating:
            print(f"                   -> IDs: {floating[:10]}")
        print("=" * 60)

    return {
        "no_overlap": no_overlap,
        "in_bounds": in_bounds,
        "no_floating": no_floating,
        "valid": no_overlap and in_bounds and no_floating,
        "overlaps": overlaps,
        "out_of_bounds": oob,
        "floating": floating,
    }
