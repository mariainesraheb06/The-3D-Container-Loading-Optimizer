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


@lru_cache(maxsize=1)
def _load_notebook_namespace() -> dict:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    blocks: List[str] = []

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
                block = _node_source(source, node)
                if block:
                    blocks.append(block)

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
    packer.tournament_k = min(packer.tournament_k, packer.pop_size)
    problem = _nb_problem(boxes, container)

    t0 = time.time()
    population = [
        random.sample(problem.seq_boxes, len(problem.seq_boxes))
        for _ in range(packer.pop_size)
    ]

    best_chromosome = population[0][:]
    best_fitness = 0.0
    history: List[float] = []

    for gen in range(packer.generations):
        fitnesses = [packer._fitness(chrom, problem.container) for chrom in population]

        gen_best_idx = max(range(packer.pop_size), key=lambda i: fitnesses[i])
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_chromosome = population[gen_best_idx][:]

        best_util = best_fitness / problem.container.volume * 100
        history.append(best_util)
        if progress_cb:
            progress_cb(gen + 1, packer.generations, best_util)

        sorted_idx = sorted(
            range(packer.pop_size),
            key=lambda i: fitnesses[i],
            reverse=True,
        )
        next_gen = [population[i][:] for i in sorted_idx[:packer.elitism]]

        while len(next_gen) < packer.pop_size:
            p1 = packer._tournament_select(population, fitnesses)
            p2 = packer._tournament_select(population, fitnesses)

            if random.random() < packer.crossover_prob:
                child = packer._ox1(p1, p2)
            else:
                child = p1[:]

            if random.random() < packer.mutation_prob:
                child = packer._mutate(child)

            next_gen.append(child)

        population = next_gen

    result = ns["decode_sequence"](
        best_chromosome,
        problem.container,
        strategy=packer.strategy,
        algorithm_name="Genetic Algorithm",
    )
    result.execution_time_ms = (time.time() - t0) * 1000
    return _to_app_placements(result), result.utilization(), result.execution_time_ms, history


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
    ns = _load_notebook_namespace()
    problem = _nb_problem(boxes, container)
    safe_T_start = max(T_start, 1e-9)
    safe_T_end = max(T_end, 1e-9)
    safe_iters = max(1, iters_per_step)
    safe_cooling = cooling if 0 < cooling < 1 else 0.995
    steps = max(
        1,
        int(math.ceil(math.log(safe_T_end / safe_T_start) / math.log(safe_cooling))),
    )
    iterations = max(1, steps * safe_iters)
    alpha = safe_cooling ** (1 / safe_iters)

    packer = ns["SAPacker"](
        T_init=safe_T_start,
        alpha=alpha,
        iterations=iterations,
        strategy="bottom",
        interactive=False,
        seed=42,
    )

    t0 = time.time()
    theoretical_max_volume = problem.total_box_volume
    container_vol = problem.container.volume
    target_volume = theoretical_max_volume * (packer.target_pct_of_max / 100.0)

    if initial_sequence is not None:
        current_seq = [_nb_box(box) for box in initial_sequence]
    else:
        current_seq = sorted(problem.seq_boxes, key=lambda b: b.volume, reverse=True)
    current_score = packer._evaluate(current_seq, problem.container)

    best_seq = current_seq[:]
    best_score = current_score
    no_improvement_count = 0
    T = packer.T_init
    target_reached = False

    for iteration in range(packer.iterations):
        current_container_util = best_score / container_vol * 100
        current_pct_of_max = best_score / theoretical_max_volume * 100

        if not target_reached and best_score >= target_volume:
            target_reached = True
            break

        if no_improvement_count > 100:
            break

        new_seq = packer._swap(current_seq)
        new_score = packer._evaluate(new_seq, problem.container)
        delta = new_score - current_score

        if delta > 0 or (T > 1e-10 and random.random() < math.exp(delta / T)):
            current_seq = new_seq
            current_score = new_score
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if current_score > best_score:
            best_score = current_score
            best_seq = current_seq[:]

        T *= packer.alpha
        if progress_cb and iteration % safe_iters == 0:
            progress_cb(T, safe_T_start, best_score / container_vol * 100)

    result = ns["decode_sequence"](
        best_seq,
        problem.container,
        strategy=packer.strategy,
        algorithm_name="Simulated Annealing",
    )
    result.execution_time_ms = (time.time() - t0) * 1000
    return _to_app_placements(result), result.utilization(), result.execution_time_ms

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
