from __future__ import annotations

import contextlib
import io
import json
import sys
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from optimizer import (
    draw_box_3d,
    pack_sequence_with_forced,
    simulated_annealing as _fast_simulated_annealing,
)


NOTEBOOK_PATH = Path(__file__).parent / "notebooks" / "NoteBook.ipynb"


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
