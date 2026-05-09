import time
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AlgorithmResult:
    """
    Unified result format for all algorithms.
    Normalizes the different output formats from Greedy (PackingResult)
    and SA (dict) into one consistent structure.
    """
    algorithm: str
    utilization_pct: float        # Volume utilization in %
    boxes_placed: int             # Number of boxes successfully placed
    total_boxes: int              # Total boxes attempted
    execution_time_ms: float      # Runtime in milliseconds
    placed_boxes: list            # Raw placement data (for visualization)

    def placement_rate(self) -> float:
        """Percentage of boxes that were placed."""
        return (self.boxes_placed / self.total_boxes) * 100 if self.total_boxes > 0 else 0.0

    def __repr__(self):
        return (
            f"AlgorithmResult(algorithm='{self.algorithm}', "
            f"utilization={self.utilization_pct:.2f}%, "
            f"boxes={self.boxes_placed}/{self.total_boxes}, "
            f"time={self.execution_time_ms:.2f}ms)"
        )


class Comparison:
    """
    Runs and compares all packing algorithms:
      - Greedy Best-Fit Decreasing
      - Simulated Annealing
      - Genetic Algorithm (optional, if implemented)

    Usage:
        comp = Comparison(problem, all_boxes, container)
        comp.run_all()
        comp.print_summary()
        comp.plot_comparison()
    """

    def __init__(self, problem, all_boxes: list, container):
        """
        Parameters
        ----------
        problem    : CLOProblem instance
        all_boxes  : list of Box objects
        container  : Container instance
        """
        self.problem   = problem
        self.all_boxes = all_boxes
        self.container = container
        self.results: List[AlgorithmResult] = []

    # ─────────────────────────────────────────────
    # Run individual algorithms
    # ─────────────────────────────────────────────

    def run_greedy(self) -> AlgorithmResult:
        """Run the Greedy Best-Fit Decreasing packer."""
        print("Running Greedy Best-Fit Decreasing...")

        packer = GreedyPacker()
        greedy_result = packer.pack(self.problem)   # returns PackingResult

        result = AlgorithmResult(
            algorithm         = greedy_result.algorithm,
            utilization_pct   = greedy_result.utilization(),
            boxes_placed      = len(greedy_result.placed_boxes),
            total_boxes       = len(self.all_boxes),
            execution_time_ms = greedy_result.execution_time_ms,
            placed_boxes      = greedy_result.placed_boxes
        )

        print(f"  ✅ Greedy done: {result.utilization_pct:.2f}% in {result.execution_time_ms:.2f}ms")
        return result

    def run_sa(
        self,
        T_start: float = 1000.0,
        T_end: float = 0.1,
        cooling_rate: float = 0.995,
        iterations_per_temp: int = 50
    ) -> AlgorithmResult:
        """Run Simulated Annealing."""
        print("Running Simulated Annealing...")

        sa_dict = simulated_annealing(
            boxes               = self.all_boxes,
            container           = self.container,
            T_start             = T_start,
            T_end               = T_end,
            cooling_rate        = cooling_rate,
            iterations_per_temp = iterations_per_temp
        )   # returns a dict

        result = AlgorithmResult(
            algorithm         = "Simulated Annealing",
            utilization_pct   = sa_dict["best_utilization"],
            boxes_placed      = len(sa_dict["best_placed"]),
            total_boxes       = len(self.all_boxes),
            execution_time_ms = sa_dict["runtime"] * 1000,  # convert s → ms
            placed_boxes      = sa_dict["best_placed"]
        )

        print(f"  ✅ SA done: {result.utilization_pct:.2f}% in {result.execution_time_ms/1000:.1f}s")
        return result

    def run_ga(self) -> Optional[AlgorithmResult]:
        """
        Run Genetic Algorithm (if implemented).
        Returns None and prints a warning if GAPacker is not yet implemented.
        """
        print("Running Genetic Algorithm...")

        packer = GAPacker()

        # Check if GA is still a stub (empty pass class)
        if not hasattr(packer, "pack"):
            print("  ⚠️  GAPacker not implemented yet — skipping.")
            return None

        try:
            ga_result = packer.pack(self.problem)   # expected to return PackingResult

            result = AlgorithmResult(
                algorithm         = "Genetic Algorithm",
                utilization_pct   = ga_result.utilization(),
                boxes_placed      = len(ga_result.placed_boxes),
                total_boxes       = len(self.all_boxes),
                execution_time_ms = ga_result.execution_time_ms,
                placed_boxes      = ga_result.placed_boxes
            )

            print(f"  ✅ GA done: {result.utilization_pct:.2f}% in {result.execution_time_ms:.2f}ms")
            return result

        except (NotImplementedError, AttributeError, TypeError) as e:
            print(f"  ⚠️  GAPacker raised an error ({e}) — skipping.")
            return None

    # ─────────────────────────────────────────────
    # Run all algorithms
    # ─────────────────────────────────────────────

    def run_all(self, run_ga: bool = True, sa_params: dict = None):
        """
        Run all algorithms and store their results internally.

        Parameters
        ----------
        run_ga    : whether to attempt running GA (set False if not implemented yet)
        sa_params : optional dict of SA hyperparameters to override defaults
                    e.g. {'T_start': 500, 'cooling_rate': 0.99}
        """
        self.results = []

        greedy_result = self.run_greedy()
        self.results.append(greedy_result)

        sa_kwargs = sa_params or {}
        sa_result = self.run_sa(**sa_kwargs)
        self.results.append(sa_result)

        if run_ga:
            ga_result = self.run_ga()
            if ga_result is not None:
                self.results.append(ga_result)

        print(f"\n✅ Comparison complete — {len(self.results)} algorithm(s) ran.")

    # ─────────────────────────────────────────────
    # Reporting
    # ─────────────────────────────────────────────

    def print_summary(self):
        """Print a formatted comparison table to the console."""
        if not self.results:
            print("No results yet — call run_all() first.")
            return

        print("\n" + "=" * 70)
        print("           ALGORITHM COMPARISON SUMMARY")
        print("=" * 70)
        print(f"{'Algorithm':<35} {'Utilization':>12} {'Boxes':>10} {'Time':>12}")
        print("-" * 70)

        for r in self.results:
            time_str = (
                f"{r.execution_time_ms/1000:.2f}s"
                if r.execution_time_ms >= 1000
                else f"{r.execution_time_ms:.1f}ms"
            )
            print(
                f"{r.algorithm:<35} "
                f"{r.utilization_pct:>11.2f}% "
                f"{r.boxes_placed:>4}/{r.total_boxes:<5} "
                f"{time_str:>12}"
            )

        print("=" * 70)

        # Highlight winner and fastest
        best    = max(self.results, key=lambda r: r.utilization_pct)
        fastest = min(self.results, key=lambda r: r.execution_time_ms)

        print(f"\n🏆 Best utilization : {best.algorithm} ({best.utilization_pct:.2f}%)")
        print(f"⚡ Fastest          : {fastest.algorithm} ({fastest.execution_time_ms:.1f}ms)")

        # SA vs Greedy delta
        greedy = next((r for r in self.results if "Greedy" in r.algorithm), None)
        sa     = next((r for r in self.results if "Simulated" in r.algorithm), None)
        if greedy and sa:
            delta = sa.utilization_pct - greedy.utilization_pct
            sign  = "+" if delta >= 0 else ""
            print(f"📈 SA vs Greedy     : {sign}{delta:.2f}% utilization")

        print()

    def to_dataframe(self) -> pd.DataFrame:
        """Return results as a pandas DataFrame for further analysis."""
        if not self.results:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "Algorithm"         : r.algorithm,
                "Utilization (%)"   : round(r.utilization_pct, 2),
                "Boxes Placed"      : r.boxes_placed,
                "Total Boxes"       : r.total_boxes,
                "Placement Rate (%)": round(r.placement_rate(), 2),
                "Time (ms)"         : round(r.execution_time_ms, 2),
                "Time (s)"          : round(r.execution_time_ms / 1000, 3),
            }
            for r in self.results
        ])

    # ─────────────────────────────────────────────
    # Visualization
    # ─────────────────────────────────────────────

    def plot_comparison(self, save_path: str = "comparison_results.png"):
        """
        Generate a 3-panel comparison chart:
          1. Volume utilization % per algorithm
          2. Execution time per algorithm
          3. Boxes placed vs total
        """
        if not self.results:
            print("No results to plot — call run_all() first.")
            return

        labels = [r.algorithm.replace(" ", "\n") for r in self.results]
        utils  = [r.utilization_pct for r in self.results]
        times  = [r.execution_time_ms / 1000 for r in self.results]   # seconds
        placed = [r.boxes_placed for r in self.results]
        total  = self.results[0].total_boxes

        colors = ["#4fc3f7", "#ff8a65", "#81c784", "#ce93d8"]

        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        fig.suptitle(
            "Algorithm Comparison — 3D Container Loading Optimizer",
            fontsize=14, fontweight="bold"
        )

        # ── Plot 1: Volume utilization ──
        bars = axes[0].bar(
            labels, utils,
            color=colors[:len(labels)], edgecolor="white", width=0.5
        )
        axes[0].axhline(y=75, color="green",  linestyle="--", linewidth=1.2, label="75% target")
        axes[0].axhline(y=80, color="orange", linestyle="--", linewidth=1.2, label="80% target")
        for bar, val in zip(bars, utils):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold"
            )
        axes[0].set_title("Volume Utilization (%)")
        axes[0].set_ylabel("Utilization (%)")
        axes[0].set_ylim(0, 105)
        axes[0].legend(fontsize=9)

        # ── Plot 2: Execution time ──
        bars2 = axes[1].bar(
            labels, times,
            color=colors[:len(labels)], edgecolor="white", width=0.5
        )
        for bar, val in zip(bars2, times):
            label = f"{val:.2f}s" if val >= 1 else f"{val*1000:.0f}ms"
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(times) * 0.01,
                label,
                ha="center", va="bottom", fontsize=11, fontweight="bold"
            )
        axes[1].set_title("Execution Time")
        axes[1].set_ylabel("Time (seconds)")

        # ── Plot 3: Boxes placed vs total ──
        x     = range(len(labels))
        width = 0.35
        axes[2].bar(
            [i - width / 2 for i in x], [total] * len(labels),
            width=width, label="Total boxes", color="#cccccc", edgecolor="white"
        )
        axes[2].bar(
            [i + width / 2 for i in x], placed,
            width=width, label="Boxes placed",
            color=colors[:len(labels)], edgecolor="white"
        )
        axes[2].set_xticks(list(x))
        axes[2].set_xticklabels(labels)
        axes[2].set_title("Boxes Placed vs Total")
        axes[2].set_ylabel("Count")
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Comparison chart saved to '{save_path}' ✅")