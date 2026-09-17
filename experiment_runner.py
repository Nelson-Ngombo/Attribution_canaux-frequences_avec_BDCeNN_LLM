"""
Central experiment orchestrator. Coordinates topology generation, solver
execution across both interference modes, metrics aggregation, and optional
LLM auditing. This is the single entry point used by the Streamlit dashboard.
"""

import time
from typing import Optional, Callable, List
import numpy as np

import config
from data_structures import (
    NetworkTopology,
    SolverResult,
    ExperimentRecord,
)
from data_generator import generate_network
from graph_model import InterferenceGraph
from bdcenn_solver import bdcenn_allocation
from baselines import (
    random_allocation,
    greedy_allocation,
    dsatur_allocation,
)


# ---------------------------------------------------------------------------
# Topology construction
# ---------------------------------------------------------------------------

def build_topology(N: int, K: int, area: float, threshold: float,
                   seed: int, scenario_name: Optional[str] = None) -> NetworkTopology:
    """
    Generates a NetworkTopology by delegating to data_generator.generate_network
    and wrapping the result in the standard dataclass.
    """
    W, positions, _ = generate_network(N, K, area, threshold, seed)
    return NetworkTopology(
        n_cells=N,
        n_channels=K,
        positions=positions,
        weight_matrix=W,
        threshold=threshold,
        area=area,
        seed=seed,
        scenario_name=scenario_name,
    )


# ---------------------------------------------------------------------------
# Solver execution wrappers
# ---------------------------------------------------------------------------

def _run_bdcenn(graph: InterferenceGraph, mode: str,
                num_restarts: int, max_iter: int, seed: int) -> SolverResult:
    """
    Runs the BD-CeNN multistart solver for a given interference mode
    and wraps the raw tuple output into a SolverResult dataclass.
    """
    M = graph.M if mode == "cci_aci" else None

    start = time.perf_counter()
    x_final, history, elapsed_internal, _, best_iter = bdcenn_allocation(
        graph.N, graph.K, graph.W, M=M,
        num_restarts=num_restarts,
        max_iter=max_iter,
        random_order=True,
        seed=seed,
        verbose=False,
    )
    elapsed = time.perf_counter() - start

    # Extract initial assignment from history
    initial_assignment = None
    if history and len(history) > 0:
        first_entry = history[0]
        if len(first_entry) >= 3:
            initial_assignment = np.asarray(first_entry[2], dtype=int)

    cost = graph.compute_cost(x_final, mode)
    conflicts = graph.count_conflicts(x_final, mode)
    used_channels = len(set(x_final.tolist()))

    return SolverResult(
        solver_name="BD-CeNN",
        assignment=np.asarray(x_final, dtype=int),
        cost=cost,
        interference_mode=mode,
        n_conflicts=conflicts,
        used_channels=used_channels,
        wall_time_seconds=elapsed,
        n_iterations=len(history) - 1 if history else 0,
        best_iteration=best_iter,
        energy_history=history,
        initial_assignment=initial_assignment,
        metadata={"num_restarts": num_restarts, "max_iter": max_iter},
    )


def _run_baseline(name: str, graph: InterferenceGraph, mode: str,
                   seed: int) -> SolverResult:
    """
    Executes a baseline heuristic and wraps its output into a SolverResult.
    """
    M = graph.M if mode == "cci_aci" else None

    start = time.perf_counter()
    if name == "Random":
        np.random.seed(seed)
        x = random_allocation(graph.N, graph.K)
    elif name == "Greedy":
        np.random.seed(seed)
        order = np.random.permutation(graph.N).tolist()
        x = greedy_allocation(graph.N, graph.K, graph.W, order=order, M=M)
    elif name == "DSATUR":
        x = dsatur_allocation(graph.N, graph.K, graph.W, M=M)
    else:
        raise ValueError(f"Unknown baseline: {name!r}")
    elapsed = time.perf_counter() - start

    x = np.asarray(x, dtype=int)
    cost = graph.compute_cost(x, mode)
    conflicts = graph.count_conflicts(x, mode)
    used_channels = len(set(x.tolist()))

    return SolverResult(
        solver_name=name,
        assignment=x,
        cost=cost,
        interference_mode=mode,
        n_conflicts=conflicts,
        used_channels=used_channels,
        wall_time_seconds=elapsed,
        n_iterations=0,
        best_iteration=0,
        energy_history=[],
        initial_assignment=None,
        metadata={},
    )


# ---------------------------------------------------------------------------
# ExperimentRunner class
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """
    Coordinates the full execution of a single experiment across all solvers
    and both interference modes. Produces a self-contained ExperimentRecord.
    """

    SOLVER_NAMES = ["BD-CeNN", "Random", "Greedy", "DSATUR"]
    INTERFERENCE_MODES = ["cci", "cci_aci"]

    def __init__(self,
                 num_restarts: int = None,
                 max_iter: int = None):
        """
        Args:
            num_restarts: BD-CeNN multistart count (defaults to config.NUM_RESTARTS)
            max_iter: BD-CeNN max iterations per restart (defaults to config.MAX_ITER_BD)
        """
        self.num_restarts = num_restarts if num_restarts is not None else config.NUM_RESTARTS
        self.max_iter = max_iter if max_iter is not None else config.MAX_ITER_BD

    def run_single(self,
                   topology: NetworkTopology,
                   modes: Optional[List[str]] = None,
                   solvers: Optional[List[str]] = None,
                   progress_callback: Optional[Callable[[str, float], None]] = None
                   ) -> ExperimentRecord:
        """
        Executes all configured solvers for the given topology under all
        requested interference modes.

        Args:
            topology: NetworkTopology to solve
            modes: Subset of ["cci", "cci_aci"] (defaults to both)
            solvers: Subset of SOLVER_NAMES (defaults to all)
            progress_callback: Optional callable receiving (message, progress_ratio)

        Returns:
            Fully populated ExperimentRecord
        """
        if modes is None:
            modes = self.INTERFERENCE_MODES
        if solvers is None:
            solvers = self.SOLVER_NAMES

        record = ExperimentRecord(topology=topology)
        graph = InterferenceGraph(topology)

        total_steps = len(modes) * len(solvers)
        current_step = 0

        def _report(msg: str):
            if progress_callback is not None:
                ratio = current_step / total_steps if total_steps > 0 else 1.0
                progress_callback(msg, ratio)

        for mode in modes:
            mode_label = "CCI-only" if mode == "cci" else "CCI+ACI"
            for solver in solvers:
                _report(f"Running {solver} ({mode_label})...")
                if solver == "BD-CeNN":
                    result = _run_bdcenn(
                        graph, mode,
                        num_restarts=self.num_restarts,
                        max_iter=self.max_iter,
                        seed=topology.seed,
                    )
                else:
                    result = _run_baseline(solver, graph, mode, seed=topology.seed)
                record.add_solver_result(result)
                current_step += 1
                _report(f"Completed {solver} ({mode_label}).")

        # Compute Delta_ACI on BD-CeNN if both modes are available
        record.compute_delta_aci("BD-CeNN")

        _report("Experiment finalized.")
        return record

    def run_batch(self,
                  topologies: List[NetworkTopology],
                  progress_callback: Optional[Callable[[str, float], None]] = None
                  ) -> List[ExperimentRecord]:
        """
        Runs the standard pipeline over a list of topologies.
        """
        records = []
        total = len(topologies)
        for i, topo in enumerate(topologies):
            def local_cb(msg, ratio):
                if progress_callback is not None:
                    global_ratio = (i + ratio) / total
                    progress_callback(f"[{i+1}/{total}] {msg}", global_ratio)
            record = self.run_single(topo, progress_callback=local_cb)
            records.append(record)
        return records