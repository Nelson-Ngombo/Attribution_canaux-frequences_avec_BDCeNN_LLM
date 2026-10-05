"""
Central experiment orchestrator. Coordinates topology generation, solver
execution across both interference modes, metrics aggregation, and optional
LLM auditing. This is the single entry point used by the Streamlit dashboard.
"""


import time
import warnings
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
from metrics import (
    compute_cost_cci,
    compute_cost_cci_aci,
    count_conflicts_cci,
    count_conflicts_aci,
    create_channel_interference_matrix,
)


# ============================================================================
# 1. CONSTRUCTION DE TOPOLOGIE
# ============================================================================

def build_topology(N: int, K: int, area: float, threshold: float,
                   seed: int, scenario_name: Optional[str] = None) -> NetworkTopology:
    """
    Génère une NetworkTopology en déléguant à data_generator.generate_network.

    Paramètres
    ----------
    N : int
        Nombre de cellules.
    K : int
        Nombre de canaux.
    area : float
        Taille de la zone de simulation.
    threshold : float
        Seuil de portée radio.
    seed : int
        Graine aléatoire.
    scenario_name : str or None
        Identifiant du scénario (ex. "S3").

    Returns
    -------
    topology : NetworkTopology
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


# ============================================================================
# 2. WRAPPERS D'EXÉCUTION DES SOLVEURS
# ============================================================================

def _run_bdcenn(graph: InterferenceGraph, mode: str,
                num_restarts: int, max_sweeps: int, seed: int) -> SolverResult:
    """
    Exécute le solveur BD-CeNN multistart pour un régime d'interférence donné.

    Paramètres
    ----------
    graph : InterferenceGraph
        Graphe d'interférence.
    mode : str
        "cci" ou "cci_aci".
    num_restarts : int
        Nombre de restarts.
    max_sweeps : int
        Nombre maximal de sweeps par restart.
    seed : int
        Graine aléatoire de base.

    Returns
    -------
    result : SolverResult
    """
    M = graph.M if mode == "cci_aci" else None

    start = time.perf_counter()
    (x_final, history, elapsed_internal,
     n_cci, n_aci, best_sweep) = bdcenn_allocation(
        graph.N, graph.K, graph.W, M=M,
        num_restarts=num_restarts,
        max_sweeps=max_sweeps,
        random_order=True,
        seed=seed,
        verbose=False,
    )
    elapsed = time.perf_counter() - start

    # Extraction de l'allocation initiale depuis l'historique
    initial_assignment = None
    if history and len(history) > 0:
        first_entry = history[0]
        if len(first_entry) >= 3:
            initial_assignment = np.asarray(first_entry[2], dtype=int)

    cost = graph.compute_cost(x_final, mode)
    used_channels = len(set(x_final.tolist()))

    # En mode CCI-only, n_aci est toujours 0
    if mode == "cci":
        n_aci = 0
    n_total = n_cci + n_aci

    return SolverResult(
        solver_name="BD-CeNN",
        assignment=np.asarray(x_final, dtype=int),
        cost=cost,
        interference_mode=mode,
        n_conflicts_cci=n_cci,
        n_conflicts_aci=n_aci,
        n_conflicts_total=n_total,
        used_channels=used_channels,
        wall_time_seconds=elapsed,
        n_sweeps=len(history) - 1 if history else 0,
        best_sweep_index=best_sweep,
        energy_history=history,
        initial_assignment=initial_assignment,
        metadata={"num_restarts": num_restarts, "max_sweeps": max_sweeps},
    )


def _run_baseline(name: str, graph: InterferenceGraph, mode: str,
                  seed: int) -> SolverResult:
    """
    Exécute une baseline heuristique et calcule les métriques de conflit.

    Paramètres
    ----------
    name : str
        "Random", "Greedy" ou "DSATUR".
    graph : InterferenceGraph
        Graphe d'interférence.
    mode : str
        "cci" ou "cci_aci".
    seed : int
        Graine aléatoire.

    Returns
    -------
    result : SolverResult
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
        raise ValueError(f"Baseline inconnue : {name!r}")
    elapsed = time.perf_counter() - start

    x = np.asarray(x, dtype=int)
    cost = graph.compute_cost(x, mode)
    used_channels = len(set(x.tolist()))

    # Conflits détaillés (Chantier A)
    n_cci = count_conflicts_cci(x, graph.W)
    if mode == "cci_aci":
        n_aci = count_conflicts_aci(x, graph.W, cutoff=graph.aci_cutoff)
    else:
        n_aci = 0
    n_total = n_cci + n_aci

    return SolverResult(
        solver_name=name,
        assignment=x,
        cost=cost,
        interference_mode=mode,
        n_conflicts_cci=n_cci,
        n_conflicts_aci=n_aci,
        n_conflicts_total=n_total,
        used_channels=used_channels,
        wall_time_seconds=elapsed,
        n_sweeps=0,
        best_sweep_index=0,
        energy_history=[],
        initial_assignment=None,
        metadata={},
    )


# ============================================================================
# 3. ORCHESTRATEUR PRINCIPAL
# ============================================================================

class ExperimentRunner:
    """
    Coordonne l'exécution complète d'une expérience sur tous les solveurs
    et les deux régimes d'interférence. Produit un ExperimentRecord.

    Parameters
    ----------
    num_restarts : int or None
        Nombre de restarts BD-CeNN (défaut : config.NUM_RESTARTS).
    max_sweeps : int or None
        Sweeps max par restart (défaut : config.MAX_SWEEPS_BD).
    """

    SOLVER_NAMES = ["BD-CeNN", "Random", "Greedy", "DSATUR"]
    INTERFERENCE_MODES = ["cci", "cci_aci"]

    def __init__(self,
                 num_restarts: int = None,
                 max_sweeps: int = None,
                 **kwargs):
        # Rétrocompatibilité : max_iter → max_sweeps
        if "max_iter" in kwargs:
            warnings.warn(
                "Le paramètre 'max_iter' est déprécié. Utiliser 'max_sweeps'.",
                DeprecationWarning,
                stacklevel=2,
            )
            max_sweeps = kwargs.pop("max_iter")

        self.num_restarts = num_restarts if num_restarts is not None else config.NUM_RESTARTS
        self.max_sweeps = max_sweeps if max_sweeps is not None else config.MAX_SWEEPS_BD

    def run_single(self,
                   topology: NetworkTopology,
                   modes: Optional[List[str]] = None,
                   solvers: Optional[List[str]] = None,
                   progress_callback: Optional[Callable[[str, float], None]] = None
                   ) -> ExperimentRecord:
        """
        Exécute tous les solveurs configurés pour la topologie donnée.

        Paramètres
        ----------
        topology : NetworkTopology
            Topologie à résoudre.
        modes : list of str or None
            Sous-ensemble de ["cci", "cci_aci"]. Par défaut les deux.
        solvers : list of str or None
            Sous-ensemble de SOLVER_NAMES. Par défaut tous.
        progress_callback : callable or None
            Fonction (message, ratio) pour le suivi de progression.

        Returns
        -------
        record : ExperimentRecord
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
                _report(f"Execution de {solver} ({mode_label})...")
                if solver == "BD-CeNN":
                    result = _run_bdcenn(
                        graph, mode,
                        num_restarts=self.num_restarts,
                        max_sweeps=self.max_sweeps,
                        seed=topology.seed,
                    )
                else:
                    result = _run_baseline(solver, graph, mode, seed=topology.seed)
                record.add_solver_result(result)
                current_step += 1
                _report(f"Termine : {solver} ({mode_label}).")

        # Calcul du Delta_ACI sur BD-CeNN si les deux modes sont disponibles
        record.compute_delta_aci("BD-CeNN")

        _report("Experience terminee.")
        return record

    def run_batch(self,
                  topologies: List[NetworkTopology],
                  progress_callback: Optional[Callable[[str, float], None]] = None
                  ) -> List[ExperimentRecord]:
        """
        Exécute le pipeline standard sur une liste de topologies.

        Paramètres
        ----------
        topologies : list of NetworkTopology
        progress_callback : callable or None

        Returns
        -------
        records : list of ExperimentRecord
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