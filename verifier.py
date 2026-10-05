"""
Independent regex-based verification module.
Extracted from the LLM assistant to provide a standalone auditor
that can be used and tested in isolation.
"""

import re
from typing import List, Set
import numpy as np

from data_structures.audit_report import AuditReport, AuditVerification

NUMBER_REGEX = re.compile(r"-?\d+(?:[.,]\d+)?")


class ResponseVerifier:
    """
    Contrôleur regex auditant les affirmations numeriques d'un texte d'analyse.
    """

    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance

    def extract_numbers(self, text: str) -> List[tuple]:
        """
        Extrait tous les nombres sous format flottant et conserve leur representation brute.
        """
        results = []
        for match in NUMBER_REGEX.findall(text):
            raw = match
            normalized = raw.replace(",", ".")
            try:
                val = float(normalized)
                results.append((val, raw))
            except ValueError:
                continue
        return results

    def collect_allowed_numbers(self, case_data: dict) -> Set[float]:
        """
        Assemble la whitelist dynamique des valeurs certifiees pour le cas donne.
        """
        allowed = set()

        def add(v):
            try:
                allowed.add(round(float(v), 6))
            except Exception:
                pass

        # Parametres structurels du reseau
        for key in ["N", "K", "seed"]:
            if key in case_data:
                add(case_data[key])

        # Identifiant unique de run
        cid = str(case_data.get("case_id", "")).replace("#", "")
        if cid.isdigit():
            add(int(cid))

        # Digits du nom de scenario
        for num_str in re.findall(r"\d+", str(case_data.get("scenario", ""))):
            try:
                add(int(num_str))
            except ValueError:
                pass

        # Bornes d'echelle academique
        for n in range(1, 6):
            add(n)

        # Metriques de simulation (Chantier A + D)
        m = case_data.get("metrics", {})
        for key in ["cost_initial", "cost_final", "conflicts", "conflicts_cci",
                    "conflicts_aci", "time_seconds", "iterations", "used_channels"]:
            if key in m:
                add(m[key])

        # Baselines
        for base_data in case_data.get("baselines", {}).values():
            for k in ["cost", "conflicts_cci", "conflicts_aci", "conflicts_total", "time"]:
                if k in base_data:
                    add(base_data[k])

        # Cellules en conflit direct
        for c in case_data.get("conflicting_cells", []):
            add(c)

        # Geometrie spatiale (coordonnees)
        for cid_pos, xy in case_data.get("cell_positions", {}).items():
            try:
                add(int(cid_pos))
            except Exception:
                pass
            if isinstance(xy, (list, tuple)) and len(xy) == 2:
                add(xy[0])
                add(xy[1])

        # Graphe d'aretes
        for (i, j, w) in case_data.get("topology_edges", []):
            add(int(i))
            add(int(j))
            add(int(w))

        if "total_edges" in case_data:
            add(case_data["total_edges"])
        if "topology_edges" in case_data:
            add(len(case_data["topology_edges"]))

        # Metadonnees de topologie
        tm = case_data.get("topology_meta", {})
        for key in ["seed", "threshold", "N", "K"]:
            if key in tm:
                add(tm[key])

        # Vecteurs d'allocations
        for alloc_list in case_data.get("allocations", {}).values():
            for c in alloc_list:
                add(c)

        # Profil des noeuds satures
        for entry in case_data.get("cell_summary", []):
            add(entry.get("cell", 0))
            add(entry.get("degree", 0))
            add(entry.get("strength", 0))

        return allowed

    def _matches_any_allowed(self, val: float, allowed: Set[float]) -> tuple:
        for a in allowed:
            if a == 0 and val == 0:
                return True, 0.0
            if a != 0 and abs(val - a) / abs(a) <= self.tolerance:
                return True, a
            if abs(val - a) < 1e-9:
                return True, a
        return False, 0.0

    def audit(self, llm_text: str, case_data: dict) -> AuditReport:
        """
        Compare de maniere stricte les assertions textuelles aux donnees de la whitelist.
        """
        extracted = self.extract_numbers(llm_text)
        allowed = self.collect_allowed_numbers(case_data)

        verifications = []
        valid_count = 0

        for val, raw in extracted:
            matched, ref = self._matches_any_allowed(val, allowed)
            verifications.append(AuditVerification(
                extracted_raw=raw,
                extracted_value=val,
                matched=matched,
                matched_reference=ref,
                tolerance=self.tolerance,
            ))
            if matched:
                valid_count += 1

        total = len(extracted)
        invented_count = total - valid_count
        accuracy = (valid_count / total * 100.0) if total > 0 else 100.0

        if total == 0:
            status = "WARNING"
        elif invented_count == 0:
            status = "CERTIFIED"
        else:
            status = "FAILED"

        return AuditReport(
            status=status,
            accuracy_rate=accuracy,
            total_numbers=total,
            valid_count=valid_count,
            invented_count=invented_count,
            verifications=verifications,
            tolerance=self.tolerance,
        )