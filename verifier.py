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

# -----------------------------------------------------------------------------
# Revendications structurées (méthode, métrique, valeur) et attributions de canaux
# -----------------------------------------------------------------------------
METHOD_NAMES = ("BD-CeNN", "Greedy", "DSATUR", "Random")
_METHOD_ALTERNATION = "|".join(re.escape(name) for name in METHOD_NAMES)
_METHOD_TOKEN = re.compile(rf"(?<![\w-])({_METHOD_ALTERNATION})(?![\w-])", re.IGNORECASE)
_CANONICAL_METHOD = {name.lower(): name for name in METHOD_NAMES}

_METRIC_PATTERNS = {
    "C_CCI": r"C_\{?CCI\}?",
    "C_ACI": r"C_\{?ACI\}?",
    "C_tot": r"C_\{?tot(?:al)?\}?",
    "time": (r"temps(?:\s+d['’]ex[ée]cution)?(?:\s+de\s+calcul)?(?:\s+t_\{?exec\}?)?"
             r"|t_\{?exec\}?"),
    "cost": (r"co[ûu]t(?:\s+(?:final|initial|global|total|[ée]nerg[ée]tique))?"
             r"(?:\s+J(?:\s*\(\s*x\s*\*?\s*\)|_\{?0\}?)?)?"
             r"|[ée]nergie(?:\s+finale|\s+initiale)?|J\s*\(\s*x\s*\*?\s*\)|J_\{?0\}?"),
}
_METRIC_ALTERNATION = "|".join(f"(?P<{name}>{pattern})"
                               for name, pattern in _METRIC_PATTERNS.items())
_CLAIM_SEPARATOR = (r"(?:=|:|\bde\b|\bvaut\b|\best\s+de\b|\best\b|\batteint\b|\baffiche\b"
                    r"|\bobtient\b|\benregistre\b|\bs['’][ée]l[èe]ve\s+[àa]"
                    r"|\bs['’][ée]tablit\s+[àa])")
# Métrique, séparateur, valeur ; la méthode est résolue autour du motif
_METRIC_CLAIM = re.compile(
    rf"(?<!\w)(?:{_METRIC_ALTERNATION})(?P<between>[^.\n;=\d]{{0,30}}?)\s*{_CLAIM_SEPARATOR}"
    r"\s*(?:d['’]environ\s+|environ\s+|seulement\s+)?"
    r"(?P<value>-?\d+(?:[.,]\d+)?)(?!\d|[.,]\d)(?!\s*%)",
    re.IGNORECASE)
_METHOD_AFTER_VALUE = re.compile(
    r"\s*(?:ms|s)?\s*\)?\s*,?\s*\(?\s*(?:pour|du|de|chez|avec|par|obtenu\s+par|atteint\s+par)\s+"
    r"(?:la\s+m[ée]thode\s+|l['’]heuristique\s+|l['’]algorithme\s+|le\s+solveur\s+)?"
    rf"({_METHOD_ALTERNATION})(?![\w-])", re.IGNORECASE)
METHOD_WINDOW_WORDS = 6
_CLAIM_SENTENCE_BREAK = re.compile(r"[.!?;](?=\s|$)|\n")


def _sentence_bounds(text: str, position: int) -> tuple:
    start = 0
    for brk in _CLAIM_SENTENCE_BREAK.finditer(text, 0, position):
        start = brk.end()
    end_match = _CLAIM_SENTENCE_BREAK.search(text, position)
    return start, (end_match.start() if end_match else len(text))


def _resolve_method(text, match, sentence_start, window_start, sentence_end):
    """
    Ordre de priorité : méthode interne au motif (« coût du BD-CeNN »), puis
    introduite après la valeur (« = 26 pour Greedy »), puis dernière méthode
    des METHOD_WINDOW_WORDS mots précédents, enfin sujet unique de la phrase.
    """
    inside = _METHOD_TOKEN.search(match.group("between") or "")
    if inside:
        return _CANONICAL_METHOD[inside.group(1).lower()]
    after = _METHOD_AFTER_VALUE.match(text, match.end(), sentence_end)
    if after:
        return _CANONICAL_METHOD[after.group(1).lower()]
    words = text[window_start:match.start()].split()[-METHOD_WINDOW_WORDS:]
    before = _METHOD_TOKEN.findall(" ".join(words))
    if before:
        return _CANONICAL_METHOD[before[-1].lower()]
    subjects = {m.lower() for m in _METHOD_TOKEN.findall(text[sentence_start:match.start()])}
    return _CANONICAL_METHOD[subjects.pop()] if len(subjects) == 1 else None


def extract_metric_claims(text: str) -> list:
    """Revendications (méthode, métrique, valeur) ; sans méthode identifiable, rien n'est retenu."""
    claims = []
    previous_end, previous_sentence = 0, None
    for match in _METRIC_CLAIM.finditer(text):
        sentence_start, sentence_end = _sentence_bounds(text, match.start())
        # La fenêtre avant ne remonte pas au-delà de la revendication précédente
        window_start = (max(sentence_start, previous_end)
                        if previous_sentence == sentence_start else sentence_start)
        method = _resolve_method(text, match, sentence_start, window_start, sentence_end)
        previous_end, previous_sentence = match.end(), sentence_start
        if method is None:
            continue
        metric = next(name for name in _METRIC_PATTERNS if match.group(name))
        claims.append({
            "method": method,
            "metric": metric,
            "value": float(match.group("value").replace(",", ".")),
            "raw": match.group(0),
            "start": match.start(),
        })
    return claims


_CELL_MENTION = re.compile(r"(?<!\w)cellule\s+(\d+)(?!\d)", re.IGNORECASE)
_CHANNEL_MENTION = re.compile(r"(?<!\w)canal\s+(\d+)(?!\d|[.,]\d)", re.IGNORECASE)
_SEGMENT_STOP = re.compile(r"[!?;\n]|\.(?!\d)|cellule|canaux|voisin", re.IGNORECASE)
_ASSIGNMENT_VERB = re.compile(r"attribu|affect|allou|assign", re.IGNORECASE)
CHANNEL_SEARCH_CHARS = 45


def _channel_after(text, cell_match):
    """« cellule i ... canal c », sans autre cellule ni fin de phrase entre les deux."""
    end = min(len(text), cell_match.end() + CHANNEL_SEARCH_CHARS)
    stop = _SEGMENT_STOP.search(text, cell_match.end(), end)
    if stop:
        end = stop.start()
    found = _CHANNEL_MENTION.search(text, cell_match.end())
    return found if found and found.start() < end else None


def _channel_before(text, cell_match):
    """« canal c attribué à la cellule i » : verbe d'affectation obligatoire."""
    segment_start = max(0, cell_match.start() - CHANNEL_SEARCH_CHARS)
    for stop in _SEGMENT_STOP.finditer(text, segment_start, cell_match.start()):
        segment_start = stop.end()
    if not _ASSIGNMENT_VERB.search(text, segment_start, cell_match.start()):
        return None
    found = list(_CHANNEL_MENTION.finditer(text, segment_start, cell_match.start()))
    return found[-1] if found else None


def check_channel_claims(text: str, allocations: dict, scenario: str = "") -> list:
    """
    Confronte chaque attribution « cellule i / canal c » au vecteur de la méthode
    nommée dans la phrase (BD-CeNN si aucune). Une attribution fausse ou une
    cellule hors réseau donne un quadruplet invalide.
    """
    claims = []
    for cell_match in _CELL_MENTION.finditer(text):
        channel_match = _channel_after(text, cell_match) or _channel_before(text, cell_match)
        if channel_match is None:
            continue
        cell, channel = int(cell_match.group(1)), int(channel_match.group(1))
        sentence_start, sentence_end = _sentence_bounds(text, cell_match.start())
        named = list(dict.fromkeys(
            _CANONICAL_METHOD[m.lower()]
            for m in _METHOD_TOKEN.findall(text[sentence_start:sentence_end])))
        candidates = named or ["BD-CeNN"]
        matched = [m for m in candidates
                   if cell < len(allocations.get(m, []))
                   and int(allocations[m][cell]) == channel]
        start = min(cell_match.start(), channel_match.start())
        end = max(cell_match.end(), channel_match.end())
        claims.append({
            "scenario": scenario,
            "method": matched[0] if matched else candidates[0],
            "metric": f"canal[cellule {cell}]",
            "value": float(channel),
            "raw": text[start:end],
            "start": start,
            "valid": bool(matched),
        })
    return claims


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
        # Fragments numériques de l'identifiant alphanumérique (ex. DASHBOARD_5c5534d8)
        cid = str(case_data.get("case_id", "")).replace("#", "")
        for num_str in re.findall(r"\d+", cid):
            add(int(num_str))

        # Digits du nom de scenario
        for num_str in re.findall(r"\d+", str(case_data.get("scenario", ""))):
            try:
                add(int(num_str))
            except ValueError:
                pass

        # Bornes d'echelle academique
        for n in range(1, 6):
            add(n)

        # [FIX-LLM-SWEEPS] "iterations" remplace par "sweeps", ajout de "num_restarts"
        m = case_data.get("metrics", {})
        for key in ["cost_initial", "cost_final", "conflicts", "conflicts_cci",
                    "conflicts_aci", "time_seconds", "sweeps", "used_channels",
                    "num_restarts"]:
            if key in m:
                add(m[key])
        # Durées admises aussi en millisecondes (unité d'affichage)
        if "time_seconds" in m:
            add(float(m["time_seconds"]) * 1e3)

        # Baselines
        for base_data in case_data.get("baselines", {}).values():
            for k in ["cost", "conflicts_cci", "conflicts_aci", "conflicts_total", "time"]:
                if k in base_data:
                    add(base_data[k])
            # Conversion en ms appliquée à chaque baseline, pas seulement à la dernière
            if "time" in base_data:
                add(float(base_data["time"]) * 1e3)

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
            add(w)

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
        # [FIX-LLM-TOL] Ajout d'une tolerance absolue en plus de la tolerance
        # relative, pour eviter les faux positifs sur les petites valeurs et
        # les faux negatifs sur les valeurs arrondies.
        abs_tol = 0.01
        for a in allowed:
            if val == a:
                return True, a
            if a == 0:
                if abs(val) <= abs_tol:
                    return True, 0.0
                continue
            if abs(val - a) / abs(a) <= self.tolerance:
                return True, a
            if abs(val - a) <= abs_tol:
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


def combine_numeric_status(audit_report: AuditReport, structured: dict = None) -> str:
    """
    Statut numérique final : la whitelist (AuditReport) ET les quadruplets
    (scénario, méthode, métrique, valeur) doivent être conformes.
    """
    if structured and structured.get("invalid_claims"):
        return "FAILED"
    return audit_report.status