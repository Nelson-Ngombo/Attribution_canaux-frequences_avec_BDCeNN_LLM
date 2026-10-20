"""
Tab 5: LLM Analysis Report with Independent Regex-based Audit.
"""

import streamlit as st
import numpy as np

import config
from dashboard import session_manager as sm
from dashboard import toast
from dashboard.chart_factory import MODE_LABELS
from dashboard.components import (
    section_title,
    empty_state,
    info_banner,
    mode_selector,
    render_error_box,
)


def _build_case_data(record, mode: str) -> dict:
    """Assembles the case_data payload for llm_assistant.build_prompt."""
    from graph_model import InterferenceGraph

    topology = record.topology
    bd_result = record.solver_results.get("BD-CeNN", {}).get(mode)
    if bd_result is None:
        return None

    baselines = {}
    for name in ["Random", "Greedy", "DSATUR"]:
        bres = record.solver_results.get(name, {}).get(mode)
        if bres:
            baselines[name] = {
                "cost": float(bres.cost),
                "conflicts_cci": int(bres.n_conflicts_cci),
                "conflicts_aci": int(bres.n_conflicts_aci),
                "conflicts_total": int(bres.n_conflicts_total),
                "time": float(bres.wall_time_seconds),
            }

    graph = InterferenceGraph(topology)
    if bd_result.initial_assignment is not None:
        cost_init = graph.compute_cost(bd_result.initial_assignment, mode)
    else:
        cost_init = 0.0
 
    conflict_edges = graph.get_conflict_edges_cci(bd_result.assignment)
    if mode == "cci_aci":
        conflict_edges += graph.get_conflict_edges_aci(bd_result.assignment)
    conflicting_cells = sorted(set(c for edge in conflict_edges for c in edge))

    allocations = {}
    for name, modes_dict in record.solver_results.items():
        if mode in modes_dict:
            allocations[name] = [int(c) for c in modes_dict[mode].assignment]

    cell_positions = {
        int(i): [float(topology.positions[i, 0]), float(topology.positions[i, 1])]
        for i in range(topology.N)
    }

    edges_all = []
    for i in range(topology.N):
        for j in range(i + 1, topology.N):
            if topology.W[i, j] > 0:
                edges_all.append((int(i), int(j), int(topology.W[i, j])))
    edges_all.sort(key=lambda e: -e[2])
    total_edges = len(edges_all)
    topology_edges = edges_all[:80]

    N = topology.N
    cell_degree = np.zeros(N, dtype=int)
    cell_strength = np.zeros(N, dtype=int)
    for (i, j, w) in edges_all:
        cell_degree[i] += 1
        cell_degree[j] += 1
        cell_strength[i] += w
        cell_strength[j] += w
    top_idx = np.argsort(-cell_strength)[:10]
    cell_summary = [
        {"cell": int(idx), "degree": int(cell_degree[idx]), "strength": int(cell_strength[idx])}
        for idx in top_idx
    ]

    topology_meta = {
        "seed": int(topology.seed),
        "threshold": float(topology.threshold),
        "N": int(topology.N),
        "K": int(topology.K),
    }

    model_label = "CCI-only" if mode == "cci" else "CCI+ACI"

    # Hyperparamètres réellement utilisés par le run (sliders du dashboard)
    bd_meta = bd_result.metadata or {}
    run_metadata = config.build_run_metadata(
        scenario=topology.scenario_name or "CUSTOM",
        topology_seed=topology.seed,
        interference_mode=mode,
        num_restarts=bd_meta.get("num_restarts", config.NUM_RESTARTS),
        max_sweeps=bd_meta.get("max_sweeps", config.MAX_SWEEPS_BD),
    )

    return {
        "case_id": f"DASHBOARD_{record.experiment_id}",
        "scenario": topology.scenario_name or "CUSTOM",
        "N": topology.N,
        "K": topology.K,
        "seed": topology.seed,
        "context": f"Analyse interactive depuis le dashboard [{model_label}]",
        "model_label": model_label,
        "metrics": {
            "cost_initial": float(cost_init),
            "cost_final": float(bd_result.cost),
            "conflicts": int(bd_result.n_conflicts_total),  # Total conflicts (Chantier A)
            "conflicts_cci": int(bd_result.n_conflicts_cci),
            "conflicts_aci": int(bd_result.n_conflicts_aci),
            "time_seconds": float(bd_result.wall_time_seconds),
            "sweeps": int(bd_result.n_sweeps),  # sweeps mapped for LLM key
            "used_channels": int(bd_result.used_channels),
            # Budget réellement utilisé : cité dans le prompt et admis par la whitelist
            "num_restarts": int(bd_meta.get("num_restarts", config.NUM_RESTARTS)),
        },
        "baselines": baselines,
        "conflicting_cells": conflicting_cells,
        "allocations": allocations,
        "cell_positions": cell_positions,
        "topology_edges": topology_edges,
        "total_edges": total_edges,
        "cell_summary": cell_summary,
        "topology_meta": topology_meta,
        # Hors prompt : en-tête des PDF et panneau de traçabilité
        "algorithm_seed": run_metadata["algorithm_seed"],
        "run_metadata": run_metadata,
    }


def _generate_llm_report(mode: str):
    record = sm.get("experiment_record")
    if record is None:
        toast.error("Aucun resultat disponible.")
        return

    case_data = _build_case_data(record, mode)
    if case_data is None:
        toast.error(f"Aucun resultat BD-CeNN pour le mode {MODE_LABELS[mode]}.")
        return

    sm.set_value(f"llm_case_data_{mode}", case_data)

    try:
        from llm_assistant import build_prompt, ask_llm_detailed, detect_truncation
        prompt = build_prompt(case_data)

        with st.spinner(f"Interrogation du LLM pour {MODE_LABELS[mode]}..."):
            call = ask_llm_detailed(prompt)

        # Même typologie d'échec que la campagne E10 (QUOTA_DAILY, UNAVAILABLE...)
        sm.set_value(f"llm_error_code_{mode}", call["error_code"])
        response = (f"[ERREUR LLM] {call['error_code']} : {call['error_message']}"
                    if call["error_code"] else call["text"])
        sm.set_value(f"llm_response_{mode}", response)

        # Échec d'appel : aucun audit sur un message d'erreur
        if call["error_code"]:
            for stale_key in ("audit_report", "audit_struct", "numeric_status",
                              "audit_semantic", "llm_truncation"):
                sm.set_value(f"{stale_key}_{mode}", None)
            toast.error(f"Le LLM n'a pas produit de rapport ({call['error_code']}).")
            return

        # Rapport interrompu : affiché, jamais certifié
        truncation = detect_truncation(response, call["finish_reason"])
        sm.set_value(f"llm_truncation_{mode}", truncation)

        from llm_assistant import (verify_structured_claims, semantic_audit,
                                   get_llm_inference_params)
        from verifier import combine_numeric_status

        # Paramètres et modèle ayant effectivement produit ce rapport
        sm.set_value(f"llm_params_{mode}", get_llm_inference_params())

        verifier = sm.get("verifier")
        audit = verifier.audit(response, case_data)
        sm.set_value(f"audit_report_{mode}", audit)

        # Audit numérique bloquant : whitelist ET quadruplets
        # (scénario, méthode, métrique, valeur)
        audit_struct = verify_structured_claims(response, case_data)
        sm.set_value(f"audit_struct_{mode}", audit_struct)
        numeric_status = ("TRUNCATED" if truncation["truncated"]
                          else combine_numeric_status(audit, audit_struct))
        sm.set_value(f"numeric_status_{mode}", numeric_status)

        # Audit sémantique séparé, sans effet sur la certification numérique
        sm.set_value(f"audit_semantic_{mode}",
                     semantic_audit(response, case_data, truncated=truncation["truncated"]))

        if numeric_status == "TRUNCATED":
            toast.warning("Rapport TRONQUE : affiche sans certification numerique")
        elif numeric_status == "CERTIFIED":
            toast.success(f"Rapport genere et certifie ({MODE_LABELS[mode]})")
        elif numeric_status == "FAILED":
            toast.warning(
                f"Rapport NON certifie : {audit.invented_count} valeur(s) hors whitelist, "
                f"{len(audit_struct['invalid_claims'])} quadruplet(s) mal attribue(s)"
            )
        else:
            toast.info("Rapport genere (audit WARNING : aucune valeur numerique)")

    except Exception as e:
        import traceback
        render_error_box(type(e).__name__, str(e), traceback.format_exc())


def _render_audit_badge(audit_report, numeric_status=None, n_invalid_quadruplets=0,
                        truncation=None):
    if audit_report is None:
        info_banner("Aucun audit realise.", kind="info")
        return

    # Statut combiné (whitelist + quadruplets) prioritaire sur la seule whitelist
    status = numeric_status or audit_report.status
    if status == "TRUNCATED":
        # Un rapport tronqué n'affiche jamais CERTIFIE
        reasons = "; ".join((truncation or {}).get("reasons", [])) or "reponse incomplete"
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
                        border-left: 4px solid #dc2626; padding: 1rem; border-radius: 8px;">
                <div style="font-size: 1.1rem; font-weight: 700; color: #991b1b;">
                    TRONQUE - RAPPORT NON CERTIFIE
                </div>
                <div style="color: #b91c1c; font-size: 0.9rem; margin-top: 0.3rem;">
                    Reponse interrompue ({reasons}). Aucun chiffre n'est certifie.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif status == "CERTIFIED":
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
                        border-left: 4px solid #059669; padding: 1rem; border-radius: 8px;">
                <div style="font-size: 1.1rem; font-weight: 700; color: #065f46;">
                    RAPPORT CERTIFIE
                </div>
                <div style="color: #047857; font-size: 0.9rem; margin-top: 0.3rem;">
                    Toutes les {audit_report.total_numbers} valeurs numeriques citees
                    correspondent aux donnees certifiees.<br>
                    Taux d'exactitude : {audit_report.accuracy_rate:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif status == "WARNING":
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
                        border-left: 4px solid #d97706; padding: 1rem; border-radius: 8px;">
                <div style="font-size: 1.1rem; font-weight: 700; color: #92400e;">
                    AVERTISSEMENT
                </div>
                <div style="color: #b45309; font-size: 0.9rem; margin-top: 0.3rem;">
                    Aucune valeur numerique extraite de la reponse. Impossible de
                    certifier ou de rejeter le contenu.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
                        border-left: 4px solid #dc2626; padding: 1rem; border-radius: 8px;">
                <div style="font-size: 1.1rem; font-weight: 700; color: #991b1b;">
                    RAPPORT NON CERTIFIE
                </div>
                <div style="color: #b91c1c; font-size: 0.9rem; margin-top: 0.3rem;">
                    {audit_report.invented_count} valeur(s) hors whitelist et
                    {n_invalid_quadruplets} quadruplet(s) mal attribue(s).<br>
                    Taux d'exactitude : {audit_report.accuracy_rate:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_verification_table(audit_report):
    if audit_report is None or not audit_report.verifications:
        return

    import pandas as pd
    rows = []
    for v in audit_report.verifications:
        rows.append({
            "Extrait": v.extracted_raw,
            "Valeur": round(v.extracted_value, 4),
            "Statut": "OK" if v.matched else "REJETE",
            "Reference": round(v.matched_reference, 4) if v.matched else "-",
        })
    df = pd.DataFrame(rows)

    def _style_row(row):
        color = "#d4edda" if row["Statut"] == "OK" else "#f8d7da"
        return [f"background-color: {color}"] * len(row)

    st.dataframe(df.style.apply(_style_row, axis=1),
                 use_container_width=True, height=300)


def _render_semantic_badge(semantic):
    """Statut sémantique, affiché séparément du statut numérique."""
    if semantic is None:
        info_banner("Aucun audit semantique realise.", kind="info")
        return
    kind = {"SEMANTIQUE_OK": "success", "SEMANTIQUE_RESERVES": "warning"}.get(
        semantic["status"], "error")
    getattr(st, kind)(
        f"{semantic['status']} : score {semantic['score']:.1f} % "
        f"({semantic['version']}, non bloquant)"
    )
    for check in semantic["checks"]:
        if not check.get("evaluable", True):
            st.caption(f"{check['code']} : {check['label']} (NON EVALUABLE, rapport tronque)")
        elif not check["passed"]:
            st.caption(f"{check['code']} : {check['label']} ({check['detail']})")


def _render_llm_params(params, case_data):
    """Modèle exact, paramètres d'inférence et métadonnées du run audité."""
    if params is None:
        from llm_assistant import get_llm_inference_params
        params = get_llm_inference_params()
    st.markdown(
        f"**Modele** : `{params['model']}`  \n"
        f"**temperature** = {params['temperature']} | **top_p** = {params['top_p']} | "
        f"**top_k** = {params['top_k']}  \n"
        f"**max_output_tokens** = {params['max_output_tokens']} | **seed** = {params['seed']}  \n"
        f"**Directive Zero-Hallucination** : {params['directive_version']}  \n"
        f"**Audit semantique** : {params['semantic_audit_version']}"
    )
    meta = (case_data or {}).get("run_metadata")
    if meta:
        st.caption(
            f"Scenario {meta['scenario']} | topology_seed = {meta['topology_seed']} | "
            f"algorithm_seed = {meta['algorithm_seed']} | num_restarts = {meta['num_restarts']} | "
            f"patience_sweeps = {meta['patience_sweeps']} | max_sweeps = {meta['max_sweeps']} | "
            f"mode {meta['interference_mode']} | code v{meta['code_version']}"
        )


def render():
    """Point d'entree de l'onglet."""
    st.header("Analyse qualitative et audit symbolique")

    record = sm.get("experiment_record")
    if record is None:
        empty_state(
            icon_text="[?]",
            title="Aucun resultat a analyser",
            description="Exécutez d'abord les solveurs dans l'onglet 'Graphe réseau' pour auditer les rèsultats.",
        )
        return

    st.caption(
        "L'assistant LLM redige un rapport d'ingeneire base sur les resultats certifies. "
        "Le module de controle regex valide ensuite de maniere rigoureuse la totalite des affirmations numeriques."
    )

    # Vérification d'initialisation du LLM
    from llm_assistant import is_llm_available, get_initialization_error
    if not is_llm_available():
        info_banner(
            "Le LLM n'est pas initialise. "
            f"Details : {get_initialization_error() or 'inconnue'}",
            kind="warning",
        )
    else:
        from llm_assistant import get_llm_inference_params
        params = get_llm_inference_params()
        st.caption(
            f"Modele LLM : `{params['model']}` | temperature = {params['temperature']} | "
            f"top_p = {params['top_p']} | top_k = {params['top_k']} | "
            f"max_output_tokens = {params['max_output_tokens']} | "
            f"directive {params['directive_version']}"
        )

    available_modes = list(record.solver_results.get("BD-CeNN", {}).keys())
    if not available_modes:
        info_banner("Aucun resultat BD-CeNN disponible.", kind="warning")
        return

    mode = mode_selector(
        session_key="llm_view_mode",
        available_modes=available_modes,
        label="Regime spectral a auditer",
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button(
            "Générer le rapport LLM",
            type="primary",
            use_container_width=True,
            disabled=(not is_llm_available()),
            key="btn_gen_llm",
        ):
            _generate_llm_report(mode)
            st.rerun()

    st.divider()

    response = sm.get(f"llm_response_{mode}")
    audit_report = sm.get(f"audit_report_{mode}")

    if response is None:
        empty_state(
            icon_text="[>]",
            title=f"Aucune analyse effectuee ({MODE_LABELS[mode]})",
            description="Cliquez sur 'Générer le rapport LLM' pour lancer le protocole neuro-symbolique.",
        )
        return

    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        section_title(f"Rapport technique ({MODE_LABELS[mode]})")
        if response.startswith("[ERREUR LLM]") or response.startswith("[LLM_ERROR]"):
            st.error(response)
        else:
            truncation = sm.get(f"llm_truncation_{mode}")
            if truncation and truncation.get("truncated"):
                st.warning("**TRONQUE** : rapport interrompu avant sa fin ("
                           + "; ".join(truncation["reasons"]) + "). Il n'est pas certifie.")
            if sm.get(f"numeric_status_{mode}") == "FAILED":
                st.error("Rapport NON CERTIFIE numeriquement : aucun de ses chiffres ne doit "
                         "etre repris (voir le detail des divergences).")
            st.markdown(response)

    with col_right:
        section_title("Audit numerique (bloquant)")
        struct_report = sm.get(f"audit_struct_{mode}")
        _render_audit_badge(
            audit_report,
            numeric_status=sm.get(f"numeric_status_{mode}"),
            n_invalid_quadruplets=len(struct_report["invalid_claims"]) if struct_report else 0,
            truncation=sm.get(f"llm_truncation_{mode}"),
        )
        section_title("Audit semantique (non bloquant)")
        _render_semantic_badge(sm.get(f"audit_semantic_{mode}"))
        with st.expander("Parametres d'inference LLM et tracabilite", expanded=True):
            _render_llm_params(sm.get(f"llm_params_{mode}"), sm.get(f"llm_case_data_{mode}"))

        if audit_report:
            with st.expander("Voir le detail des verifications"):
                _render_verification_table(audit_report)

            with st.expander("Statistiques d'audit"):
                st.markdown(f"**Nombres analyses** : {audit_report.total_numbers}")
                st.markdown(f"**Assertions exactes** : {audit_report.valid_count}")
                st.markdown(f"**Divergences (hallucinations)** : {audit_report.invented_count}")
                st.markdown(f"**Tolerance appliquee** : {audit_report.tolerance * 100:.1f}%")
                st.markdown(f"**Heure de l'audit** : {audit_report.timestamp}")
        audit_struct = sm.get(f"audit_struct_{mode}")
        if audit_struct is not None and audit_struct.get("total_claims", 0) > 0:
            with st.expander("Verification structurelle (methode, metrique, valeur)"):
                st.markdown(f"**Revendications extraites** : {audit_struct['total_claims']}")
                st.markdown(f"**Valides** : {len(audit_struct['valid_claims'])}")
                st.markdown(f"**Invalides** : {len(audit_struct['invalid_claims'])}")
            if audit_struct["invalid_claims"]:
                for claim in audit_struct["invalid_claims"]:
                    st.warning(
                        f"Association invalide : {claim['method']} / "
                        f"{claim['metric']} / {claim['value']}"
                    )