"""
Tab 1: Scenario Configuration and LLM-based Natural Language Translation.
Refined version with:
    - French prompt for LLM translation
    - Strict JSON validation
    - Human-in-the-loop parameter review before applying
    - Automatic topology generation and immediate visualization
"""

import json
import re
import streamlit as st
import numpy as np

import config
from scenarios import list_scenarios, get_scenario, get_scenario_description
from experiment_runner import build_topology
from dashboard import session_manager as sm
from dashboard import toast
from dashboard.chart_factory import create_weight_matrix_heatmap
from dashboard.components import (
    section_title,
    kpi_row,
    info_banner,
    render_scenario_summary,
    divider_with_label,
    empty_state,
)


# ---------------------------------------------------------------------------
# LLM-based NL to parameters translation (in French)
# ---------------------------------------------------------------------------

TRANSLATION_SYSTEM_PROMPT = """
Tu es un assistant specialise en configuration de reseau radio cellulaire.
Ton unique role est d'extraire les parametres numeriques suivants a partir
d'une description en langage naturel :

- N : nombre de cellules (antennes) - entier entre 5 et 200
- K : nombre de canaux (frequences) disponibles - entier entre 2 et 20
- area : taille de la zone geographique carree - entier entre 50 et 500
- threshold : seuil de distance d'interference - entier entre 10 et 150

REGLES STRICTES :
1. Reponds UNIQUEMENT avec un objet JSON valide, sans aucun texte avant ou apres.
2. Le format doit etre exactement : {"N": <int>, "K": <int>, "area": <int>, "threshold": <int>}
3. Si un parametre n'est pas explicitement present dans la description, utilise ces valeurs par defaut :
   N=30, K=4, area=150, threshold=35
4. Interprete les qualificatifs :
   - "petit reseau" -> N ~= 10-20
   - "reseau moyen" -> N ~= 30-50
   - "grand reseau" -> N ~= 80-150
   - "tres grand" -> N ~= 150-200
   - "dense" -> threshold eleve (60-100)
   - "peu dense" ou "espace" -> threshold faible (20-35)
   - "urbain" -> area petit (100-150)
   - "rural" -> area grand (250-400)
5. Assure-toi que les valeurs restent dans les bornes autorisees.

Description a analyser :
"""


def _translate_nl_to_params(description: str) -> dict:
    """
    Calls the LLM to extract structured parameters from a natural language description.
    Returns dict with keys N, K, area, threshold, plus 'source' ('llm' or 'fallback')
    and 'raw_response' for transparency.
    """
    result = {
        "N": 30, "K": 4, "area": 150, "threshold": 35,
        "source": "fallback",
        "raw_response": None,
        "error": None,
    }

    try:
        from llm_assistant import ask_llm, is_llm_available

        if not is_llm_available():
            result["error"] = (
                "LLM non initialise. Utilisation de l'extraction heuristique par defaut."
            )
            _heuristic_extraction(description, result)
            return result

        full_prompt = TRANSLATION_SYSTEM_PROMPT + "\n" + description.strip()
        response = ask_llm(full_prompt, inter_request_delay=0.5)
        result["raw_response"] = response

        if response.startswith("[ERREUR LLM]"):
            result["error"] = response
            _heuristic_extraction(description, result)
            return result

        # Extract JSON from response
        json_match = re.search(r"\{[^{}]*\}", response)
        if not json_match:
            result["error"] = "Aucun JSON trouve dans la reponse du LLM."
            _heuristic_extraction(description, result)
            return result

        parsed = json.loads(json_match.group(0))

        # Validate and clamp
        result["N"] = int(np.clip(parsed.get("N", 30), 5, 200))
        result["K"] = int(np.clip(parsed.get("K", 4), 2, 20))
        result["area"] = int(np.clip(parsed.get("area", 150), 50, 500))
        result["threshold"] = int(np.clip(parsed.get("threshold", 35), 10, 150))
        result["source"] = "llm"

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        _heuristic_extraction(description, result)

    return result


def _heuristic_extraction(description: str, result: dict):
    """Fallback heuristic that extracts numbers from text."""
    numbers = [int(n) for n in re.findall(r"\d+", description)]
    if len(numbers) >= 1:
        result["N"] = int(np.clip(numbers[0], 5, 200))
    if len(numbers) >= 2:
        result["K"] = int(np.clip(numbers[1], 2, 20))
    if len(numbers) >= 3:
        result["area"] = int(np.clip(numbers[2], 50, 500))
    if len(numbers) >= 4:
        result["threshold"] = int(np.clip(numbers[3], 10, 150))


# ---------------------------------------------------------------------------
# UI panels
# ---------------------------------------------------------------------------

def _render_scenario_selector():
    """Preset scenario dropdown with description panel."""
    st.markdown("#### Choisir un scenario predefini (S1 a S7)")
    scenarios = list_scenarios()

    selected = st.selectbox(
        "Scenario",
        options=scenarios,
        index=scenarios.index(sm.get("selected_scenario", "S3"))
              if sm.get("selected_scenario") in scenarios else 0,
        key="scenario_selectbox",
        label_visibility="collapsed",
    )
    sm.set_value("selected_scenario", selected)

    params = get_scenario(selected)
    if params:
        st.info(f"**{selected}** - {get_scenario_description(selected)}")
        kpi_row([
            {"label": "N (cellules)", "value": str(params["N"])},
            {"label": "K (canaux)", "value": str(params["K"])},
            {"label": "Zone (area)", "value": str(params["area"])},
            {"label": "Threshold", "value": str(params["threshold"])},
        ])


def _render_custom_params():
    """Sliders for manual parameter definition."""
    st.markdown("#### Parametres personnalises")

    col1, col2 = st.columns(2)
    with col1:
        n = st.slider("Nombre de cellules N", 5, 200,
                      sm.get("custom_N", 30), step=5, key="slider_N")
        k = st.slider("Nombre de canaux K", 2, 20,
                      sm.get("custom_K", 4), key="slider_K")
        area = st.slider("Taille de la zone (area)", 50, 500,
                         sm.get("custom_area", 150), step=10, key="slider_area")
    with col2:
        threshold = st.slider("Seuil d'interference (threshold)", 10, 150,
                              sm.get("custom_threshold", 35), key="slider_thr")
        seed = st.number_input("Seed", 1, 10000,
                               sm.get("custom_seed", 1), key="input_seed")

    sm.set_value("custom_N", n)
    sm.set_value("custom_K", k)
    sm.set_value("custom_area", area)
    sm.set_value("custom_threshold", threshold)
    sm.set_value("custom_seed", int(seed))


def _render_nl_translation():
    """Natural language description translation via LLM."""
    st.markdown("#### Description en langage naturel")

    st.caption(
        "Decrivez le reseau que vous souhaitez tester en francais. Le LLM extraira "
        "les parametres (N, K, area, threshold). Vous pourrez ensuite les ajuster "
        "manuellement avant de lancer la simulation."
    )

    example_placeholder = (
        "Exemple : Un reseau urbain dense de 40 antennes avec 5 frequences "
        "disponibles, deploye sur une petite zone de 120 metres avec un rayon "
        "d'interference de 45 metres."
    )

    nl_text = st.text_area(
        "Description",
        value=sm.get("nl_input_text", ""),
        placeholder=example_placeholder,
        height=120,
        key="nl_text_area",
        label_visibility="collapsed",
    )
    sm.set_value("nl_input_text", nl_text)

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        translate_clicked = st.button(
            "Traduire via LLM",
            type="primary",
            use_container_width=True,
            key="btn_translate",
            disabled=(not nl_text.strip()),
        )
    with col2:
        clear_clicked = st.button(
            "Effacer",
            use_container_width=True,
            key="btn_clear_nl",
        )
        if clear_clicked:
            sm.set_value("nl_input_text", "")
            sm.set_value("nl_translated_params", None)
            st.rerun()

    if translate_clicked and nl_text.strip():
        with st.spinner("Interrogation du LLM en cours..."):
            translated = _translate_nl_to_params(nl_text)
        sm.set_value("nl_translated_params", translated)

    # Display translation result
    translated = sm.get("nl_translated_params")
    if translated:
        st.markdown("---")

        if translated["source"] == "llm":
            st.success("Parametres extraits par le LLM")
        else:
            st.warning(
                f"Extraction heuristique utilisee (LLM indisponible ou erreur). "
                f"Raison : {translated.get('error', 'inconnue')}"
            )

        params_cols = st.columns(4)
        params_cols[0].metric("N", translated["N"])
        params_cols[1].metric("K", translated["K"])
        params_cols[2].metric("area", translated["area"])
        params_cols[3].metric("threshold", translated["threshold"])

        st.caption(
            "Ces valeurs seront pre-remplies dans les parametres personnalises. "
            "Vous pouvez les ajuster manuellement dans les sliders ci-dessus "
            "avant de generer la topologie."
        )

        if st.button(
            "Appliquer aux parametres personnalises",
            type="secondary",
            use_container_width=True,
            key="btn_apply_nl",
        ):
            sm.set_value("custom_N", translated["N"])
            sm.set_value("custom_K", translated["K"])
            sm.set_value("custom_area", translated["area"])
            sm.set_value("custom_threshold", translated["threshold"])
            sm.set_value("use_custom_params", True)
            toast.success("Parametres appliques. Verifiez les sliders puis generez la topologie.")
            st.rerun()

        if translated.get("raw_response"):
            with st.expander("Reponse brute du LLM"):
                st.code(translated["raw_response"], language="json")


def _render_bdcenn_hyperparams():
    """BD-CeNN hyperparameter controls."""
    st.markdown("#### Hyperparametres BD-CeNN")

    col1, col2 = st.columns(2)
    with col1:
        nr = st.slider(
            "Nombre de redemarrages (multistart)",
            1, 30,
            sm.get("num_restarts", 10),
            key="slider_restarts",
            help="Plus de redemarrages ameliorent la robustesse aux minima locaux mais augmentent le temps."
        )
    with col2:
        mi = st.slider(
            "Iterations max par redemarrage",
            10, 200,
            sm.get("max_iter", 50),
            step=10,
            key="slider_max_iter",
            help="Nombre maximal de balayages du reseau par redemarrage."
        )

    if nr != sm.get("num_restarts") or mi != sm.get("max_iter"):
        sm.set_value("num_restarts", nr)
        sm.set_value("max_iter", mi)
        sm.refresh_runner()


def _render_mode_toggles():
    """Toggles to enable/disable interference mode execution."""
    st.markdown("#### Modes d'interference a evaluer")

    col1, col2 = st.columns(2)
    with col1:
        run_cci = st.checkbox(
            "CCI-only (co-canal uniquement)",
            value=sm.get("run_mode_cci", True),
            key="chk_cci",
            help="Modele strict : conflit uniquement si les cellules voisines utilisent le meme canal."
        )
    with col2:
        run_aci = st.checkbox(
            "CCI+ACI (avec fuite adjacente)",
            value=sm.get("run_mode_cci_aci", True),
            key="chk_aci",
            help="Modele etendu : ajoute la penalite pour les canaux contigus via la matrice M."
        )

    sm.set_value("run_mode_cci", run_cci)
    sm.set_value("run_mode_cci_aci", run_aci)


def _generate_topology_action():
    """Callback that generates the topology and updates state."""
    if sm.get("use_custom_params"):
        N = sm.get("custom_N")
        K = sm.get("custom_K")
        area = sm.get("custom_area")
        threshold = sm.get("custom_threshold")
        scenario_name = "CUSTOM"
    else:
        params = get_scenario(sm.get("selected_scenario"))
        N = params["N"]
        K = params["K"]
        area = params["area"]
        threshold = params["threshold"]
        scenario_name = sm.get("selected_scenario")

    seed = sm.get("custom_seed")

    try:
        topology = build_topology(N, K, area, threshold, seed,
                                   scenario_name=scenario_name)
        sm.set_value("topology", topology)
        sm.reset_experiment_state()
        toast.success(f"Topologie generee : {scenario_name} (N={N}, K={K}, seed={seed})")
        return True
    except Exception as e:
        toast.error(f"Erreur de generation : {type(e).__name__}: {e}")
        return False


# ---------------------------------------------------------------------------
# MAIN RENDERER
# ---------------------------------------------------------------------------

def render():
    """Main tab entrypoint."""
    st.header("Configuration du scenario")

    st.caption(
        "Configurez la topologie du reseau a evaluer. Trois methodes disponibles : "
        "scenario predefini, parametres manuels, ou description en langage naturel "
        "traduite par le LLM."
    )

    # Toggle: preset vs custom
    use_custom = st.toggle(
        "Utiliser des parametres personnalises",
        value=sm.get("use_custom_params", False),
        key="toggle_custom",
    )
    sm.set_value("use_custom_params", use_custom)

    st.divider()

    # Layout: 2 columns
    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        section_title("Definition du reseau")

        if use_custom:
            _render_custom_params()
        else:
            _render_scenario_selector()

        divider_with_label("Assistance LLM")
        _render_nl_translation()

        divider_with_label("Parametres du solveur")
        _render_bdcenn_hyperparams()
        _render_mode_toggles()

        st.divider()

        # Launch button
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if st.button(
                "Generer la topologie",
                type="primary",
                use_container_width=True,
                key="btn_generate_topo",
            ):
                if _generate_topology_action():
                    st.rerun()

        with col_btn2:
            if sm.get("topology") is not None:
                st.caption(
                    "Topologie prete. Passez a l'onglet **Convergence** pour "
                    "lancer les solveurs."
                )

    with col_right:
        section_title("Topologie actuelle")
        topology = sm.get("topology")

        if topology is None:
            empty_state(
                icon_text="[]",
                title="Aucune topologie",
                description=(
                    "Configurez les parametres a gauche puis cliquez sur "
                    "'Generer la topologie'."
                ),
            )
        else:
            render_scenario_summary(topology)

            st.markdown("---")

            with st.expander("Matrice d'interference W", expanded=False):
                fig = create_weight_matrix_heatmap(topology.weight_matrix)
                fig.update_layout(transition_duration=300)
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"heatmap_{topology.scenario_name}_{topology.seed}",
                )

            st.markdown("---")
            st.caption(
                f"**Densite** : {topology.density:.3f}  |  "
                f"**Aretes actives** : {topology.n_edges}  |  "
                f"**Aretes possibles** : {topology.N * (topology.N - 1) // 2}"
            )