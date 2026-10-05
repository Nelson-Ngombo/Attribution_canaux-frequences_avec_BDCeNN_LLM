# Attribution de Canaux et de Fréquences par BD-CeNN et LLM

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic%20Use-grey.svg)](./LICENSE)


# BD-CeNN + LLM

**Attribution neuro-symbolique de canaux radio sous contraintes d'interference**

Framework complet pour l'optimisation de l'attribution de frequences dans les
reseaux cellulaires, combinant un solveur BD-CeNN (Binary Discretized Cellular
Neural Network) avec un assistant LLM (Google Gemini) audite par un garde-fou
regex independant.

## Fonctionnalites

- **Solveur BD-CeNN multistart** avec mises a jour locales asynchrones
- **Baselines heuristiques** : Random, Greedy (multi-ordres), DSATUR
- **Deux modeles d'interference** :
  - CCI-only (co-canal strict)
  - CCI+ACI (co-canal + fuite de canal adjacent via matrice M)
- **Assistant LLM** en francais avec directive Zero-Hallucination
- **Auditeur regex independant** verifiant chaque nombre cite par le LLM
- **Dashboard Streamlit interactif** avec 7 onglets :
  1. Configuration (scenarios S1-S7, parametres manuels, traduction NL via LLM)
  2. Graphe reseau (visualisation dynamique avec coloration des aretes)
  3. Convergence (courbes forward-filled BD-CeNN)
  4. Comparaison (BD-CeNN vs baselines, tableaux et graphiques)
  5. Rapport LLM (analyse qualitative + badge de certification)
  6. Campagnes E1-E10 (execution batch en tache de fond avec logs live)
  7. Export (CSV, JSON, ZIP)
- **Auto-sauvegarde** dans `outputs/interactive_runs/{cochannel|adjacent}/`
- **10 campagnes d'experiences** reproductibles sur 30 seeds fixes

## Architecture

```text
BD-CeNN_LLM/
|-- main.py                          # Point d'entree Streamlit
|-- config.py                        # Configuration globale
|-- data_generator.py                # Generation topologies
|-- graph_model.py                   # Abstraction reseau (CCI / CCI+ACI)
|-- bdcenn_solver.py                 # Solveur BD-CeNN multistart
|-- baselines.py                     # Random, Greedy, DSATUR
|-- metrics.py                       # Calcul certifie des metriques
|-- experiment_runner.py             # Orchestrateur des simulations
|-- llm_assistant.py                 # Interface Google Gemini + prompts francais
|-- verifier.py                      # Auditeur regex independant
|-- export_manager.py                # Exports CSV/JSON/ZIP
|-- validate_results.py              # valider les metriques obtenues par les methodes
|-- visualize_scenarios.py           # Visualisation statique
|-- experiments.py                   # Campagnes E1-E10
|-- dashboard/                       # Interface Streamlit (7 onglets)
|   |-- theme.py                     # Theme visuel professionnel
|   |-- components.py                # Composants UI reutilisables
|   |-- session_manager.py           # Gestion du session state
|   |-- bootstrap.py                 # Initialisation au demarrage
|   |-- toast.py                     # Notifications
|   |-- graph_renderer.py            # Rendu graphe (Plotly / PyVis)
|   |-- chart_factory.py             # Fabrique de graphiques
|   |-- log_capture.py               # Capture stdout thread-safe
|   |-- experiment_wrappers.py       # Wrappers E1-E10
|   |-- campaign_manager.py          # Gestion des campagnes async
|   |-- interactive_save.py          # Auto-sauvegarde
|   |-- tab_config.py                # Onglet 1
|   |-- tab_graph.py                 # Onglet 2
|   |-- tab_convergence.py           # Onglet 3
|   |-- tab_comparison.py            # Onglet 4
|   |-- tab_llm_report.py            # Onglet 5
|   |-- tab_experiments.py           # Onglet 6 (campagnes)
|   |-- tab_export.py                # Onglet 7
|-- data_structures/                 # Dataclasses (contrats inter-modules)
|   |-- __init__.py                  # Exports centralises
|   |-- topology.py                  # NetworkTopology
|   |-- solver_result.py             # SolverResult
|   |-- experiment_record.py         # ExperimentRecord, MetricsRecord
|   |-- llm_exchange.py              # LLMQuery, LLMResponse
|   |-- audit_report.py              # AuditReport, AuditVerification
|
|-- scenarios/                       # Registre S1-S7
|   |-- __init__.py                  # Exports du package
|   |-- presets.json                 # Definition JSON des scenarios
|   |-- scenario_registry.py         # Chargement et lookup
|
|-- tests/                           # Suite pytest
|   |-- __init__.py                  # Marqueur de package
|   |-- test_baselines.py            # Tests Random / Greedy / DSATUR
|   |-- test_bdcenn_solver.py        # Tests mono-run et multistart
|   |-- test_data_generator.py       # Tests generation de topologies
|   |-- test_graph_model.py          # Tests facade InterferenceGraph
|   |-- test_metrics.py              # Tests metriques cout et conflits
|   |-- test_verifier.py             # Tests verificateur regex
|-- outputs/                         # Sorties interactives (auto)
|-- results/                         # Sorties batch (E1-E10)
|-- requirements.txt
|-- README.md
|-- .env.example
|-- .gitignore
```

## Installation
### 1. Cloner le dépôt
```bash
git clone https://github.com/Nelson-Ngombo/Attribution_canaux-frequences_avec_BDCeNN_LLM.git
cd Attribution_canaux-frequences_avec_BDCeNN_LLM
```

For LLM features, create a .env file at the project root: GOOGLE_API_KEY=your_api_key_here
### 2.  Créer et activer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
.\venv\Scripts\activate    # Windows
```

### 3.  Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4.  Exécution principale

```bash
streamlit run main.py
```
Le dashboard s'ouvre dans votre navigateur (http://localhost:8501).
Au premier demarrage, il genere automatiquement les 30 topologies par
scenario (fichier data/scenarios_data.json).

Aucune autre commande terminale n'est necessaire : tout se fait depuis
l'interface (generation, simulations, campagnes, exports, tests).

## Lancement d'une simulation interactive
### 1.Onglet Configuration : 
choisissez un scenario S1-S7 ou definissez des parametres personnalises (N, K, area, threshold). Vous pouvez aussi decrire le reseau en francais et laisser le LLM extraire les
parametres.
Cliquez sur Generer la topologie.

### 2.Onglet Convergence : 
cliquez sur Lancer la simulation. BD-CeNN et les trois baselines s'executent sous les modes d'interference actifs. Les resultats sont automatiquement sauvegardes dans
outputs/interactive_runs/{cochannel|adjacent}/.
Visualisez le graphe colore (onglet Graphe reseau), comparez les
solveurs (onglet Comparaison), generez un rapport LLM (onglet
Rapport LLM), ou exportez les resultats (onglet Export).
Lancement d'une campagne d'experiences
### 3. Onglet Campagnes E1-E10 : 
cochez les experiences a lancer (E1 a E10) Cliquez sur Lancer la campagne selectionnee Le suivi live affiche les logs en temps reel et l'avancement Les resultats sont sauvegardes dans results/csv/E{n}/ et
results/figures/E{n}/ avec separation stricte cochannel / adjacent
## Author
Nelson N. – 2e ICE/EN

Encadreur principal : Prof. Kyandoghere Kyamakya

Co-encadreur: Ass. Ir. Bisuta, Ir.Gédeon Nkishi, Ir.Exaucé Maruba

Laboratoire d'attache:Bastion-Lab

## Licence : 
Ce projet est réalisé dans le cadre d'un travail de mémoire universitaire. Le code source est fourni à des fins de recherche et d'évaluation académique.
Toute réutilisation, modification ou citation de ce travail doit impérativement mentionner l'auteur et les superviseurs académiques cités ci-dessus.
Pour toute question ou collaboration, veuillez contacter l'auteur via le dépôt GitHub.