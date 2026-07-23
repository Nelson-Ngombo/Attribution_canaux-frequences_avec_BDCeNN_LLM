#  Attribution de canaux / fréquences avec BD-CeNN + LLM

##  Description du projet

Ce projet de mémoire (2e ICE/EN) propose un cadre complet pour l’attribution de canaux/fréquences dans les réseaux radio, en combinant :

- Un **solveur d’optimisation discret BD-CeNN** (Binary Discrete Cellular Neural Network) pour minimiser les interférences.
- Un **assistant LLM** (via Ollama) pour générer des rapports et interpréter les résultats.

Le problème est modélisé comme une **coloration de graphe pondéré**, où les cellules radio sont des sommets, les interférences des arêtes, et les canaux des couleurs. Une **matrice d’interférence entre canaux** (M) est utilisée pour pénaliser à la fois les conflits co-canal et les interférences entre canaux adjacents.

##  Architecture du projet
BD_CeNN_LLM/
│
├── config.py                      # Configuration centrale (chemins, seeds, paramètres globaux)
├── data_generator.py              # Génération des 50 scénarios (S1 à S50)
├── visualize_scenarios.py         # Visualisation des graphes et matrices W (exploration)
├── baselines.py                   # Random, Greedy, DSATUR (avec matrice M)
├── bdcenn_solver.py               # Solveur BD-CeNN
├── metrics.py                     # Métriques : coût, conflits, canaux, conflits spectraux
├── validation.py                  # Campagne de validation (30 runs × 50 scénarios)
├── plots.py                       # Génération du rapport final (figures, tables, heatmap)
├── llm_assistant.py               # Assistant LLM (analyse d’un scénario via Ollama)
├── main.py                        # Point d’entrée (orchestre tout)
├── requirements.txt               # Dépendances Python
├── README.md                      # Documentation du projet
│
├── data/                          # Scénarios générés (JSON)
│   └── scenarios_data.json
│
└── results/                       # Tous les résultats produits
    ├── excel/
    │   └── validation_results.xlsx          # Résultats bruts + résumé (30 runs)
    ├── csv/
    │   ├── comparison_full_table.csv        # Résumé des métriques
    │   └── validation_summary_table.csv     # Moyennes et écarts-types
    ├── figures/
    │   ├── comparison_global_cost.png
    │   ├── comparison_spectrum_conflicts.png
    │   ├── comparison_used_channels.png
    │   ├── comparison_time.png
    │   ├── comparison_ranking.png
    │   ├── convergence_S1.png
    │   ├── convergence_S2.png
    │   └── ... (S50)
    ├── logs/                                # Logs d’exécution système
    │   └── experiment_log_*.txt
    └── llm_logs/                            # Logs des analyses LLM
        └── llm_analysis_*.txt



## Installation et dépendances

### 1. Cloner le dépôt

git clone https://github.com/Nelson-Ngombo/Attribution_canaux-frequences_avec_BDCeNN_LLM.git
cd BD_CeNN_LLM

### 2.  Créer et activer un environnement virtuel
python -m venv venv
source venv/bin/activate   # Linux/Mac
.\venv\Scripts\activate    # Windows


### 3.  Installer les dépendances

pip install -r requirements.txt


### 4.  Exécution principale

python data_generator.py

python visualize_scenarios.py

python main.py

## Auteur
Nelson N. – 2e ICE/EN
Encadrement : Prof. Kyandoghere Kyamakya, Ass. Ir. Bisuta, Ir.Gédeon Nkishi, Ir.Exaucé Maruba
Bastion-Lab

## Licence : 
Ce projet est réalisé dans le cadre d’un mémoire universitaire. Toute réutilisation doit citer l’auteur et le superviseur.