# Attribution de Canaux et de Fréquences par BD-CeNN et LLM

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic%20Use-grey.svg)](./LICENSE)


## Description du projet

Ce projet, réalisé dans le cadre d'un mémoire de fin d'études (2e cycle ICE/EN), propose un cadre complet pour l'attribution dynamique de canaux et de fréquences dans les réseaux radio. L'approche hybride combine :

- Un **solveur d'optimisation discret BD-CeNN** (Binary Discrete Cellular Neural Network) dédié à la minimisation des interférences.
- Un **assistant basé sur un modèle de langage (LLM)**  pour l'analyse contextuelle des résultats et la génération de rapports interprétatifs.

Le problème d'attribution est modélisé comme une **coloration de graphe pondéré**. Les cellules radio sont représentées par des sommets, les interférences par des arêtes, et les canaux par des couleurs. Une **matrice d'interférence entre canaux (M)** est intégrée pour pénaliser simultanément les conflits co-canal et les interférences entre canaux adjacents.

## Fonctionnalités principales

- Génération automatisée des scénarios de réseaux radio.
- Implémentation et comparaison d'algorithmes de référence (Random, Greedy, DSATUR) adaptés à la matrice d'interférence.
- Résolution optimisée via l'architecture BD-CeNN.
- Campagne de validation statistique robuste (30 exécutions par scénario).
- Analyse post-traitement et génération de rapports automatisés via un LLM.
- Visualisation complète des métriques (coût global, conflits spectraux, canaux utilisés, temps de calcul).

## Architecture du projet

```text
BD_CeNN_LLM/
│
├── config.py                      # Configuration centrale (chemins, seeds, paramètres globaux)
├── data_generator.py              # Génération des scénarios
├── visualize_scenarios.py         # Visualisation des graphes et matrices d'interférence (W)
├── baselines.py                   # Implémentation des heuristiques : Random, Greedy, DSATUR
├── bdcenn_solver.py               # Cœur du solveur BD-CeNN
├── metrics.py                     # Calcul des métriques : coût, conflits, canaux utilisés
├── validation.py                  # Orchestration de la campagne de validation (30 runs x 50 scénarios)
├── plots.py                       # Génération des figures, tableaux et heatmaps du rapport final
├── llm_assistant.py               # Interface avec le LLM  pour l'analyse des scénarios
├── main.py                        # Point d'entrée principal (orchestration du pipeline complet)
├── requirements.txt               # Dépendances Python requises
├── README.md                      # Documentation du projet
│
├── data/                          # Données d'entrée générées
│   └── scenarios_data.json        # Fichier JSON contenant les scénarios
│
└── results/                       # Répertoire de sortie (généré à l'exécution)
    ├── excel/
    │   └── validation_results.xlsx          # Résultats bruts et résumé statistique
    ├── csv/
    │   ├── comparison_full_table.csv        # Tableau comparatif complet des métriques
    │   └── validation_summary_table.csv     # Moyennes et écarts-types par algorithme
    ├── figures/                             # Visualisations graphiques (comparaisons, convergences)
    ├── logs/                                # Journaux d'exécution système
    └── llm_logs/                            # Journaux des analyses générées par le LLM
```

## Prérequis et installation

### 1. Cloner le dépôt
```bash
git clone https://github.com/Nelson-Ngombo/Attribution_canaux-frequences_avec_BDCeNN_LLM.git
cd Attribution_canaux-frequences_avec_BDCeNN_LLM
```

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
Le pipeline est conçu pour être exécuté de manière séquentielle. Exécutez les commandes suivantes dans l'ordre :
```bash
python data_generator.py

python visualize_scenarios.py

python main.py
```
## Auteur
Nelson N. – 2e ICE/EN

Encadreur principal : Prof. Kyandoghere Kyamakya

Co-encadreur: Ass. Ir. Bisuta, Ir.Gédeon Nkishi, Ir.Exaucé Maruba

Laboratoire d'attache:Bastion-Lab

## Licence : 
Ce projet est réalisé dans le cadre d'un travail de mémoire universitaire. Le code source est fourni à des fins de recherche et d'évaluation académique.
Toute réutilisation, modification ou citation de ce travail doit impérativement mentionner l'auteur et les superviseurs académiques cités ci-dessus.
Pour toute question ou collaboration, veuillez contacter l'auteur via le dépôt GitHub.