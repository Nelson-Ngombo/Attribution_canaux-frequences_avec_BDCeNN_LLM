# llm_assistant.py
import pandas as pd
import requests
import json
import os
from datetime import datetime

# --- 1. Configuration ---
MODEL = "llama3.2:1b"
CSV_FILE = "validation_summary_table.csv"
OUTPUT_DIR = "logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. Fonction pour appeler Ollama ---
def ask_llm(prompt):
    """Envoie une requête à Ollama et retourne la réponse."""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        return f"Erreur lors de l'appel à Ollama : {e}"

# --- 3. Charger les données ---
if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
    print(f"✅ Chargement de {CSV_FILE}")
else:
    try:
        df = pd.read_excel("validation_results.xlsx", sheet_name="Summary")
        print("✅ Chargement de validation_results.xlsx (feuille Summary)")
    except:
        raise FileNotFoundError("Aucun fichier de validation trouvé. Exécute d'abord main.py.")

# Sélectionner un scénario
scenario = "S2"
row = df[df["scenario"] == scenario].iloc[0]

# Extraire les métriques
def get_metric(method, metric):
    return row[f"{method}_{metric}"]

# --- 4. Construire un prompt TRÈS SIMPLE ---
prompt = f"""
Scénario S2 : N=30 cellules, K=5 canaux.

Voici les résultats moyens sur 30 exécutions :

Random :
- Coût global : {get_metric('Random', 'global_cost_mean'):.1f}
- Conflits spectraux : {get_metric('Random', 'spectrum_conflicts_mean'):.1f}
- Temps : {get_metric('Random', 'time_mean'):.6f}s

Greedy :
- Coût global : {get_metric('Greedy', 'global_cost_mean'):.1f}
- Conflits spectraux : {get_metric('Greedy', 'spectrum_conflicts_mean'):.1f}
- Temps : {get_metric('Greedy', 'time_mean'):.6f}s

DSATUR :
- Coût global : {get_metric('DSATUR', 'global_cost_mean'):.1f}
- Conflits spectraux : {get_metric('DSATUR', 'spectrum_conflicts_mean'):.1f}
- Temps : {get_metric('DSATUR', 'time_mean'):.6f}s

BD-CeNN :
- Coût global : {get_metric('BD-CeNN', 'global_cost_mean'):.1f}
- Conflits spectraux : {get_metric('BD-CeNN', 'spectrum_conflicts_mean'):.1f}
- Temps : {get_metric('BD-CeNN', 'time_mean'):.6f}s

Questions (réponds en une courte phrase par question) :
1. Quelle méthode a le plus petit coût global ?
2. Quelle méthode a le moins de conflits spectraux ?
3. Quelle méthode est la plus rapide ?
4. Pour ce scénario, quelle méthode recommandes-tu ?
"""

# --- 5. Appeler le LLM et sauvegarder ---
print(f"🔄 Interrogation de {MODEL} pour le scénario {scenario}...")
reponse = ask_llm(prompt)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(OUTPUT_DIR, f"llm_analysis_{scenario}_{timestamp}.txt")
with open(log_file, "w", encoding="utf-8") as f:
    f.write("="*80 + "\n")
    f.write(f"ANALYSE LLM - SCÉNARIO {scenario}\n")
    f.write(f"Modèle : {MODEL}\n")
    f.write(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*80 + "\n\n")
    f.write("--- PROMPT ---\n")
    f.write(prompt + "\n\n")
    f.write("--- RÉPONSE DU LLM ---\n")
    f.write(reponse + "\n")
    f.write("="*80 + "\n")

print(f"✅ Analyse sauvegardée dans : {log_file}")
print("\n📋 Réponse du LLM :")
print("-"*60)
print(reponse)
print("-"*60)