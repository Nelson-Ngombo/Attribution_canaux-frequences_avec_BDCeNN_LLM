# main.py
import data_generator
import validation
import plots
import experiments
import os
import config

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" PIPELINE COMPLET - MÉMOIRE BD-CeNN + LLM")
    print("="*80)
    
    # 1. Génération des scénarios (déjà faite par data_generator à l'import)
    print("\n✅ Scénarios chargés.")
    
    # 2. Lancer la validation si le fichier Excel n'existe pas
    if not os.path.exists(config.VALIDATION_EXCEL_FILE):
        print("🔁 Lancement de la validation (30 runs)...")
        validation.run_validation()
    else:
        print("✅ Fichier validation_results.xlsx trouvé.")
    
    # 3. Générer le rapport final (plots)
    plots.generate_full_experiment_plots()
    
    # 4. Lancer les expériences spécifiques (E1, E3, E4, E6, E7, E8, E9)
    experiments.run_all_experiments()
    
    print("\n" + "="*80)
    print("🏁 PIPELINE COMPLET TERMINÉ.")
    print("="*80)