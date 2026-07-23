# main.py
import data_generator   # génère et sauvegarde les scénarios
import plots            # génère les figures et table

if __name__ == "__main__":
    # Lance l'expérience complète incluant Random, Greedy et BD-CeNN
    plots.generate_full_experiment_plots()