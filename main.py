import os
import gymnasium as gym
from gymnasium.envs.registration import pprint_registry
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import pickle
import argparse
import csv
from pathlib import Path

from stable_baselines3 import PPO, A2C
from sb3_contrib import TRPO, RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from gymnasium.wrappers import RecordVideo

from src.menu import select_model_saved
from src.metrics import evaluate_model
from src.models import ReinforcementLearningModels

""" 
Este proyecto usa el entorno Ant-v5 para entrenar agentes de RL usando distintos algoritmos.
Se usa también stable-baselines3 para probar algoritmos ya implementados
"""

MODELS = {
    "PPO": PPO,
    "A2C": A2C,
    "TRPO": TRPO,
    "RecurrentPPO": RecurrentPPO,
}

def save_avg_rewards(rewards_csv_path: Path, name_model: str, avg_score: float, n_evaluation_episodes: int):
    """
    se escribe la recompensa promedio del modelo en 'n_evaluation_episodes' episodios en un csv (para luego leer y graficar en plots)
    """
    # si no existe el archivo se debe escribir el header
    should_write_header = not rewards_csv_path.exists()

    with rewards_csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if should_write_header:
            writer.writerow(["model", "avg_reward", "n_episodes"])

        writer.writerow([name_model, avg_score, n_evaluation_episodes])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--load", type=str, help="Cargar o no un modelo: True o False")
    parser.add_argument(
        "--model", type=str, help="Tipo de modelo (PPO, A2C, TRPO o RecurrentPPO)"
    )
    parser.add_argument("--file", type=str, help="Nombre del archivo")

    args = parser.parse_args()

    
    # --- CONFIGURACIÓN ENTORNO ---
    # NOTE: render_mode=None para entrenamiento más rápido (mucho) y sin visualización. render_mode="human" para visualización (para evaluaciones),
    # luego también se van aplicando wrappers útiles
    
    eval_env = gym.make("Ant-v5", render_mode=None)
    eval_env = DummyVecEnv([lambda: eval_env])
    
    if any(vars(args).values()):
        model_saved = True if args.load == "True" else False
        name_model = args.model
        model_file = args.file

    else:
        model_saved, name_model, model_file = select_model_saved()

    if not model_saved:
        # se realiza el entrenamiento
        train_env = gym.make("Ant-v5", render_mode=None)
        train_env = Monitor(train_env, f"logs/{args.model}_monitor.csv") # -> guarda datos de entrenamiento del modelo
        train_env = DummyVecEnv([lambda: train_env]) # -> sb3 requiere entorno vectorizado
        re = ReinforcementLearningModels(
            eval_env=eval_env, train_env=train_env, model=name_model
        )

        model = re.execute_model()
    
    else: 
        
        if os.path.exists(model_file):
            print(f"Cargando modelo {model_file}")
        
            model = MODELS[name_model].load(model_file, env=eval_env, device="cpu")
        else:
            print(
                f"El modelo {model_file} no existe (Verifique que esté guardado este archivo)."
            )
            return
    
    # --- Evaluación del modelo ---

    # TODO: habria que hacer una tablita comparando las recompensas promedio de cada modelo en x episodios
    n_evaluation_episodes = 30
    scores = evaluate_model(model, n_evaluation_episodes, render=False)
    avg_score = (sum(scores) / len(scores)).round(2)

    print(f"Recompensas promedio: {avg_score} ({len(scores)} episodios)")
    
    # se escribe la recompensa promedio en un csv, para luego leerlo en plots y graficar

    rewards_csv_path = Path("logs/avg_rewards.csv")
    rewards_csv_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Guardando recompensas promedio en {rewards_csv_path}")

    save_avg_rewards(rewards_csv_path, name_model, avg_score, n_evaluation_episodes)



if __name__ == "__main__":
    main()
