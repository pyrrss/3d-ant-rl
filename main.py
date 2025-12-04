import os
import gymnasium as gym
from gymnasium.envs.registration import pprint_registry
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import pickle
import argparse

from stable_baselines3 import PPO, A2C
from sb3_contrib import TRPO, RecurrentPPO

from stable_baselines3.common.vec_env import DummyVecEnv

from src.menu import select_model_saved
from src.metrics import evaluate_model
from src.models import ReinforcementLearningModels


MODELS = {
    "PPO": PPO,
    "A2C": A2C,
    "TRPO": TRPO,
    "RecurrentPPO": RecurrentPPO,
}


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--load", type=str, help="Cargar o no un modelo: True o False")
    parser.add_argument(
        "--model", type=str, help="Tipo de modelo (PPO, A2C, TRPO o RecurrentPPO)"
    )
    parser.add_argument("--file", type=str, help="Nombre del archivo")

    args = parser.parse_args()

    # --- CONFIGURACIÓN ENTORNO ---
    # NOTE: render_mode=None para entrenamiento más rápido (mucho) y sin visualización. render_mode="human" para visualización (para evaluaciones)

    train_env = DummyVecEnv([lambda: gym.make("Ant-v5", render_mode=None)])
    eval_env = DummyVecEnv([lambda: gym.make("Ant-v5", render_mode=None)])

    if any(vars(args).values()):
        model_saved = True if args.load == "True" else False
        name_model = args.model
        model_file = args.file

    else:
        model_saved, name_model, model_file = select_model_saved()

    if not model_saved:
        re = ReinforcementLearningModels(
            eval_env=eval_env, train_env=train_env, model=name_model
        )
        model = re.execute_model()

    else:
        if os.path.exists(model_file):
            print(f"Cargando modelo {model_file}")
            model = MODELS[name_model].load(model_file, env=train_env, device="cpu")
        else:
            print(
                f"El modelo {model_file} no existe (Verifique que esté guardado este archivo)."
            )
            return

    scores = evaluate_model(model, 5, render=True)
    print(
        f"Recompensas promedio: {sum(scores)/len(scores):.1f} ({len(scores)} episodios)"
    )


if __name__ == "__main__":
    main()
