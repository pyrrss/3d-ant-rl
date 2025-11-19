import os
import gymnasium as gym
from gymnasium.envs.registration import pprint_registry
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import pickle

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback


""" 
Este proyecto usa el entorno Ant-v5 para entrenar agentes de RL usando distintos algoritmos.
Se usa también stable-baselines3 para probar algoritmos ya implementados
"""

def evaluate_model(model: PPO, n_episodes: int = 100, render: bool = False) -> list:
    """
    se realiza una evaluación de un modelo PPO previamente entrenado
    """
    env = gym.make("Ant-v5", render_mode="human" if render else None) # -> se crea un nuevo entorno para evaluación
    scores = []
    
    """
    el ciclo de aprendizaje, a nivel general, es el siguiente:
        1. agente recibe una observación del entorno (estado actual)
        2. agente elige una acción basado en la observación y su política
        3. entorno responde a la acción con un nuevo estado y recompensa
        4. repetir hasta que se termina
    """
    for _ in range(n_episodes):
        observation, info = env.reset()
        done = False
        episode_score = 0.0
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_score += reward
            observation = next_observation
        
        scores.append(episode_score)

    env.close()
    return scores


def main():
    model_file = "PPO_Ant.zip"
    
    # --- CONFIGURACIÓN ENTORNO ---
    # NOTE: render_mode=None para entrenamiento más rápido (mucho) y sin visualización. render_mode="human" para visualización (para evaluaciones)
    env = gym.make("Ant-v5", render_mode=None)
    n_episodes = 100_00

    train_env = DummyVecEnv([lambda: env]) # -> sb3 requiere entorno vectorizado
    eval_env = DummyVecEnv([lambda: env])
    
    # --- Configuración modelo PPO ---
    if os.path.exists(model_file):
        print(f"Cargando modelo {model_file}")
        model = PPO.load(model_file, env=train_env, device="cpu")
    else:
        print(f"No se encontró modelo, se creará uno nuevo")

        eval_callback = EvalCallback( # -> cada cierta cantidad de pasos ocurre una evaluación
            eval_env,
            best_model_save_path="logs",
            log_path="logs",
            eval_freq=10_000, # -> cada 10k pasos se evalúa          
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        )

        # NOTE: por ahora se usan hiperparámetros genéricos
        model = PPO(
            "MlpPolicy",
            train_env,
            device="cpu", # -> por ahora se usa CPU para entrenar el modelo (cuda me daba problemas)
            verbose=1,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            learning_rate=3e-4,
            tensorboard_log="logs"
        )

        # --- Entrenamiento ---
        model.learn(total_timesteps=1_000_000, callback=eval_callback)

        # --- Guardado del modelo ---
        model.save("PPO_Ant")
        print(f"Modelo guardado en {model_file}")

    # --- Evaluación del modelo ---
    scores = evaluate_model(model, n_episodes, render=True)
    
    print(f"Recompensas promedio: {sum(scores)/len(scores):.1f} ({len(scores)} episodios)")

if __name__ == "__main__":
    main()
