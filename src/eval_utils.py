import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

def plot_learning_curve(monitor_csv: str, window: int = 20):
    """
    se grafica la curva de aprendizaje de un modelo
    relacionando timesteps acumulados con recompensas
    """
    df = pd.read_csv(monitor_csv, skiprows=1) # -> 1er fila es metadata
    
    # -> timesteps acumulados := suma de las longitudes (columna 'l' del csv)
    df["timesteps"] = df["l"].cumsum()
    
    rewards = df["r"].to_numpy()

    smooth_rewards = (
        np.convolve(rewards, np.ones(window) / window, mode="valid") if len(rewards) >= window else None
    )
    smooth_timesteps = df["timesteps"].to_numpy()[-len(smooth_rewards):]

    # -- PLOT --
    plt.figure(figsize=(8, 4))
    
    # -> grafica recompensa por episodio
    plt.plot(df["timesteps"], df["r"], alpha=0.3, label="Return por episodio") 
    
    # -> grafica de forma suavizada := se aplica media móvil sobre los últimos 'window' episodios, reduciendo ruido de la curva
    # cada punto de la curva es el promedio de 'window' episodios consecutivos
    plt.plot(smooth_timesteps, smooth_rewards, color="C1", label=f"Media móvil (w={window})") 
    
    plt.xlabel("Timesteps acumulados")
    plt.ylabel("Retorno del episodio")
    plt.title("Curva de aprendizaje")
    plt.legend()
    plt.tight_layout()
    plt.savefig("visualizations/learning_curve.png")

    plt.show()

def record_checkpoint(model_file: str, name_prefix: str, video_length: int = 1000):
    """
    se graba un video del checkpoint de un modelo
    """
    model = PPO.load(model_file, device="cpu")
    env = gym.make("Ant-v5", render_mode="rgb_array")
    env = DummyVecEnv([lambda: env])
    env = VecVideoRecorder(env, video_folder="videos", record_video_trigger=lambda step: step == 0, video_length=video_length, name_prefix=name_prefix)
    
    observation = env.reset()
    for _ in range(video_length):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, dones, _ = env.step(action)
        if dones:
            break

    env.close()


def main():

    # la idea es que cada modelo genere un archivo monitor.csv en logs/ que tenga sus datos de entrenamiento
    # asi se construye la curva de aprendizaje para cada uno
    
    # -- Curvas de aprednizake --
    plot_learning_curve("logs/monitor.csv")
    plot_learning_curve("logs/monitor.csv", window=100)
    
    # -- Grabación de checkpoints --
    record_checkpoint("checkpoints/PPO_Ant_0_steps.zip", "PPO_Ant_0_steps")
    record_checkpoint("checkpoints/PPO_Ant_100000_steps.zip", "PPO_Ant_100000_steps")
    record_checkpoint("checkpoints/PPO_Ant_400000_steps.zip", "PPO_Ant_400000_steps")
    record_checkpoint("checkpoints/PPO_Ant_600000_steps.zip", "PPO_Ant_600000_steps")
    record_checkpoint("checkpoints/PPO_Ant_800000_steps.zip", "PPO_Ant_800000_steps")
    record_checkpoint("checkpoints/PPO_Ant_1000000_steps.zip", "PPO_Ant_1000000_steps")


if __name__ == "__main__":
    main()
