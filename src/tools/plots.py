import os
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


COLOR_MAP = {
    "A2C": "#1f77b4",          # azul
    "TRPO": "#ff7f0e",         # naranjo
    "RecurrentPPO": "#2ca02c", # verde
    "PPO": "#d62728",          # rojo
}

def plot_learning_curve(df: pd.DataFrame, label: str, window: int, ax = None):
    """
    se grafica la curva de aprendizaje de un modelo
    relacionando timesteps acumulados con recompensas

    se grafica de forma cruda := sin suavizado
    se grafica de forma suavizada := se aplica media móvil sobre bloques de 'window' episodios
    """
    
    if ax is None:
        ax = plt.gca()
    
    # se quita notación científica del eje x para mostrar timestamps tal cual (0.1 -> 100000, 0.2 -> 200000, ..., 1.0 -> 1000000)
    ax.ticklabel_format(style="plain", axis="x")

    rewards = df["r"]
    timesteps = df["timesteps"]
    
    # media móvil
    mean = rewards.rolling(window=window, min_periods=window).mean()
    
    std = rewards.rolling(window=window, min_periods=window).std()
    
    # curva suavizada
    ax.plot(timesteps, mean, label=f"{label} (w={window})")
    
    # std
    ax.fill_between(timesteps, mean - std, mean + std, alpha=0.15, label=f"{label} +- std")

def plot_avg_rewards(csv_path: str = "logs/avg_rewards.csv"):
    """
    se grafican las recompensas promedio y desviación estándar de los modelos que hayan sido evaluados (en logs/avg_rewards.csv)
    """
    if not os.path.exists("logs/avg_rewards.csv"):
        raise FileNotFoundError("No se encontró logs/avg_rewards.csv")

    df = pd.read_csv("logs/avg_rewards.csv")
    colors = [COLOR_MAP.get(m, "C0") for m in df["model"]]
    
    # std
    yerr = df["std_reward"]
    yerr_kwargs = {"yerr": yerr, "capsize": 6, "linewidth": 1.5}

    n_evaluation_episodes = df["n_episodes"].max()
    plt.figure(figsize=(9, 5))
    plt.bar(df["model"], df["avg_reward"], color=colors, alpha=0.8, **yerr_kwargs)
    plt.ylabel("Recompensa promedio")
    plt.title(f"Recompensas promedio por modelo en {n_evaluation_episodes} episodios")
    plt.tight_layout()
    

    plt.savefig("visualizations/avg_rewards.png")
    print("Recompensas promedio guardada en visualizations/avg_rewards.png")

    plt.show()

def find_monitors(logs_dir: str, model: str | None) -> dict[str, str]:
    """
    se buscan archivos monitor.csv en directorio logs_dir
    """
    labeled: dict[str, str] = {}

    # si se especifica modelo se carga su monitor.csv
    if model:
        file_path = os.path.join(logs_dir, f"{model}_monitor.csv")
        if os.path.exists(file_path):
            labeled[model] = file_path
        else:
            raise FileNotFoundError(f"No se encontró el archivo {file_path}")
    
    # si no se especifica se cargan todos los monitor.csv que hayan
    else:
        for path in glob.glob(os.path.join(logs_dir, "*_monitor.csv")):
            label = os.path.basename(path).replace("_monitor.csv", "") # -> se extrae el nombre de cada modelo   
            labeled[label] = path

    return labeled


def load_monitor(path: str) -> pd.DataFrame:
    """
    se carga un archivo monitor.csv y se retorna el dataframe
    """
    df = pd.read_csv(path, skiprows=1) # -> 1er fila es metadata
    df["timesteps"] = df["l"].cumsum()
    return df



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Nombre de modelo a graficar; al omitir, se comparan todos los *_monitor.csv")
    parser.add_argument("--logs-dir", type=str, default="logs", help="Directorio base con monitor.csv")
    parser.add_argument("--window", type=int, default=100, help="Longitud de ventana de suavizado (media móvil)")
    parser.add_argument("--out", type=str, default="visualizations/learning_curve.png", help="Ruta para guardar imagen")


    args = parser.parse_args()
    
    # -- LEARNING CURVES --
    monitors = find_monitors(args.logs_dir, args.model)
    if not monitors:
        raise SystemExit("No se encontraron monitor.csv")
    
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, path in monitors.items():
        df = load_monitor(path)
        plot_learning_curve(df, label, args.window, ax=ax)
    
    ax.set_xlabel("Timesteps acumulados")
    ax.set_ylabel("Retorno por episodio")
    ax.set_title(f"Curva de aprendizaje: {args.model}" if args.model else "Curvas de aprendizaje (comparativa)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), framealpha=0.9)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    
    plt.savefig(args.out, bbox_inches="tight")
    print(f"Curva de aprendizaje guardada en {args.out}")
    plt.show()
    
    # -- AVG REWARDS --
    plot_avg_rewards()

    # TODO: agregar más métricas?
            

    
