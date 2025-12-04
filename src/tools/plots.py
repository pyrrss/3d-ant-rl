import os
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def smooth(values: np.ndarray, window: int) -> tuple[np.ndarray, bool]:
    """
    se aplica suavizado mediante media móvil sobre bloques de episodios de longitud 'window'
    """
    if len(values) < window:
        return values, False

    smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
    return smoothed, True


def plot_learning_curve(df: pd.DataFrame, label: str, window: int, ax = None):
    """
    se grafica la curva de aprendizaje de un modelo
    relacionando timesteps acumulados con recompensas

    se grafica de forma cruda := sin suavizado
    se grafica de forma suavizada := se aplica media móvil sobre bloques de 'window' episodios
    """
    
    if ax is None:
        ax = plt.gca()

    rewards = df["r"].to_numpy()
    timesteps = df["timesteps"].to_numpy()


    smoothed, is_smoothed = smooth(rewards, window)
    if is_smoothed:
        timesteps_smoothed = timesteps[-len(smoothed):]
    else:
        timesteps_smoothed = timesteps

    ax.plot(timesteps, rewards, alpha=0.2, label=f"{label} (crudo)")
    ax.plot(timesteps_smoothed, smoothed, label=f"{label} (w={window})")




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
    parser.add_argument("--window", type=int, default=20, help="Longitud de ventana de suavizado (media móvil)")
    parser.add_argument("--out", type=str, default="visualizations/learning_curve.png", help="Ruta para guardar imagen")


    args = parser.parse_args()
    
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
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    
    plt.savefig(args.out)
    print(f"Guardado en {args.out}")
    plt.show()
    

            

    
