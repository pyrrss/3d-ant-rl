import argparse

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, A2C
from sb3_contrib import TRPO, RecurrentPPO
from pathlib import Path

from ..metrics import evaluate_model



"""
la idea es hacer evaluaciones a los modelos ya entrenados con el mismo entorno pero con cambios en sus configuraciones, por ejemplo:
    * fricción
    * masa
    * obstáculos
    * ...

con estas pruebas se evalúa la capacidad de generalización y robustez del modelo ante cambios
"""

MODELS = {
    "PPO": PPO,
    "A2C": A2C,
    "TRPO": TRPO,
    "RecurrentPPO": RecurrentPPO,
}

def scale_mass(env: gym.Env, factor: float = 1.0):
    """
    se escalan todas las masas de cuerpos por un factor
    """
    env.unwrapped.model.body_mass[:] *= factor

def scale_damping(env: gym.Env, factor: float = 1.0):
    """
    se escalan los coeficientes de damping

    damping := coeficiente de amortiguación de las articulaciones, actúa como una fricción viscosa en el movimiento
               damping alto hace que la articulación se frene más rápido, damping bajo la deja más suave e inestable
    """
    env.unwrapped.model.dof_damping[:] *= factor

def set_friction(env: gym.Env, slide=None, spin=None, roll=None, multiplier=None):
    """
    se ajusta la fricción de los cuerpos
    """
    geom_friction = env.unwrapped.model.geom_friction

    if multiplier is not None:
        geom_friction[:] = geom_friction * multiplier
    
    if slide is not None:
        geom_friction[:, 0] = slide
    
    if spin is not None:
        geom_friction[:, 1] = spin

    if roll is not None:
        geom_friction[:, 2] = roll




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Nombre de modelo a evaluar")
    parser.add_argument("--file", type=str, default=None, help="Archivo con configuración personalizada del entorno")
    args = parser.parse_args()

    if not args.model:
        raise SystemExit("No se especificó modelo")
    
    # -- Creación del entorno personalizado --
    
    # lo siguiente es necesario porque gym.make busca el .xml relativo a la carpeta de gymnasium, no del repo, asi que se le pasa ruta absoluta
    xml_path = Path(__file__).resolve().parents[2] / "special_envs" / f"{args.file}"
    env = gym.make("Ant-v5", xml_file=str(xml_path), render_mode="rgb_array", camera_name="track")
    
    # NOTE: ahora mismo en los videos que se graban la cámara va más rápido que la hormiga y en un punto la deja atrás.

    # NOTE: para configuraciones como colocar obstáculos, hacerlo desde archivo xml. para configs. como fricción, masa o damping, ajustar desde aquí
    
    # EJEMPLO DE AJUSTES
    # NOTE: no estoy 100% seguro de que funcione bien. revisar/ajustar
    # scale_mass(env, factor=1.2) # 20% mas de masa en cuerpos
    # set_friction(env, multiplier=1.5) # 150% de fricción en cuerpos
    # scale_damping(env, factor=0.1) # 10% de damping en articulaciones (muy sueltos)

    # -- Carga del modelo --
    model = MODELS[args.model].load(f"{args.model}_Ant.zip", device="cpu", env=None)

    # -- Evaluación del modelo --
    print(f"Grabando videos y guardando en videos/special_evaluations/{args.model}")
    scores = evaluate_model(
        model=model,
        name_model=args.model, 
        n_episodes=30,
        env=env, 
        render=False, 
        should_record_video=True
    )

    # recompensa media 
    print(f"Recompensa promedio: {np.mean(scores)}")
    
    # std
    print(f"Desviación estpándar: {np.std(scores)}")




