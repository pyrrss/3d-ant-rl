import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

"""
aquí se podría hacer cualquier evaluación/simulación de un modelo
para obtener métricas y poder visualizarlas, simulaciones con cambios
en la configuración del entorno...
"""

def evaluate_model(model, name_model: str, env: gym.Env = None, n_episodes: int = 5, render: bool = False, should_record_video: bool = False) -> list:
    """
    se realiza la evaluación de un modelo previamente entrenado sobre un entorno dado,
    durante n_episodes y se devuelve la lista de recompensas obtenidas en cada episodio
    """
    if env is None:
        render_mode = "rgb_array" if should_record_video else ("human" if render else None)
        env = gym.make(
            "Ant-v5", render_mode=render_mode
        )
    
    # esto es solo para evaluaciones especiales (con cambios en config. entorno). Para grabar evaluaciones normales se usa record_videos.py
    if should_record_video: 
        os.makedirs(f"videos/special_evaluations/{name_model}", exist_ok=True)
        env = RecordVideo(
                env, 
                video_folder=f"videos/special_evaluations/{name_model}",
                episode_trigger=lambda ep: True, # -> se graban todos
                name_prefix=name_model
        )

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

    # NOTE: al parecer esto es necesario para que entorno se cierre correctamente
    if render and hasattr(env.unwrapped, "viewer") and env.unwrapped.viewer is not None:
        env.unwrapped.viewer.close()


    env.close()
    return scores

