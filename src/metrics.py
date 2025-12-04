import gymnasium as gym

""" 
Este proyecto usa el entorno Ant-v5 para entrenar agentes de RL usando distintos algoritmos.
Se usa también stable-baselines3 para probar algoritmos ya implementados
"""


def evaluate_model(model, n_episodes: int = 5, render: bool = False) -> list:
    """
    se realiza una evaluación de un modelo PPO previamente entrenado
    """
    env = gym.make(
        "Ant-v5", render_mode="human" if render else None
    )  # -> se crea un nuevo entorno para evaluación
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

    # Important: kill MuJoCo viewer!
    if render and hasattr(env.unwrapped, "viewer") and env.unwrapped.viewer is not None:
        env.unwrapped.viewer.close()

    env.close()
    return scores
