import gymnasium as gym
from gymnasium.envs.registration import pprint_registry
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import pickle

from blackjack_agent import BlackjackAgent
from test_agent import test_agent

""" 
EN ESTE PROYECTO SE CREA UN AGENTE QUE USA MODELO Q-LEARNING TABULAR
PARA JUGAR BLACKJACK
"""

pprint_registry()

# --- CONFIGURACIÓN ENTORNO ---
# NOTE: render_mode=None para entrenamiento más rápido (mucho) y sin visualización. render_mode="human" para visualización
env = gym.make("Blackjack-v1", render_mode=None)
n_episodes = 100_000

# -> se aplica wrapper para guardar estadísticas de episodio
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

# --- SETUP AGENTE ---
learning_rate = 0.01
initial_epsilon = 1.0
epsilon_decay = initial_epsilon / (n_episodes / 2)
final_epsilon = 0.1

# -> se crea un nuevo agente desde 0
agent = BlackjackAgent(
    env,
    learning_rate,
    initial_epsilon,
    epsilon_decay,
    final_epsilon
)

# --- CARGADO DEL AGENTE PREVIAMMENTE ENTRENADO ---
# with open("blackjack_agent.pkl", "rb") as f:
#     agent = pickle.load(f)
#     print("Cargando agente previamente entrenado")

# print(f"Agente cargado: {type(agent)}")

# --- TRAINING LOOP ---
for episode in tqdm(range(n_episodes)):
    observation, info = env.reset()
    done = False

    while not done:
        action = agent.get_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        
        agent.update(observation, action, float(reward), terminated, next_observation)

        done = terminated or truncated
        observation = next_observation
    
    agent.decay_epsilon()

env.close()


# --- VISUALIZACIÓN DEL PROGRESO ---

def get_moving_avgs(arr, window, convolution_mode):
    return np.convolve(
            np.array(arr).flatten(),
            np.ones(window),
            mode=convolution_mode
            ) / window

rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(15, 5))

axs[0].set_title("Recompensa por Episodio")
reward_moving_average = get_moving_avgs(
        env.return_queue,
        rolling_length,
        "valid"
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[0].set_ylabel("Recompensa Promedio")
axs[0].set_xlabel("Episodio")

axs[1].set_title("Acciones por Episodio")
length_moving_average = get_moving_avgs(
    env.length_queue,
    rolling_length,
    "valid"
)
axs[1].plot(range(len(length_moving_average)), length_moving_average) 
axs[1].set_ylabel("Duración Promedio")
axs[1].set_xlabel("Episodio")

axs[2].set_title("Error de Entrenamiento")
training_error_moving_average = get_moving_avgs(
        agent.training_error,
        rolling_length,
        "same"
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
axs[2].set_ylabel("Error de Diferencia Temporal")
axs[2].set_xlabel("Paso de Entrenamiento (Step)")

plt.tight_layout()
plt.savefig("rl_progress.png")
print("\nGráficos de progreso guardados en 'rl_progress.png'")


# --- TESTEO DEL AGENTE CON CONOCIMIENTO ADQUIRIDO ---
test_agent(agent, env)


# --- GUARDADO DEL AGENTE CON CONOCIMIENTO ADQUIRIDO ---
# NOTE: se está guardando el agente completo (el objeto) en un archivo
# otra forma es simplemente guardar el 'conocimiento' (q-table)

print("\n blackjack_agent.pkl guardado")
with open("blackjack_agent.pkl", "wb") as f:
    pickle.dump(agent, f)





