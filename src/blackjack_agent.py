from collections import defaultdict
import gymnasium as gym
import numpy as np


class BlackjackAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float, # -> ej: epsilon = 1.0: siempre explora al inicio
        epsilon_decay: float, # -> cuánto decae epsilon en el tiempo
        final_epsilon: float, # -> epsilon final (exploración final)
        discount_factor: float = 0.95 # -> cuánto importan los rewards futuros
    ):
        self.env = env
        
        # -> q-table: mapea observación a q-value (reward)
        self.q_values = defaultdict(self._zero_action_values)
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # --- EXPLORACIÓN ---
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        
        # --- TRACKEO PROCESO APRENDIZAJE ---
        self.training_error = []

    def get_action(self, observation: tuple[int, int, bool]) -> int:
        """
        se elige una acción usando estrategia epsilon-greedy

        para blackjack devuelve 0 (quedarse) o 1 (hit)

        """    
        # -> con probabilidad epsilon se explora (acción aleatoria)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # -> con probabilidad 1-epsilon se explota (acción mejor según conocimiento)
        else:
           return int(np.argmax(self.q_values[observation]))

    def update(
        self,
        observation: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_observation: tuple[int, int, bool]
    ):
        """
        se actualiza el conocimiento, es decir, se actualizan q-values según (obs, action, reward, next_obs)
        """
        
        current_q_value = self.q_values[observation][action]

        # -> mejor acción para la siguiente observación
        # es 0 si episodio termina; no hay recompensas futuras
        future_q_value = (not terminated) * np.max(self.q_values[next_observation])

        # -> cálculo de q-value usando ecuación de Bellman 
        # ec. Bellman describe el valor de un estado en función de recompensa inmediata y recompensa futura
        target = reward + self.discount_factor * future_q_value
        
        # -> se calcula el error de la acción
        error = target - current_q_value

        # -> se actualiza el q-value
        self.q_values[observation][action] = (
            current_q_value + self.learning_rate * error
        )

        # -> se trackea proceso de aprendizaje
        self.training_error.append(error)

    def decay_epsilon(self):
        # -> se reduce epsilon a medida que se avanza
        self.epsilon = self.epsilon - self.epsilon_decay
        if self.epsilon < self.final_epsilon:
            self.epsilon = self.final_epsilon
        
    def _zero_action_values(self):
        # -> se establece todos los valores de q-values a 0
        return np.zeros(self.env.action_space.n)


