from stable_baselines3 import PPO, A2C
from sb3_contrib import MaskablePPO, TRPO, RecurrentPPO

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from src.metrics import evaluate_model

MODELS = {
    "PPO": PPO,
    "A2C": A2C,
    "TRPO": TRPO,
    "RecurrentPPO": RecurrentPPO,
}

# TODO: se podría hacer búsqueda de hiperparámetros
HYPERPARAMS = {
    "PPO": {
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "clip_range": 0.2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.0,
        "learning_rate": 3e-4,
    },
    "RecurrentPPO": {
        "n_steps": 128,
        "batch_size": 64,
        "n_epochs": 5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "learning_rate": 3e-4,
        "policy_kwargs": dict(lstm_hidden_size=256, shared_lstm=False),
    },
    "A2C": {
        "n_steps": 5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "learning_rate": 7e-4,
    },
    "TRPO": {
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "cg_max_steps": 10,
        "learning_rate": 3e-4,  # optional
    },
}


class ReinforcementLearningModels:

    def __init__(
        self, eval_env: DummyVecEnv, train_env: DummyVecEnv, model: str
    ) -> None:
        self.__eval_env = eval_env
        self.__train_env = train_env
        self.__name_model = model
        self.__model_file = f"{self.__name_model}_Ant.zip"
        self.__model = MODELS[model]

    def execute_model(self):

        policy = "MlpLstmPolicy" if self.__name_model == "RecurrentPPO" else "MlpPolicy"

        eval_callback = (
            EvalCallback(  # -> cada cierta cantidad de pasos ocurre una evaluación
                self.__eval_env,
                best_model_save_path="logs",
                log_path="logs",
                eval_freq=10_000,  # -> cada 10k pasos se evalúa
                n_eval_episodes=5,
                deterministic=True,
                render=False,
            )
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=100_000, # -> cada 100k se guarda un checkpoint del modelo
            save_path="checkpoints",
            name_prefix=f"{self.__name_model}_Ant"
        )


        hyper = HYPERPARAMS[self.__name_model]

        model_reinforcement = self.__model(
            policy,
            self.__train_env,
            device="cuda",  # -> por ahora se usa CPU para entrenar el modelo (cuda me daba problemas)
            verbose=1,
            tensorboard_log="logs",
            **hyper,
        )

        # --- Entrenamiento ---
        model_reinforcement.save(f"checkpoints/{self.__name_model}_Ant_0_steps") # -> se guarda el modelo inicial
        
        # NOTE: 1M timesteps no es óptimo para la mayoría de modelos; probablemente hay que aumentar       
        model_reinforcement.learn(total_timesteps=1_000_000, callback=[eval_callback, checkpoint_callback])

        # --- Guardado del modelo ---
        model_reinforcement.save(f"{self.__name_model}_Ant")
        print(f"Modelo guardado en {self.__model_file}")

        self.__train_env.close()
        self.__eval_env.close()

        return model_reinforcement
