import argparse
import gymnasium as gym

from stable_baselines3 import PPO, A2C
from sb3_contrib import TRPO, RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

MODELS = {
    "PPO": PPO,
    "A2C": A2C,
    "TRPO": TRPO,
    "RecurrentPPO": RecurrentPPO,
}


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
        
        # si al hacer un step se acaba el episodio (por ejemplo si la hormiga se cae, se reinicia y asi se asegura largo del video)
        if dones.any():
            observation = env.reset()
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, choices=MODELS.keys() , help="Nombre de modelo a grabar")
   

    args = parser.parse_args()
    
    if not args.model:
        raise SystemExit("No se especificó modelo")


    # -- Grabación de checkpoints --
    # record_checkpoint(f"checkpoints/{args.model}_Ant_0_steps", f"{args.model}_0_steps")
    record_checkpoint(f"checkpoints/{args.model}_Ant_100000_steps", f"{args.model}_100000_steps")
    record_checkpoint(f"checkpoints/{args.model}_Ant_200000_steps", f"{args.model}_200000_steps")
    record_checkpoint(f"checkpoints/{args.model}_Ant_300000_steps", f"{args.model}_300000_steps")
    record_checkpoint(f"checkpoints/{args.model}_Ant_400000_steps", f"{args.model}_400000_steps")
    record_checkpoint(f"checkpoints/{args.model}_Ant_500000_steps", f"{args.model}_500000_steps")
    record_checkpoint(f"checkpoints/{args.model}_Ant_600000_steps", f"{args.model}_600000_steps")
    record_checkpoint(f"checkpoints/{args.model}_Ant_700000_steps", f"{args.model}_700000_steps")
    record_checkpoint(f"checkpoints/{args.model}_Ant_800000_steps", f"{args.model}_800000_steps")
    record_checkpoint(f"checkpoints/{args.model}_Ant_900000_steps", f"{args.model}_900000_steps")
    record_checkpoint(f"checkpoints/{args.model}_Ant_1000000_steps", f"{args.model}_1000000_steps")
    


