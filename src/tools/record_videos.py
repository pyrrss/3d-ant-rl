import argparse
import gymnasium as gym
import mujoco
import numpy as np
from stable_baselines3 import PPO, A2C
from sb3_contrib import TRPO, RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

"""
herramienta para grabar checkpoints de modelos en distintas etapas de aprendizaje

uso:
    python -m src.tools.record_videos --model <modelo>

ej:
    python -m src.tools.record_videos --model PPO
    

comentar/descomentar lineas en main para grabar los checkpoints que se quieran
"""

MODELS = {
    "PPO": PPO,
    "A2C": A2C,
    "TRPO": TRPO,
    "RecurrentPPO": RecurrentPPO,
    "Random": None
}

def record_checkpoint(model_name: str = "Random", model_file: str = None, name_prefix: str = None, video_length: int = 1000):
    """
    se graba un video del checkpoint de un modelo
    """
    model = None
    if model_name != "Random":
        model = MODELS[model_name].load(model_file, device="cpu")

    env = gym.make("Ant-v5", render_mode="rgb_array")
    
    # extensión del suelo
    m = env.unwrapped.model
    floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    m.geom_size[floor_id] = [200.0, 200.0, 1.0]
    
    # ajuste de la textura del suelo
    mat_id = m.geom_matid[floor_id]
    if mat_id != -1:
      # duplica la repetición para no estirar la textura (ajusta a tu factor)
      m.mat_texrepeat[mat_id] *= 10.0  # o el factor que uses para size

    env.reset()

    env = DummyVecEnv([lambda: env])
    env = VecVideoRecorder(env, video_folder="videos", record_video_trigger=lambda step: step == 0, video_length=video_length, name_prefix=name_prefix)
    
    observation = env.reset()
    for _ in range(video_length):
        
        if model is not None:
            action, _ = model.predict(observation, deterministic=True) 
        else:
            action = np.asarray([env.action_space.sample()])
            if action.ndim == 1:
                action = action[None, ...]

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
    
    # record_checkpoint(model_name="Random", name_prefix="Random")

    # -- Grabación de checkpoints --
    record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_0_steps", f"{args.model}_0_steps")
    record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_100000_steps", f"{args.model}_100000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_200000_steps", f"{args.model}_200000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_300000_steps", f"{args.model}_300000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_400000_steps", f"{args.model}_400000_steps")
    record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_500000_steps", f"{args.model}_500000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_600000_steps", f"{args.model}_600000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_700000_steps", f"{args.model}_700000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_800000_steps", f"{args.model}_800000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_900000_steps", f"{args.model}_900000_steps")
    record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_1000000_steps", f"{args.model}_1000000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_1100000_steps", f"{args.model}_1100000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_1200000_steps", f"{args.model}_1200000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_1300000_steps", f"{args.model}_1300000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_1400000_steps", f"{args.model}_1400000_steps")
    record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_1500000_steps", f"{args.model}_1500000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_1600000_steps", f"{args.model}_1600000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_1700000_steps", f"{args.model}_1700000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_1800000_steps", f"{args.model}_1800000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_1900000_steps", f"{args.model}_1900000_steps")
    record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_2000000_steps", f"{args.model}_2000000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_2100000_steps", f"{args.model}_2100000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_2200000_steps", f"{args.model}_2200000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_2300000_steps", f"{args.model}_2300000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_2400000_steps", f"{args.model}_2400000_steps")
    record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_2500000_steps", f"{args.model}_2500000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_2600000_steps", f"{args.model}_2600000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_2700000_steps", f"{args.model}_2700000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_2800000_steps", f"{args.model}_2800000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_2900000_steps", f"{args.model}_2900000_steps")
    record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_3000000_steps", f"{args.model}_3000000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_3100000_steps", f"{args.model}_3100000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_3200000_steps", f"{args.model}_3200000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_3300000_steps", f"{args.model}_3300000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_3400000_steps", f"{args.model}_3400000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_3500000_steps", f"{args.model}_3500000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_3600000_steps", f"{args.model}_3600000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_3700000_steps", f"{args.model}_3700000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_3800000_steps", f"{args.model}_3800000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_3900000_steps", f"{args.model}_3900000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_4000000_steps", f"{args.model}_4000000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_4100000_steps", f"{args.model}_4100000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_4200000_steps", f"{args.model}_4200000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_4300000_steps", f"{args.model}_4300000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_4400000_steps", f"{args.model}_4400000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_4500000_steps", f"{args.model}_4500000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_4600000_steps", f"{args.model}_4600000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_4700000_steps", f"{args.model}_4700000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_4800000_steps", f"{args.model}_4800000_steps")
    # record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_4900000_steps", f"{args.model}_4900000_steps")
    record_checkpoint(args.model, f"checkpoints/{args.model}_Ant_5000000_steps", f"{args.model}_5000000_steps")


