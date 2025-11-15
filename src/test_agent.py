import gymnasium as gym
import numpy as np

from blackjack_agent import BlackjackAgent

def test_agent(agent: BlackjackAgent, env: gym.Env, num_episodes=1000):
    total_rewards = []

    # -> se desactiva temporalmente exploración para testear
    # (agente solo usa su conocimiento)
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0 # -> solo explotación
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += float(reward)
            done = terminated or truncated

        total_rewards.append(episode_reward)           

    # -> se restaura epsilon previo del agente
    agent.epsilon = old_epsilon
    
    win_rate = np.mean(np.array(total_rewards) > 0)
    average_rewards = np.mean(total_rewards)
    
    print(f"Resultados de tests en {num_episodes} episodios:")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Reward promedio: {average_rewards:.3f}")
    print(f"Desviación estándar: {np.std(total_rewards):.3f}")


