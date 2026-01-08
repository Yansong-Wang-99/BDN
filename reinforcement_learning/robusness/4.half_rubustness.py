import torch
import gymnasium as gym
import numpy as np
import os
import csv
import datetime
from spikeactor_bilinea import SpikeActor
from dataclasses import dataclass
import math
import random
# ==============================
# 配置
# ==============================
@dataclass
class Args:
    env_id: str = "HalfCheetah-v4"
    seed: int = random.randint(0, 10000)
    cuda: bool = True
    exp_name: str = "bdn"
    device: str = "cuda:1" if torch.cuda.is_available() else "cpu"

args = Args()

# 多模型路径列表
model_paths = [

]

# 噪声水平
# 噪声水平设置（可以调整）

noise_levels = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,]

num_eval_episodes = 10


# ==============================
# 工具函数
# ==============================
class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        if isinstance(self.observation_space, gym.spaces.Box):
            self.observation_space = gym.spaces.Box(
                low=self.observation_space.low.astype(np.float32),
                high=self.observation_space.high.astype(np.float32),
                dtype=np.float32,
            )

    def observation(self, observation):
        if isinstance(observation, np.ndarray) and observation.dtype == np.float64:
            return observation.astype(np.float32)
        return observation


def load_model(model_path, args):
    """加载单个模型"""
    checkpoint = torch.load(model_path, map_location=args.device, weights_only=False)
    print(f"✅ Loaded {os.path.basename(model_path)} | step={checkpoint.get('global_step', 0)} | best_reward={checkpoint.get('best_reward', 0):.2f}")

    dummy_env = gym.make(args.env_id)
    obs_dim = dummy_env.observation_space.shape[0]
    act_dim = dummy_env.action_space.shape[0]

    actor = SpikeActor(
        env=dummy_env,
        obs_dim=obs_dim,
        act_dim=act_dim,
        en_pop_dim=10,
        de_pop_dim=10,
        hidden_sizes=256,
        mean_range=(-3, 3),
        std=math.sqrt(0.15),
        spike_ts=5,
        device=args.device
    ).to(args.device)

    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    dummy_env.close()
    return actor

def evaluate_with_noise_parallel(actor, device, args, noise_level=0.0, num_eval_episodes=10, num_envs=16):
    """并行评估一个模型在指定噪声水平下的平均回报"""
    def make_env(seed_offset):
        def thunk():
            env = gym.make(args.env_id)
            env = ObservationWrapper(env)
            env.reset(seed=args.seed + seed_offset)
            return env
        return thunk

    # 创建并行环境
    envs = gym.vector.SyncVectorEnv([make_env(i) for i in range(num_envs)])
    obs, _ = envs.reset(seed=args.seed)

    episode_returns = []
    current_returns = np.zeros(num_envs, dtype=np.float32)

    while len(episode_returns) < num_eval_episodes:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(device)
        with torch.no_grad():
            actions = actor(obs_tensor, batch_size=obs_tensor.shape[0]).cpu().numpy()

        # 添加高斯噪声
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, size=actions.shape)
            actions = np.clip(actions + noise, envs.single_action_space.low, envs.single_action_space.high)

        next_obs, rewards, terms, truncs, _ = envs.step(actions)
        dones = np.logical_or(terms, truncs)
        current_returns += rewards

        for i, done in enumerate(dones):
            if done:
                episode_returns.append(current_returns[i])
                current_returns[i] = 0.0
                if len(episode_returns) >= num_eval_episodes:
                    break

        obs = next_obs

    envs.close()
    return float(np.mean(episode_returns)), float(np.std(episode_returns)), episode_returns


# ==============================
# 主流程
# ==============================
if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{timestamp}"
    csv_path = f"robustness_results_{run_name}.csv"

    # ✅ 提前加载模型
    print("🔹 Loading all models once...")
    actors = []
    for path in model_paths:
        actor = load_model(path, args)
        actors.append(actor)
    print(f"✅ Loaded {len(actors)} models successfully.\n")

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["noise_level", "mean_across_models", "std_across_models", *[f"model_{i+1}_return" for i in range(len(model_paths))]])

        # ✅ 遍历不同噪声水平
        for noise in noise_levels:
            model_means = []
            all_returns = []

            for i, actor in enumerate(actors):
                mean_r, std_r, returns = evaluate_with_noise_parallel(actor, args.device, args, noise, num_eval_episodes, num_envs=16)

                model_means.append(mean_r)
                all_returns.append(mean_r)
                print(f"Model={os.path.basename(model_paths[i])} | Noise={noise:.3f} | Mean={mean_r:.2f} | Std={std_r:.2f}")

            mean_across_models = np.mean(model_means)
            std_across_models = np.std(model_means)

            writer.writerow([noise, mean_across_models, std_across_models, *all_returns])
            print(f"✅ Noise={noise:.3f} | MeanAcross={mean_across_models:.2f} | StdAcross={std_across_models:.2f}")

    print(f"\n✅ 所有模型鲁棒性测试完成，结果已保存到：{csv_path}")
