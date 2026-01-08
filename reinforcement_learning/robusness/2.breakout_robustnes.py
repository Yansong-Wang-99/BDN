import os
import torch
import numpy as np
import gymnasium as gym
import csv
from bdn_bilinear_breakout import Agent, Args, make_env
from cleanrl_utils.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

# ------------------------
# 配置
# ------------------------
model_paths = [

]
# noise_levels = [0.0, 0.01, 0.02, 0.03, 0.04]
num_eval_episodes = 10
noise_levels = [0,0.005, 0.01,0.015 , 0.02 ,0.025,0.03,0.035,0.04]
import os
import time

# ------------------------
# 自动生成结果文件名
# ------------------------
script_name = os.path.splitext(os.path.basename(__file__))[0]  # e.g. "bdn_v953a_break_eval_noise_final"
timestamp = time.strftime("%Y%m%d_%H%M%S")                     # e.g. "20251012_231512"
output_dir = "results"                                         # 保存目录
output_csv = os.path.join(output_dir, f"{script_name}__{timestamp}.csv")

# 后面照常使用 output_csv 保存结果

# ------------------------
# 加载模型
# ------------------------
def load_agent(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    args_dict = checkpoint.get("args", {})
    args = Args()
    for k, v in args_dict.items():
        setattr(args, k, v)

    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, 0, False, "eval")])
    agent = Agent(envs).to(device)

    model_state = checkpoint["model_state_dict"]
    # ✅ 修复 torch.compile 前缀问题
    cleaned_state_dict = {k.replace("_orig_mod.", ""): v for k, v in model_state.items()}
    load_result = agent.load_state_dict(cleaned_state_dict, strict=False)

    missing = load_result.missing_keys
    unexpected = load_result.unexpected_keys
    print(f"Loaded checkpoint: {model_path}")
    print(f"  -> missing keys  : {len(missing)} (examples: {missing[:3]})")
    print(f"  -> unexpected keys: {len(unexpected)} (examples: {unexpected[:3]})")

    envs.close()
    agent.eval()
    return agent, args

def evaluate_with_noise(agent, args, device, noise_level, num_eval_episodes=10):
    """评估一个agent在指定噪声水平下的平均回报"""
    eval_env = gym.make(args.env_id, obs_type="grayscale", render_mode=None)
    eval_env = gym.wrappers.RecordEpisodeStatistics(eval_env)
    eval_env = NoopResetEnv(eval_env, noop_max=30)
    eval_env = MaxAndSkipEnv(eval_env, skip=4)
    # eval_env = EpisodicLifeEnv(eval_env)
    if "FIRE" in eval_env.unwrapped.get_action_meanings():
        eval_env = FireResetEnv(eval_env)
    # eval_env = ClipRewardEnv(eval_env)
    eval_env = gym.wrappers.ResizeObservation(eval_env, (84, 84))
    env = gym.wrappers.FrameStack(eval_env, 4)

    returns = []
    for ep in range(num_eval_episodes):
        obs, _ = env.reset(seed=args.seed + 100 + ep)
        obs = torch.from_numpy(np.array(obs)).float().to(device).unsqueeze(0)
        done = False
        total_reward = 0
        while not done:
            noisy_obs = obs + torch.randn_like(obs) * (noise_level * 255.0)
            noisy_obs = torch.clamp(noisy_obs, 0, 255)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(noisy_obs, deterministic=True)
            obs_next, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            obs = torch.from_numpy(np.array(obs_next)).float().to(device).unsqueeze(0)
            total_reward += reward
        returns.append(total_reward)

    env.close()
    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    return mean_ret, std_ret


# ------------------------
# 主流程
# ------------------------
if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # ✅ 先一次性加载所有模型
    loaded_models = []
    print("Loading all models...\n")
    for i, model_path in enumerate(model_paths):
        agent, args = load_agent(model_path, device)
        loaded_models.append((agent, args))
        print(f"Loaded model {i+1}/{len(model_paths)} ✅\n")

    # ✅ 对每个噪声水平测试所有模型
    all_results = []
    for noise_level in noise_levels:
        sigma_pixel = noise_level * 255.0
        print(f"\n=== Evaluating noise_frac={noise_level:.3f} (sigma_pixel={sigma_pixel:.3f}) ===")

        model_returns = []
        for i, (agent, args) in enumerate(loaded_models):
            mean_return, std_return = evaluate_with_noise(agent, args, device, noise_level, num_eval_episodes)
            print(f" Model {i+1}: mean_return={mean_return:.3f}, std={std_return:.3f}")
            model_returns.append(mean_return)

        mean_across = float(np.mean(model_returns))
        std_across = float(np.std(model_returns))
        print(f" -> Average across {len(model_returns)} models: {mean_across:.3f} (std {std_across:.3f})")

        all_results.append([noise_level, mean_across, std_across, *model_returns])

    # ✅ 保存 CSV（安全处理空路径）
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    header = ["noise_level", "mean_across_models", "std_across_models"] + [
        f"model_{i+1}_return" for i in range(len(model_paths))
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_results)

    print("\n✅ Robustness test finished.")
    print(f"Results saved to: {output_csv}")
