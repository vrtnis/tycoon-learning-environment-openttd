from __future__ import annotations

import argparse
import random

from openttd_le.adapters.gymnasium import OpenTTDFIRSGymEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", default="scenario.xlsx")
    parser.add_argument("--scenario", default="lab_raw_to_processor")
    parser.add_argument("--openttd-user-dir", default=".openttd")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    env = OpenTTDFIRSGymEnv(
        workbook=args.workbook,
        task_id=args.scenario,
        openttd_user_dir=args.openttd_user_dir,
        max_steps=args.max_steps,
    )
    try:
        obs, info = env.reset(seed=args.seed)
        total_reward = 0.0
        for step in range(1, args.max_steps + 1):
            mask = list(env.action_masks())
            valid = [index for index, value in enumerate(mask) if int(value)]
            action = rng.choice(valid) if valid else 0
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            print(step, "action_index", action, "reward", reward, "done", terminated or truncated)
            if terminated or truncated:
                break
        print(env.render())
        print("total_reward", round(total_reward, 3))
    finally:
        env.close()


if __name__ == "__main__":
    main()
