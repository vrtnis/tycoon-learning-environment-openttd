from __future__ import annotations

import argparse

from openttd_le.adapters.gymnasium import make_firs_vector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", default="scenario.xlsx")
    parser.add_argument("--scenario", default="lab_raw_to_processor")
    parser.add_argument("--openttd-user-dir", default=".openttd")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--async-envs", action="store_true")
    args = parser.parse_args()

    envs = make_firs_vector(
        args.num_envs,
        asynchronous=args.async_envs,
        workbook=args.workbook,
        task_id=args.scenario,
        openttd_user_dir=args.openttd_user_dir,
        max_steps=args.steps,
    )
    try:
        obs, info = envs.reset(seed=list(range(1, args.num_envs + 1)))
        for step in range(1, args.steps + 1):
            masks = info.get("action_mask")
            actions = []
            for env_index in range(args.num_envs):
                mask = masks[env_index] if masks is not None else obs["action_mask"][env_index]
                valid = [idx for idx, value in enumerate(mask) if int(value)]
                actions.append(valid[0] if valid else 0)
            obs, rewards, terminated, truncated, info = envs.step(actions)
            print(step, "actions", actions, "rewards", rewards.tolist())
            if all(bool(done) for done in (terminated | truncated)):
                break
    finally:
        envs.close()


if __name__ == "__main__":
    main()
