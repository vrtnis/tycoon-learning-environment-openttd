from __future__ import annotations

import argparse

from openttd_le.adapters.gymnasium import OpenTTDFIRSGymEnv


def main() -> None:
    try:
        from stable_baselines3 import DQN
    except ImportError as exc:
        raise SystemExit("Install training dependencies first: python -m pip install stable-baselines3") from exc

    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", default="scenario.xlsx")
    parser.add_argument("--scenario", default="lab_raw_to_processor")
    parser.add_argument("--openttd-user-dir", default=".openttd")
    parser.add_argument("--timesteps", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--save", default="dqn_openttd_firs.zip")
    args = parser.parse_args()

    env = OpenTTDFIRSGymEnv(
        workbook=args.workbook,
        task_id=args.scenario,
        openttd_user_dir=args.openttd_user_dir,
        max_steps=args.max_steps,
        deterministic=True,
    )
    try:
        model = DQN(
            "MultiInputPolicy",
            env,
            verbose=1,
            learning_starts=0,
            buffer_size=max(128, args.timesteps),
            train_freq=1,
            gradient_steps=1,
        )
        model.learn(total_timesteps=args.timesteps)
        model.save(args.save)
    finally:
        env.close()


if __name__ == "__main__":
    main()
