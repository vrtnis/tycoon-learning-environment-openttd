from __future__ import annotations

import argparse

from openttd_le.adapters.gymnasium import OpenTTDFIRSGymEnv


def main() -> None:
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise SystemExit("Install training dependencies first: python -m pip install sb3-contrib stable-baselines3") from exc

    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", default="scenario.xlsx")
    parser.add_argument("--scenario", default="lab_raw_to_processor")
    parser.add_argument("--openttd-user-dir", default=".openttd")
    parser.add_argument("--timesteps", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--save", default="maskable_ppo_openttd_firs.zip")
    args = parser.parse_args()

    env = OpenTTDFIRSGymEnv(
        workbook=args.workbook,
        task_id=args.scenario,
        openttd_user_dir=args.openttd_user_dir,
        max_steps=args.max_steps,
    )
    try:
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            n_steps=max(2, args.max_steps),
            batch_size=max(2, args.max_steps),
        )
        model.learn(total_timesteps=args.timesteps)
        model.save(args.save)
    finally:
        env.close()


if __name__ == "__main__":
    main()
