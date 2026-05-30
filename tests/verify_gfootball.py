"""Headless smoke test for gfootball. Run with: .venv/bin/python tests/verify_gfootball.py"""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def main() -> None:
    import gfootball
    import gfootball.env as football_env

    print("gfootball_file:", gfootball.__file__)
    env = football_env.create_environment(
        env_name="academy_empty_goal", render=False
    )
    obs = env.reset()
    print("reset_ok shape:", getattr(obs, "shape", None))
    obs2, reward, done, info = env.step(env.action_space.sample())
    print("step_ok reward:", reward, "done:", done)
    print("GFOOTBALL_WORKS")


if __name__ == "__main__":
    main()
