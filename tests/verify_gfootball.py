"""Headless smoke test for gfootball (run: python tests/verify_gfootball.py)."""

import os


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def main() -> None:
    """Create a gfootball env, step it once, and print a success marker."""
    # Imported lazily: gfootball only installs inside the Linux container, and
    # the SDL env vars above must be set before it is imported.
    import gfootball  # noqa: PLC0415
    import gfootball.env as football_env  # noqa: PLC0415

    print("gfootball_file:", gfootball.__file__)
    env = football_env.create_environment(
        env_name="academy_empty_goal",
        render=False,
    )
    obs = env.reset()
    print("reset_ok shape:", getattr(obs, "shape", None))
    _obs2, reward, done, _info = env.step(env.action_space.sample())
    print("step_ok reward:", reward, "done:", done)
    print("GFOOTBALL_WORKS")


if __name__ == "__main__":
    main()
