from __future__ import annotations

import argparse

from gymnasium.envs.registration import registry
import minigrid  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description="List locally registered MiniGrid environments.")
    parser.add_argument("--contains", default="", help="Optional case-insensitive substring filter.")
    args = parser.parse_args()

    needle = args.contains.lower()
    env_ids = sorted(
        spec.id
        for spec in registry.values()
        if spec.id.startswith("MiniGrid-") and (not needle or needle in spec.id.lower())
    )
    for env_id in env_ids:
        print(env_id)
    print(f"\ncount={len(env_ids)}")


if __name__ == "__main__":
    main()
