"""Temporary runner for the Stage-2 home-pose layout search.

Loads config, builds the PyBullet scene, runs neurosim.layout.search_home_poses,
reports the found poses and the residual collision report, then exits.
"""
from __future__ import annotations

import json
from pathlib import Path

from neurosim.config import load_config
from neurosim.layout import home_collision_report, search_home_poses
from neurosim.scene import build_scene, close

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "robot_config.yaml"


def main() -> int:
    cfg = load_config(CONFIG_PATH)
    scene = build_scene(cfg, gui=False)

    print("=== Collision report at CONFIGURED home poses ===")
    before = home_collision_report(cfg, scene)
    total_before = sum(len(v) for v in before.values())
    for arm_id, contacts in before.items():
        print(f"  {arm_id}: {len(contacts)} contacts -> {contacts}")
    print(f"  TOTAL contacts: {total_before}")

    print("\n=== Running home-pose layout search ===")
    found = search_home_poses(cfg, scene)

    print("\n=== Collision report at FOUND home poses ===")
    after = home_collision_report(cfg, scene)
    total_after = sum(len(v) for v in after.values())
    for arm_id, contacts in after.items():
        print(f"  {arm_id}: {len(contacts)} contacts -> {contacts}")
    print(f"  TOTAL contacts: {total_after}")

    print("\n=== Found poses (deg) ===")
    for arm_id, q in found.items():
        print(f"  {arm_id}: {[round(v, 3) for v in q]}")

    # Save to JSON for the config edit step.
    out = {
        arm_id: [float(v) for v in q]
        for arm_id, q in found.items()
    }
    (PROJECT_ROOT / "layout_search_results.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print("\nWrote layout_search_results.json")

    close(scene)
    return 0 if total_after == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
