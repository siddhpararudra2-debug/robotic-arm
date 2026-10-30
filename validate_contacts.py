"""Validate _contacts_for reproduces the original home-collision report."""
from __future__ import annotations

from neurosim.config import load_config
from neurosim.layout import home_collision_report
from neurosim.scene import build_scene, close

CONFIG = "c:\\Users\\rudra\\Desktop\\neurosurgical_robot\\robot_config.yaml"


def main() -> None:
    cfg = load_config(CONFIG)
    scene = build_scene(cfg, gui=False)
    report = home_collision_report(cfg, scene)
    total = 0
    for arm_id in ("A1", "A2", "A3", "A4"):
        contacts = report.get(arm_id, [])
        n_arm = sum(1 for c in contacts if c.startswith("arm:"))
        n_head = sum(1 for c in contacts if c.startswith("head:"))
        n_ct = sum(1 for c in contacts if c.startswith("clamp_table:"))
        total += len(contacts)
        print(f"{arm_id}: {len(contacts)} contacts (arm={n_arm}, head={n_head}, clamp_table={n_ct})")
        for c in contacts:
            print(f"    {c}")
    print(f"TOTAL: {total}")
    close(scene)


if __name__ == "__main__":
    main()
