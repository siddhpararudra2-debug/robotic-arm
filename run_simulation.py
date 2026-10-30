"""Stage 1 entry point: assemble the scene, verify it, optionally show it.

Run from the project folder:

    python run_simulation.py            # headless: build + report + exit
    python run_simulation.py --gui      # open the PyBullet 3D viewer

In GUI mode, one slider per revolute joint lets you pose the arms in real
time and the tool-tip positions are shown as floating text above each
instrument. Press H to return all arms to home, Q to quit.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Any

import pybullet as p

from neurosim.config import MM_PER_M, load_config, resolve_arm_joints
from neurosim.scene import (
    DEFAULT_CONFIG_PATH,
    Scene,
    arm_arm_contacts,
    build_scene,
    close,
    contacts_with_body,
    reset_to_home,
    update_debug_visuals,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gui", action="store_true", help="open the PyBullet 3D viewer"
    )
    parser.add_argument(
        "--config", default=None, help="path to robot_config.yaml"
    )
    parser.add_argument(
        "--headless-time",
        type=float,
        default=2.0,
        help="seconds to run in headless mode before exiting",
    )
    args = parser.parse_args(argv)

    config_path = args.config if args.config else str(DEFAULT_CONFIG_PATH)
    cfg = load_config(config_path)
    scene = build_scene(cfg, gui=args.gui)

    print("Scene assembled:")
    print(f"  robot URDF body id: {scene.robot_id}")
    print(f"  phantom head body id: {scene.head_id}")
    print(f"  clamp bodies: {len(scene.clamp_ids)} (posts + ring segments)")
    print(f"  table body id: {scene.table_id}")
    print(f"  ground body id: {scene.ground_id}")
    for arm_id, indices in sorted(scene.arm_joint_indices.items()):
        print(f"  {arm_id}: {len(indices)} revolute joints at home")

    arm_contacts = arm_arm_contacts(scene)
    clamp_contacts: list[str] = []
    for body_id in scene.clamp_ids:
        clamp_contacts.extend(contacts_with_body(scene, body_id))
    table_contacts = contacts_with_body(scene, scene.table_id)
    head_contacts = contacts_with_body(scene, scene.head_id)

    print("Collision check at home poses:")
    print(f"  arm-arm: {'NONE' if not arm_contacts else arm_contacts}")
    print(f"  robot-clamp: {'NONE' if not clamp_contacts else clamp_contacts}")
    print(f"  robot-table: {'NONE' if not table_contacts else table_contacts}")
    print(f"  robot-head: {'NONE' if not head_contacts else head_contacts}")

    if args.gui:
        _run_gui_loop(scene, cfg)
    else:
        time.sleep(args.headless_time)

    close(scene)
    return 0


def _run_gui_loop(scene: Scene, cfg: dict[str, Any]) -> None:
    """Interactive GUI: joint sliders, live tip readout, keyboard controls."""
    client_id = scene.client_id
    robot_id = scene.robot_id

    # Create one slider per revolute joint, initialized to its home angle.
    sliders: dict[int, tuple[int, str, int]] = {}
    for arm in cfg["arms"]:
        arm_id = arm["id"]
        joints = resolve_arm_joints(cfg, arm)
        for joint_num, joint_index in enumerate(
            scene.arm_joint_indices[arm_id], start=1
        ):
            joint = joints[joint_num - 1]
            home_deg = float(arm["home_deg"][joint_num - 1])
            slider_id = p.addUserDebugParameter(
                f"{arm_id} j{joint_num}",
                joint["lower_deg"],
                joint["upper_deg"],
                home_deg,
                physicsClientId=client_id,
            )
            sliders[joint_index] = (slider_id, arm_id, joint_num)

    # On-screen control hint.
    p.addUserDebugText(
        "Keys:  H = home  |  Q = quit  |  drag sliders to pose joints",
        (0.0, 0.0, 1.55),
        textColorRGB=(1.0, 1.0, 1.0),
        textSize=1.2,
        physicsClientId=client_id,
    )

    print(
        "GUI controls: drag any joint slider to pose the arms in real time. "
        "Press H to return all arms to home, Q to quit."
    )
    print("  Tool-tip positions appear as floating text above each instrument.")

    last_print = 0.0
    try:
        while True:
            keys = p.getKeyboardEvents(physicsClientId=client_id)

            if keys.get(ord("q"), 0) & p.KEY_WAS_TRIGGERED:
                break
            if keys.get(ord("h"), 0) & p.KEY_WAS_TRIGGERED:
                reset_to_home(scene, cfg)

            # Read every slider and servo its joint to the target angle.
            for joint_index, (slider_id, _arm_id, _joint_num) in sliders.items():
                target_deg = p.readUserDebugParameter(
                    slider_id, physicsClientId=client_id
                )
                target_rad = math.radians(target_deg)
                p.setJointMotorControl2(
                    robot_id,
                    joint_index,
                    p.POSITION_CONTROL,
                    targetPosition=target_rad,
                    force=500.0,
                    physicsClientId=client_id,
                )

            p.stepSimulation(physicsClientId=client_id)
            update_debug_visuals(scene, cfg)

            # Periodic console readout: tip positions + collision distances.
            now = time.time()
            if now - last_print > 0.5:
                last_print = now
                _print_tip_readout(scene, cfg)
                _print_collision_report(scene)

            time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
        pass


def _print_collision_report(scene: Scene) -> None:
    """Print the closest approach between arms and to the head/clamp/table."""
    arm_contacts = arm_arm_contacts(scene)
    head_contacts = contacts_with_body(scene, scene.head_id)
    other_contacts: list[str] = []
    for body_id in scene.clamp_ids + [scene.table_id]:
        other_contacts.extend(contacts_with_body(scene, body_id))

    lines: list[str] = []
    if arm_contacts:
        lines.append(f"  arm-arm contacts: {arm_contacts}")
    else:
        lines.append("  arm-arm: clear")
    if head_contacts:
        lines.append(f"  robot-head contacts: {head_contacts}")
    else:
        lines.append("  robot-head: clear")
    if other_contacts:
        lines.append(f"  robot-clamp/table contacts: {other_contacts}")
    else:
        lines.append("  robot-clamp/table: clear")
    print("Collision distances:")
    print("\n".join(lines))


def _print_tip_readout(scene: Scene, cfg: dict[str, Any]) -> None:
    """Print the current tool-tip position of every arm to the console."""
    robot_id = scene.robot_id
    client_id = scene.client_id
    lines: list[str] = []
    for arm in cfg["arms"]:
        arm_id = arm["id"]
        link_name = f"{arm_id}_link6"
        link_index = scene.link_index_by_name.get(link_name)
        if link_index is None:
            continue
        link_state = p.getLinkState(robot_id, link_index, physicsClientId=client_id)
        pos = link_state[0]
        orn = link_state[1]
        tool_length_m = (
            float(arm["geometry"]["link_lengths_mm"]["tool"]) / MM_PER_M
        )
        _, tip_end = p.multiplyTransforms(
            pos, orn, (0.0, 0.0, 0.0), (tool_length_m, 0.0, 0.0)
        )
        lines.append(
            f"  {arm_id}: ({tip_end[0] * 1000:.1f}, "
            f"{tip_end[1] * 1000:.1f}, {tip_end[2] * 1000:.1f}) mm"
        )
    print("Tool-tip positions:")
    print("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
