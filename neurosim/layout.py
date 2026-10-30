"""Stage 2: collision-aware layout review of the assumed home poses.

The Stage-0 base mounts and home poses are marked ASSUMPTION in the config;
this module verifies them in the assembled PyBullet scene and can search for
collision-free home poses when the assumptions turn out to collide.

The search is a greedy per-arm grid over (j1, j3, j5) around the configured
home, scoring candidates by collision count and distance to the original
home. It is a layout review tool, not a motion planner.
"""

from __future__ import annotations

import math
from typing import Any

import pybullet as p

from .config import MM_PER_M
from .kinematics import build_arm_chain
from .scene import Scene, _link_name

TOOL_LINK_PATTERN = ("link6", "tip")


def _arm_link_indices(scene: Scene, arm_id: str) -> list[int]:
    return [
        index for index, owner in scene.arm_of_link.items()
        if owner == arm_id
    ]


def _contacts_for(
    scene: Scene,
    moved_arm: str,
    allow_head_tool_contact: bool = True,
) -> list[str]:
    """Collision labels for the moved arm against everything else.

    Every other arm counts as an obstacle at whatever pose it currently holds.
    """
    robot = scene.robot_id
    labels: list[str] = []
    moved_links = _arm_link_indices(scene, moved_arm)

    def record(kind: str, other: str) -> None:
        labels.append(f"{kind}:{other}")

    for link_index in moved_links:
        link_name = _link_name(scene, link_index)
        is_tool = any(link_name.endswith(suffix) for suffix in TOOL_LINK_PATTERN)

        # Robot links of other arms (at their current poses).
        for other_index, owner in scene.arm_of_link.items():
            if owner == moved_arm or owner == "world":
                continue
            points = p.getClosestPoints(
                robot, robot, distance=0.0005,
                linkIndexA=link_index, linkIndexB=other_index,
                physicsClientId=scene.client_id,
            )
            if points:
                record("arm", _link_name(scene, other_index))

        # Phantom head: tool links may rest against it.
        head_points = p.getClosestPoints(
            robot, scene.head_id, distance=0.0005,
            linkIndexA=link_index, physicsClientId=scene.client_id,
        )
        if head_points and not (allow_head_tool_contact and is_tool):
            record("head", link_name)

        # Clamp and table: never allowed.
        for body_id in scene.clamp_ids + [scene.table_id]:
            if p.getClosestPoints(
                robot, body_id, distance=0.0005,
                linkIndexA=link_index, physicsClientId=scene.client_id,
            ):
                record("clamp_table", _link_name(scene, link_index))
    return labels


def _set_arm_joints(
    scene: Scene, cfg: dict[str, Any], arm_id: str, q_deg: list[float]
) -> None:
    for joint_index, angle in zip(scene.arm_joint_indices[arm_id], q_deg):
        p.resetJointState(
            scene.robot_id, joint_index, math.radians(angle),
            physicsClientId=scene.client_id,
        )


def home_collision_report(cfg: dict[str, Any], scene: Scene) -> dict[str, list[str]]:
    """Collision labels per arm at the configured home poses."""
    report: dict[str, list[str]] = {}
    for arm in cfg["arms"]:
        _set_arm_joints(scene, cfg, arm["id"], [float(v) for v in arm["home_deg"]])
    p.performCollisionDetection(physicsClientId=scene.client_id)
    for arm in cfg["arms"]:
        report[arm["id"]] = _contacts_for(scene, arm["id"])
    return report


def search_home_poses(
    cfg: dict[str, Any],
    scene: Scene,
    yaw_span_deg: float = 40.0,
    yaw_step_deg: float = 10.0,
    shoulder_deltas_deg: tuple[float, ...] = (0.0, -15.0, -30.0, 15.0),
    elbow_deltas_deg: tuple[float, ...] = (0.0, 15.0, 30.0, 45.0, 60.0),
    wrist_deltas_deg: tuple[float, ...] = (0.0, 15.0, -15.0, 30.0),
) -> dict[str, list[float]]:
    """Find collision-free home poses, staying as close to the config as possible.

    Greedy per arm in config order; every other arm counts as an obstacle at
    its current pose. Scores candidates by (contacts, total change from the
    configured home). Returns arm_id -> six joint angles in degrees.
    """
    fixed: dict[str, list[float]] = {}

    for arm in cfg["arms"]:
        arm_id = arm["id"]
        base_home = [float(v) for v in arm["home_deg"]]
        chain = build_arm_chain(cfg, arm)
        lower, upper = chain.lower_deg, chain.upper_deg

        best: tuple[int, float, list[float]] | None = None
        yaw = float(base_home[0]) - yaw_span_deg
        while yaw <= float(base_home[0]) + yaw_span_deg + 1e-9:
            for shoulder_delta in shoulder_deltas_deg:
                for elbow_delta in elbow_deltas_deg:
                    for wrist_delta in wrist_deltas_deg:
                        candidate = list(base_home)
                        candidate[0] = yaw
                        candidate[1] = base_home[1] + shoulder_delta
                        candidate[2] = base_home[2] + elbow_delta
                        candidate[4] = base_home[4] + wrist_delta
                        candidate = [
                            min(max(v, lower[i]), upper[i])
                            for i, v in enumerate(candidate)
                        ]
                        _set_arm_joints(scene, cfg, arm_id, candidate)
                        for other_id, other_q in fixed.items():
                            _set_arm_joints(scene, cfg, other_id, other_q)
                        # getClosestPoints queries current link transforms
                        # directly; performCollisionDetection is only needed to
                        # populate the constraint-solver contact manifold, so it
                        # is redundant (and expensive) here.
                        contacts = _contacts_for(scene, arm_id)
                        distance = sum(
                            abs(candidate[i] - base_home[i]) for i in range(6)
                        )
                        score = (len(contacts), distance)
                        if best is None or score < best[:2]:
                            best = (score[0], score[1], candidate)
        yaw += yaw_step_deg
        assert best is not None
        fixed[arm_id] = best[2]
        print(
            f"    {arm_id}: {best[0]} contacts, {best[1]:.2f} deg total change -> "
            f"{[round(v, 2) for v in best[2]]}",
            flush=True,
        )

    # Leave the scene at the found poses.
    for arm_id, q in fixed.items():
        _set_arm_joints(scene, cfg, arm_id, q)
    p.performCollisionDetection(physicsClientId=scene.client_id)
    return fixed
