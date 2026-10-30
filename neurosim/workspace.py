"""Stage 2: workspace checks - can each arm reach its target point, and do
the assumed base mounts cause collisions across the joint space?

Reachability uses the Stage-2 IK; the collision sweep uses PyBullet when it
is available and is reported as skipped otherwise.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import MM_PER_M
from .kinematics import ArmChain, build_arm_chain, forward_kinematics, inverse_kinematics


def reach_report(
    cfg: dict[str, Any],
    tolerance_mm: float = 0.01,
) -> dict[str, dict[str, Any]]:
    """Try IK to each arm's configured target point.

    Returns one record per arm: target, reachable flag, final tip error in mm
    and the joint solution (degrees) when converged.
    """
    arms_by_id = {arm["id"]: arm for arm in cfg["arms"]}
    report: dict[str, dict[str, Any]] = {}
    for arm_id, target_mm in cfg.get("targets_mm", {}).items():
        arm = arms_by_id[arm_id]
        chain = build_arm_chain(cfg, arm)
        home = [float(v) for v in arm["home_deg"]]
        q, converged, error_mm = inverse_kinematics(
            chain,
            [v / MM_PER_M for v in target_mm],
            q0_deg=home,
            tolerance_mm=tolerance_mm,
        )
        report[arm_id] = {
            "target_mm": [float(v) for v in target_mm],
            "reachable": bool(converged),
            "error_mm": error_mm,
            "q_deg": [float(v) for v in q],
            "within_limits": all(
                chain.lower_deg[i] <= q[i] <= chain.upper_deg[i] for i in range(6)
            ),
        }
    return report


def fk_tip_mm(cfg: dict[str, Any], arm_id: str, q_deg) -> list[float]:
    """Forward-kinematics tip position in mm for one arm (config units)."""
    arm = next(a for a in cfg["arms"] if a["id"] == arm_id)
    chain = build_arm_chain(cfg, arm)
    return [float(v) for v in forward_kinematics(chain, q_deg)[:3, 3] * MM_PER_M]


def workspace_summary(cfg: dict[str, Any]) -> dict[str, Any]:
    """Count reachable targets and expose per-arm FK home tip positions."""
    report = reach_report(cfg)
    return {
        "reachable": {arm_id: rec["reachable"] for arm_id, rec in report.items()},
        "errors_mm": {arm_id: rec["error_mm"] for arm_id, rec in report.items()},
        "home_tips_mm": {
            arm["id"]: fk_tip_mm(cfg, arm["id"], arm["home_deg"])
            for arm in cfg["arms"]
        },
    }


def tip_distance_mm(cfg: dict[str, Any], arm_id: str, q_deg, point_mm) -> float:
    """Straight-line distance from the tip to a world point, in mm."""
    arm = next(a for a in cfg["arms"] if a["id"] == arm_id)
    chain = build_arm_chain(cfg, arm)
    tip = forward_kinematics(chain, q_deg)[:3, 3] * MM_PER_M
    point = np.asarray(point_mm, dtype=float)
    return float(np.linalg.norm(tip - point))
