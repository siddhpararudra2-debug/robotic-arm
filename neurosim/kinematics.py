"""Stage 2: forward/inverse kinematics for the 4-arm robot.

The chain layout is the one the Stage-0 URDF generator uses:

    mount -> base column -> j1 -> j2 (same origin as j1)
          -> [upper_arm] -> j3 -> [forearm] -> j4
          -> [wrist_roll] -> j5 -> [wrist_pitch] -> j6
          -> [tool] -> tip

Joint axes and every length come from ``robot_config.yaml``; nothing here is
hardcoded. All public functions take angles in degrees and return positions in
metres (PyBullet units).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .config import MM_PER_M

AXIS_VECTORS = {"x": np.array([1.0, 0.0, 0.0]),
                "y": np.array([0.0, 1.0, 0.0]),
                "z": np.array([0.0, 0.0, 1.0])}


@dataclass
class ArmChain:
    """Precomputed FK step list for one arm."""

    arm_id: str
    base_xyz_m: np.ndarray
    base_rpy_rad: np.ndarray
    column_m: float
    axes: list[np.ndarray]
    offsets: list[np.ndarray]  # translation applied after each joint rotation
    lower_deg: np.ndarray
    upper_deg: np.ndarray


def build_arm_chain(cfg: dict[str, Any], arm: dict[str, Any]) -> ArmChain:
    """Flatten the config into a chain of rotations and translations."""
    mount = arm["base_mount"]
    lengths = arm["geometry"]["link_lengths_mm"]
    axes = [AXIS_VECTORS[name].copy() for name in arm["joint_axes"]]
    offsets = [
        np.array([0.0, 0.0, float(arm["base_column_mm"]) / MM_PER_M]),
        np.zeros(3),                                    # j2 sits at j1's origin
        np.array([float(lengths["upper_arm"]) / MM_PER_M, 0.0, 0.0]),
        np.array([float(lengths["forearm"]) / MM_PER_M, 0.0, 0.0]),
        np.array([float(lengths["wrist_roll"]) / MM_PER_M, 0.0, 0.0]),
        np.array([float(lengths["wrist_pitch"]) / MM_PER_M, 0.0, 0.0]),
        np.array([float(lengths["tool"]) / MM_PER_M, 0.0, 0.0]),
    ]
    limits = arm["joint_limits_deg"]
    lower = np.array([limits[f"j{i}"]["min"] for i in range(1, 7)], dtype=float)
    upper = np.array([limits[f"j{i}"]["max"] for i in range(1, 7)], dtype=float)
    return ArmChain(
        arm_id=arm["id"],
        base_xyz_m=np.array([v / MM_PER_M for v in mount["xyz_mm"]]),
        base_rpy_rad=np.radians(np.array(mount["rpy_deg"], dtype=float)),
        column_m=float(arm["base_column_mm"]) / MM_PER_M,
        axes=axes,
        offsets=offsets,
        lower_deg=lower,
        upper_deg=upper,
    )


def _rot(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation matrix about a unit axis."""
    x, y, z = axis
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def _rpy_matrix(rpy_rad: Sequence[float]) -> np.ndarray:
    """URDF fixed-joint convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = rpy_rad
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def forward_kinematics(chain: ArmChain, q_deg: Sequence[float]) -> np.ndarray:
    """Return the 4x4 tip pose (metres) for joint angles in degrees."""
    q = np.radians(np.asarray(q_deg, dtype=float))
    rotation = _rpy_matrix(chain.base_rpy_rad)
    position = chain.base_xyz_m.copy()

    def apply_translation(offset: np.ndarray) -> None:
        nonlocal position
        position = position + rotation @ offset

    def apply_joint(index: int) -> None:
        nonlocal rotation
        rotation = rotation @ _rot(chain.axes[index], q[index])

    apply_translation(chain.offsets[0])   # up the base column
    apply_joint(0)                        # j1
    apply_joint(1)                        # j2 (same origin)
    apply_translation(chain.offsets[2])   # upper arm
    apply_joint(2)                        # j3
    apply_translation(chain.offsets[3])   # forearm
    apply_joint(3)                        # j4
    apply_translation(chain.offsets[4])   # wrist roll link
    apply_joint(4)                        # j5
    apply_translation(chain.offsets[5])   # wrist pitch link
    apply_joint(5)                        # j6
    apply_translation(chain.offsets[6])   # tool -> tip
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = position
    return pose


def tip_position_mm(chain: ArmChain, q_deg: Sequence[float]) -> np.ndarray:
    """Tip position in millimetres (config units) for convenience."""
    return forward_kinematics(chain, q_deg)[:3, 3] * MM_PER_M


def position_jacobian(
    chain: ArmChain, q_deg: Sequence[float], delta_deg: float = 1e-3
) -> np.ndarray:
    """Numeric 3x6 Jacobian of the tip position w.r.t. joint angles (deg)."""
    base_q = np.asarray(q_deg, dtype=float)
    jacobian = np.zeros((3, 6))
    for i in range(6):
        q_plus = base_q.copy()
        q_minus = base_q.copy()
        q_plus[i] += delta_deg
        q_minus[i] -= delta_deg
        jacobian[:, i] = (
            forward_kinematics(chain, q_plus)[:3, 3]
            - forward_kinematics(chain, q_minus)[:3, 3]
        ) / (2.0 * delta_deg)
    return jacobian


DEFAULT_IK_TOLERANCE_MM = 0.01
DEFAULT_IK_MAX_ITERATIONS = 600
DEFAULT_IK_DAMPING = 1e-6
DEFAULT_IK_MAX_STEP_DEG = 10.0
DEFAULT_IK_RESTART_SEEDS = [
    (0.0, -45.0, 60.0, 0.0, 30.0, 0.0),
    (45.0, -30.0, 45.0, -30.0, 20.0, 15.0),
    (-45.0, -60.0, 90.0, 30.0, -20.0, -15.0),
    (90.0, 0.0, 30.0, 0.0, 0.0, 0.0),
]


def inverse_kinematics(
    chain: ArmChain,
    target_xyz_m: Sequence[float],
    q0_deg: Sequence[float] | None = None,
    tolerance_mm: float = DEFAULT_IK_TOLERANCE_MM,
    max_iterations: int = DEFAULT_IK_MAX_ITERATIONS,
    damping: float = DEFAULT_IK_DAMPING,
) -> tuple[np.ndarray, bool, float]:
    """Damped-least-squares position IK.

    Returns ``(q_deg, converged, final_error_mm)``. Starts from ``q0_deg``
    when given, otherwise from a small set of workspace seed poses; joint
    limits are enforced every step.
    """
    if q0_deg is not None:
        starts = [np.asarray(q0_deg, dtype=float)]
    else:
        starts = [np.zeros(6)]
        starts.extend(
            np.clip(np.array(seed, dtype=float), chain.lower_deg, chain.upper_deg)
            for seed in DEFAULT_IK_RESTART_SEEDS
        )
    best_q = starts[0]
    best_error = float("inf")
    for start in starts:
        q, converged, error_mm = _ik_from_start(
            chain, target_xyz_m, start, tolerance_mm, max_iterations, damping
        )
        if error_mm < best_error:
            best_q, best_error = q, error_mm
        if converged:
            return q, True, error_mm
    return best_q, False, best_error


def _ik_from_start(
    chain: ArmChain,
    target_xyz_m: Sequence[float],
    q_start: np.ndarray,
    tolerance_mm: float,
    max_iterations: int,
    damping: float,
) -> tuple[np.ndarray, bool, float]:
    q = np.clip(np.array(q_start, dtype=float), chain.lower_deg, chain.upper_deg)
    target = np.asarray(target_xyz_m, dtype=float)
    tolerance_m = tolerance_mm / MM_PER_M
    for _ in range(max_iterations):
        error = target - forward_kinematics(chain, q)[:3, 3]
        error_norm = float(np.linalg.norm(error))
        if error_norm <= tolerance_m:
            return q, True, error_norm * MM_PER_M
        jac = position_jacobian(chain, q)
        # Damping dominated by the smallest singular value direction.
        lambda_sq = damping + 1e-6 * error_norm
        step = jac.T @ np.linalg.solve(
            jac @ jac.T + lambda_sq * np.eye(3), error
        )
        step = np.clip(step, -DEFAULT_IK_MAX_STEP_DEG, DEFAULT_IK_MAX_STEP_DEG)
        q = np.clip(q + step, chain.lower_deg, chain.upper_deg)
    final_error = float(np.linalg.norm(
        target - forward_kinematics(chain, q)[:3, 3]) * MM_PER_M)
    return q, False, final_error


def within_limits(chain: ArmChain, q_deg: Sequence[float], slack_deg: float = 0.0) -> bool:
    """True when every joint angle lies inside its configured limits."""
    q = np.asarray(q_deg, dtype=float)
    return bool(
        np.all(q >= chain.lower_deg - slack_deg)
        and np.all(q <= chain.upper_deg + slack_deg)
    )
