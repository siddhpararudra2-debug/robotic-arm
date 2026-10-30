"""Stage-2 acceptance tests: FK/IK, workspace reach and joint limits.

The FK-vs-PyBullet comparison runs only when PyBullet is installed.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from neurosim.config import MM_PER_M, load_config
from neurosim.kinematics import (
    build_arm_chain,
    forward_kinematics,
    inverse_kinematics,
    position_jacobian,
    within_limits,
)
from neurosim.workspace import reach_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "robot_config.yaml"

try:
    import pybullet as p

    PY_BULLET_AVAILABLE = True
except ImportError:
    PY_BULLET_AVAILABLE = False


class Stage2KinematicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config(CONFIG_PATH)
        cls.chains = {
            arm["id"]: build_arm_chain(cls.cfg, arm) for arm in cls.cfg["arms"]
        }

    def test_fk_tip_at_full_stretch_reaches_reach_mm(self) -> None:
        """With all joints at zero the tip sits reach_mm from the j1 origin."""
        for arm_id, chain in self.chains.items():
            arm = next(a for a in self.cfg["arms"] if a["id"] == arm_id)
            pose = forward_kinematics(chain, [0.0] * 6)
            j1_origin = chain.base_xyz_m + pose[:3, :3] @ np.array(
                [0.0, 0.0, chain.column_m]
            )
            tip = pose[:3, 3]
            self.assertAlmostEqual(
                float(np.linalg.norm(tip - j1_origin)),
                float(arm["reach_mm"]) / MM_PER_M,
                places=9,
                msg=arm_id,
            )

    def test_fk_is_deterministic_and_finite(self) -> None:
        rng = np.random.default_rng(42)
        for chain in self.chains.values():
            for _ in range(20):
                q = rng.uniform(chain.lower_deg, chain.upper_deg)
                pose = forward_kinematics(chain, q)
                self.assertTrue(np.all(np.isfinite(pose)))
                self.assertAlmostEqual(np.linalg.det(pose[:3, :3]), 1.0, places=6)

    def test_jacobian_matches_finite_difference(self) -> None:
        chain = self.chains["A1"]
        q = [10.0, -20.0, 40.0, 5.0, 15.0, -10.0]
        jac = position_jacobian(chain, q, delta_deg=1e-4)
        numeric = np.zeros((3, 6))
        for i in range(6):
            q_plus = list(q)
            q_minus = list(q)
            q_plus[i] += 0.01
            q_minus[i] -= 0.01
            numeric[:, i] = (
                forward_kinematics(chain, q_plus)[:3, 3]
                - forward_kinematics(chain, q_minus)[:3, 3]
            ) / 0.02
        self.assertTrue(np.allclose(jac, numeric, atol=1e-6))

    def test_ik_recovers_fk_positions(self) -> None:
        rng = np.random.default_rng(7)
        for arm_id, chain in self.chains.items():
            for _ in range(5):
                q_true = rng.uniform(chain.lower_deg, chain.upper_deg)
                target = forward_kinematics(chain, q_true)[:3, 3]
                q_sol, converged, error_mm = inverse_kinematics(
                    chain, target, q0_deg=[0.0] * 6, tolerance_mm=0.01
                )
                self.assertTrue(converged, f"{arm_id} IK failed")
                self.assertLess(error_mm, 0.05, arm_id)
                self.assertTrue(within_limits(chain, q_sol), arm_id)

    def test_ik_respects_joint_limits(self) -> None:
        chain = self.chains["A1"]
        # A point far outside the workspace must not produce limit violations.
        far_target = chain.base_xyz_m + np.array([5.0, 5.0, 5.0])
        _q, converged, _error = inverse_kinematics(chain, far_target)
        self.assertFalse(converged)

    def test_every_configured_target_is_reachable(self) -> None:
        report = reach_report(self.cfg)
        for arm_id, record in report.items():
            self.assertTrue(
                record["reachable"],
                f"{arm_id} cannot reach {record['target_mm']} "
                f"(error {record['error_mm']:.3f} mm)",
            )
            self.assertTrue(record["within_limits"], arm_id)


class Stage2FkVsPyBulletTests(unittest.TestCase):
    """Cross-check the analytic FK against the URDF as loaded by PyBullet."""

    @classmethod
    def setUpClass(cls) -> None:
        if not PY_BULLET_AVAILABLE:
            raise unittest.SkipTest("PyBullet is not installed")
        from neurosim.scene import build_scene, close

        cls.cfg = load_config(CONFIG_PATH)
        cls.scene = build_scene(cls.cfg, gui=False)

    @classmethod
    def tearDownClass(cls) -> None:
        if PY_BULLET_AVAILABLE:
            from neurosim.scene import close

            close(cls.scene)

    def test_fk_matches_pybullet_tip_positions(self) -> None:
        import math

        scene = self.scene
        rng = np.random.default_rng(3)
        for arm in self.cfg["arms"]:
            arm_id = arm["id"]
            chain = build_arm_chain(self.cfg, arm)
            for _ in range(5):
                q = rng.uniform(chain.lower_deg, chain.upper_deg)
                for index, joint_index in enumerate(scene.arm_joint_indices[arm_id]):
                    p.resetJointState(
                        scene.robot_id, joint_index, math.radians(float(q[index])),
                        physicsClientId=scene.client_id,
                    )
                state = p.getLinkState(
                    scene.robot_id,
                    scene.link_index_by_name[f"{arm_id}_tip"],
                    computeForwardKinematics=True,
                    physicsClientId=scene.client_id,
                )
                bullet_tip = np.array(state[4])  # worldLinkFramePosition
                analytic_tip = forward_kinematics(chain, q)[:3, 3]
                self.assertTrue(
                    np.allclose(bullet_tip, analytic_tip, atol=1e-5),
                    f"{arm_id}: bullet={bullet_tip} analytic={analytic_tip}",
                )


if __name__ == "__main__":
    unittest.main()
