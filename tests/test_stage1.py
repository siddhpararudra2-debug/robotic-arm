"""Stage-1 acceptance tests for the PyBullet scene assembly.

Run from the project folder:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

from neurosim.config import MM_PER_M, load_config
from neurosim.scene import (
    Scene,
    arm_arm_contacts,
    build_scene,
    close,
    contacts_with_body,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "robot_config.yaml"

try:
    import pybullet as p

    PY_BULLET_AVAILABLE = True
except ImportError:
    PY_BULLET_AVAILABLE = False


@unittest.skipUnless(PY_BULLET_AVAILABLE, "PyBullet is not installed")
class Stage1SceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config(CONFIG_PATH)
        cls.scene: Scene = build_scene(cls.cfg, gui=False)

    @classmethod
    def tearDownClass(cls) -> None:
        close(cls.scene)

    def test_scene_bodies_exist(self) -> None:
        scene = self.scene
        self.assertGreaterEqual(scene.robot_id, 0)
        self.assertGreaterEqual(scene.head_id, 0)
        self.assertGreaterEqual(scene.table_id, 0)
        clamp = self.cfg["phantom"]["head_clamp"]
        expected_clamp = int(clamp["post_count"]) + int(clamp["segment_count"])
        self.assertEqual(len(scene.clamp_ids), expected_clamp)

    def test_all_24_revolute_joints_are_found(self) -> None:
        scene = self.scene
        self.assertEqual(set(scene.arm_joint_indices), {"A1", "A2", "A3", "A4"})
        for arm_id, indices in scene.arm_joint_indices.items():
            self.assertEqual(len(indices), 6, arm_id)
            self.assertEqual(
                len({p.getJointInfo(scene.robot_id, i)[0] for i in indices}), 6
            )

    def test_arms_rest_at_home(self) -> None:
        scene = self.scene
        for arm in self.cfg["arms"]:
            for index, joint_index in enumerate(scene.arm_joint_indices[arm["id"]]):
                state = p.getJointState(
                    scene.robot_id, joint_index, physicsClientId=scene.client_id
                )
                self.assertAlmostEqual(
                    state[0],
                    math.radians(float(arm["home_deg"][index])),
                    places=6,
                    msg=f"{arm['id']} joint j{index + 1} not at home",
                )

    def test_head_dimensions_match_config(self) -> None:
        scene = self.scene
        radii_mm = self.cfg["phantom"]["head_radii_mm"]
        aabb_min, aabb_max = p.getAABB(
            scene.head_id, -1, physicsClientId=scene.client_id
        )
        size_mm = [(aabb_max[i] - aabb_min[i]) * MM_PER_M for i in range(3)]
        for measured, expected in zip(size_mm, [2.0 * r for r in radii_mm]):
            self.assertAlmostEqual(measured, expected, delta=expected * 0.02)

    def test_table_top_matches_config(self) -> None:
        scene = self.scene
        table = self.cfg["phantom"]["table"]
        top_z = float(table["top_surface_z_mm"]) / MM_PER_M
        aabb_min, aabb_max = p.getAABB(
            scene.table_id, -1, physicsClientId=scene.client_id
        )
        self.assertAlmostEqual(aabb_max[2], top_z, places=6)
        size_x = (aabb_max[0] - aabb_min[0]) * MM_PER_M
        size_y = (aabb_max[1] - aabb_min[1]) * MM_PER_M
        self.assertAlmostEqual(size_x, float(table["top_size_mm"][0]), delta=1.0)
        self.assertAlmostEqual(size_y, float(table["top_size_mm"][1]), delta=1.0)

    def test_head_is_inside_clamp_ring(self) -> None:
        scene = self.scene
        clamp = self.cfg["phantom"]["head_clamp"]
        head_min, head_max = p.getAABB(
            scene.head_id, -1, physicsClientId=scene.client_id
        )
        ring_radius = float(clamp["ring_radius_mm"]) / MM_PER_M
        # The ring is wider than the head in both horizontal directions.
        self.assertLess(abs(head_min[0]), ring_radius)
        self.assertLess(abs(head_max[0]), ring_radius)
        self.assertLess(abs(head_min[1]), ring_radius)
        self.assertLess(abs(head_max[1]), ring_radius)

    def test_no_arm_arm_collision_at_home(self) -> None:
        contacts = arm_arm_contacts(self.scene)
        self.assertEqual(contacts, [], f"arms touch at home: {contacts}")

    def test_no_collision_with_clamp_or_table_at_home(self) -> None:
        scene = self.scene
        for body_id in scene.clamp_ids + [scene.table_id]:
            touching = contacts_with_body(scene, body_id)
            self.assertEqual(
                touching, [], f"robot touches clamp/table body {body_id}: {touching}"
            )

    def test_head_contacts_involve_tools_only(self) -> None:
        """Instrument tips may rest near the phantom; arms may not hit it."""
        scene = self.scene
        touching = contacts_with_body(scene, scene.head_id)
        for link_name in touching:
            self.assertRegex(
                link_name,
                r"^(A[1-4]_(link6|tip)|A[1-4]_link5)$",
                f"{link_name} penetrates the phantom head",
            )


if __name__ == "__main__":
    unittest.main()
