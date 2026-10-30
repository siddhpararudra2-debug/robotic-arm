"""Stage-0 acceptance tests for config, URDF generation and mesh fallback.

Run from the project folder:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import copy
import math
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from neurosim.config import (
    ConfigError,
    config_summary,
    load_config,
    resolve_arm_joints,
    validate_config,
)
from neurosim.urdf_generator import generate_urdf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "robot_config.yaml"

try:
    import pybullet as p

    PY_BULLET_AVAILABLE = True
except ImportError:
    PY_BULLET_AVAILABLE = False


class Stage0ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config(CONFIG_PATH)

    def test_parts_block_is_captured(self) -> None:
        cfg = self.cfg
        arms = {arm["id"]: arm for arm in cfg["arms"]}
        self.assertEqual(len(arms), 4)
        self.assertEqual(
            len([j for arm in cfg["arms"] for j in resolve_arm_joints(cfg, arm)]),
            24,
        )

        self.assertEqual(arms["A1"]["reach_mm"], 450)
        self.assertEqual(arms["A1"]["payload_kg"], 1.0)
        for arm_id in ("A2", "A3"):
            self.assertEqual(arms[arm_id]["reach_mm"], 250)
            self.assertEqual(arms[arm_id]["payload_kg"], 0.3)
            self.assertEqual(arms[arm_id]["wrist"]["type"], "tendon_driven")
            self.assertIsNotNone(arms[arm_id]["fine_stage"])
        self.assertEqual(arms["A4"]["reach_mm"], 350)
        self.assertEqual(arms["A4"]["payload_kg"], 0.3)
        self.assertIn("camera", arms["A4"]["end_effector"])
        self.assertIn("laser", arms["A4"]["end_effector"])
        self.assertIn("force_sensor", arms["A1"])

        a1_joints = resolve_arm_joints(cfg, arms["A1"])
        self.assertTrue(all(j["gear_id"] == "CSF-17-100" for j in a1_joints[:3]))
        self.assertTrue(all(j["gear_id"] == "CSF-14-100" for j in a1_joints[3:]))
        for arm_id in ("A2", "A3"):
            self.assertTrue(
                all(j["gear_id"] == "CSF-11-100" for j in resolve_arm_joints(cfg, arms[arm_id]))
            )
        self.assertTrue(
            all(j["gear_id"] == "CSF-14-100" for j in resolve_arm_joints(cfg, arms["A4"]))
        )

        for arm in cfg["arms"]:
            for joint in resolve_arm_joints(cfg, arm):
                self.assertEqual(joint["brake"]["type"], "power-off electromagnetic")
                self.assertEqual(joint["joint_encoder"]["bits"], 14)
                self.assertEqual(joint["joint_encoder"]["counts_per_rev"], 16384)
                self.assertEqual(joint["motor_encoder"]["counts_per_rev"], 16384)
                self.assertLessEqual(joint["backlash_deg"] * 60.0, 1.0)

        world = cfg["world"]
        self.assertEqual(world["control_loop_hz"], 1000)
        self.assertEqual(world["vision_loop_options_hz"], [30, 60])
        self.assertEqual(world["bus"], "CAN-FD")
        self.assertTrue(any("EPOS4" in item for item in world["driver_options"]))
        self.assertTrue(any("ODrive S1" in item for item in world["driver_options"]))
        self.assertIn("PREEMPT_RT", world["controller"]["kernel"])
        self.assertIn("PNOZ", world["safety"]["safety_relay_class"])

        components = cfg["components"]
        self.assertEqual(
            components["prototype_alternative"]["encoder_counts_per_rev"], 4096
        )
        laser = components["lasers"]["laser_650nm_dot"]
        self.assertEqual(laser["wavelength_nm"], 650)
        self.assertEqual(laser["safety_class"], 2)
        self.assertLess(laser["max_power_mw"], 1.0)
        self.assertEqual(laser["camera_filter_bandpass_nm"], 10)
        camera = components["cameras"]["intel_realsense_d405"]
        self.assertEqual(camera["depth_range_mm"], [70, 500])
        self.assertEqual(camera["depth_resolution_px"], [1280, 720])
        self.assertEqual(components["fine_stages"]["pi_p611_nanocube_class"]["travel_um"], 100)
        self.assertIn("ati_nano17", components["force_sensor"])
        self.assertEqual(
            components["tracking"]["ndi_polaris_vega_st"]["accuracy_mm"], 0.12
        )
        leader = components["leader_handles"]
        self.assertEqual(leader["motion_scale_range"], [1.0, 20.0])
        self.assertIn("3D Systems Touch", leader["prototype"])
        self.assertIn("sigma.7", leader["alternative"])

        instrument = cfg["instrument"]
        self.assertEqual(instrument["shaft_diameter_mm"], 5.0)
        self.assertIn("stainless_steel", instrument["material_options"])
        self.assertIn("polymer_3d_printed", instrument["material_options"])

        budget = cfg["error_budget"]
        self.assertLessEqual(budget["backlash"]["max_arcmin"], 1.0)
        self.assertEqual(budget["encoder_quantisation"]["bits"], 14)
        self.assertEqual(budget["calibration_offset_mm"], 0.2)
        self.assertEqual(budget["tissue_shift"]["max_mm"], 1.0)
        self.assertEqual(cfg["performance_targets_mm"]["A1"], 0.5)
        self.assertEqual(cfg["performance_targets_mm"]["A2"], 0.1)
        self.assertEqual(cfg["performance_targets_mm"]["A3"], 0.1)

    def test_geometry_and_home_poses_are_consistent(self) -> None:
        for arm in self.cfg["arms"]:
            lengths = arm["geometry"]["link_lengths_mm"]
            total = sum(
                lengths[key]
                for key in ("upper_arm", "forearm", "wrist_roll", "wrist_pitch", "tool")
            )
            self.assertAlmostEqual(total, arm["reach_mm"], places=6)
            for joint in resolve_arm_joints(self.cfg, arm):
                self.assertLessEqual(joint["lower_deg"], joint["home_deg"])
                self.assertLessEqual(joint["home_deg"], joint["upper_deg"])

    def test_derived_joint_values(self) -> None:
        arm = next(arm for arm in self.cfg["arms"] if arm["id"] == "A1")
        j1 = resolve_arm_joints(self.cfg, arm)[0]
        self.assertEqual(j1["ratio"], 100.0)
        self.assertAlmostEqual(j1["backlash_deg"], 1.0 / 60.0)
        self.assertAlmostEqual(j1["joint_encoder_lsb_deg"], 360.0 / 16384.0)
        self.assertAlmostEqual(
            j1["motor_encoder_lsb_at_output_deg"], 360.0 / (16384.0 * 100.0)
        )
        self.assertAlmostEqual(
            j1["peak_output_torque_nm"],
            j1["motor"]["peak_torque_nm"] * j1["ratio"] * j1["gear"]["efficiency"],
        )
        summary = config_summary(self.cfg)
        self.assertEqual(summary["arm_count"], 4)
        self.assertEqual(summary["total_axes"], 24)
        self.assertTrue(summary["all_joints_braked"])
        self.assertAlmostEqual(
            summary["backlash_rad"], (1.0 / 60.0) * math.pi / 180.0
        )

    def test_bad_configs_are_rejected(self) -> None:
        broken = copy.deepcopy(self.cfg)
        broken["arms"][0]["geometry"]["link_lengths_mm"]["tool"] += 1.0
        with self.assertRaises(ConfigError):
            validate_config(broken)

        broken = copy.deepcopy(self.cfg)
        broken["components"]["encoders"]["as5047p_joint_side"]["counts_per_rev"] = 4096
        broken["components"]["encoders"]["as5047p_joint_side"]["bits"] = 14
        broken["error_budget"]["encoder_quantisation"]["counts_per_rev"] = 4096
        with self.assertRaises(ConfigError):
            validate_config(broken)

        broken = copy.deepcopy(self.cfg)
        broken["arms"][0]["home_deg"][0] = 999.0
        with self.assertRaises(ConfigError):
            validate_config(broken)


class Stage0UrdfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config(CONFIG_PATH)

    def _generate(
        self,
        cfg: dict | None = None,
        mesh_files: dict[str, str] | None = None,
        collision_from_mesh: bool | None = None,
    ) -> tuple[Path, dict, ET.Element]:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root_dir = Path(temp.name)
        mesh_dir = root_dir / "meshes"
        mesh_dir.mkdir()
        for filename, content in (mesh_files or {}).items():
            (mesh_dir / filename).write_text(content, encoding="utf-8")

        used_cfg = copy.deepcopy(cfg or self.cfg)
        if collision_from_mesh is not None:
            used_cfg["cad_meshes"]["collision_from_mesh"] = collision_from_mesh
        out = root_dir / "urdf" / "robot_cell.urdf"
        report = generate_urdf(used_cfg, out, mesh_dir)
        xml_root = ET.parse(out).getroot()
        return out, report, xml_root

    @staticmethod
    def _joints(root: ET.Element) -> dict[str, ET.Element]:
        return {joint.attrib["name"]: joint for joint in root.findall("joint")}

    @staticmethod
    def _links(root: ET.Element) -> dict[str, ET.Element]:
        return {link.attrib["name"]: link for link in root.findall("link")}

    @staticmethod
    def _origin_xyz(joint: ET.Element) -> list[float]:
        return [float(v) for v in joint.find("origin").attrib["xyz"].split()]

    def test_structure_is_24_axes_in_one_world(self) -> None:
        out, report, root = self._generate()
        self.assertTrue(out.is_file())
        self.assertEqual(root.tag, "robot")

        links = self._links(root)
        joints = self._joints(root)
        self.assertIn("world", links)
        self.assertIn("A1_tip", links)
        self.assertIn("A4_tip", links)
        self.assertEqual(report["links"], 33)
        self.assertEqual(report["revolute_joints"], 24)
        self.assertEqual(report["links"], len(links))

        revolute = [j for j in joints.values() if j.attrib["type"] == "revolute"]
        fixed = [j for j in joints.values() if j.attrib["type"] == "fixed"]
        self.assertEqual(len(revolute), 24)
        self.assertEqual(len(fixed), 8)
        self.assertEqual(len(revolute) + len(fixed), 32)

        for arm_id in ("A1", "A2", "A3", "A4"):
            for index in range(1, 7):
                self.assertIn(f"{arm_id}_j{index}", joints)
                self.assertIn(f"{arm_id}_link{index}", links)
            mount = joints[f"{arm_id}_mount"]
            self.assertEqual(mount.attrib["type"], "fixed")
            self.assertEqual(mount.find("parent").attrib["link"], "world")
            self.assertEqual(mount.find("child").attrib["link"], f"{arm_id}_base")

        for joint in revolute:
            self.assertIsNotNone(joint.find("parent"))
            self.assertIsNotNone(joint.find("child"))
            self.assertIsNotNone(joint.find("axis"))
            limit = joint.find("limit")
            self.assertIsNotNone(limit)
            self.assertGreater(float(limit.attrib["effort"]), 0.0)
            self.assertGreater(float(limit.attrib["velocity"]), 0.0)
            self.assertLess(float(limit.attrib["lower"]), float(limit.attrib["upper"]))

    def test_chain_and_config_values_reach_the_urdf(self) -> None:
        _out, _report, root = self._generate()
        joints = self._joints(root)
        names = [
            "A1_j1", "A1_j2", "A1_j3", "A1_j4", "A1_j5", "A1_j6",
        ]
        parents = [
            "A1_base", "A1_link1", "A1_link2", "A1_link3", "A1_link4", "A1_link5",
        ]
        self.assertEqual(
            [joints[name].find("parent").attrib["link"] for name in names], parents
        )
        self.assertEqual(
            [joints[name].find("child").attrib["link"] for name in names],
            [f"A1_link{i}" for i in range(1, 7)],
        )

        a1_lengths = self.cfg["arms"][0]["geometry"]["link_lengths_mm"]
        self.assertAlmostEqual(self._origin_xyz(joints["A1_j1"])[2],
                               self.cfg["arms"][0]["base_column_mm"] / 1000.0)
        self.assertAlmostEqual(self._origin_xyz(joints["A1_j2"])[0], 0.0)
        self.assertAlmostEqual(self._origin_xyz(joints["A1_j3"])[0],
                               a1_lengths["upper_arm"] / 1000.0)
        self.assertAlmostEqual(self._origin_xyz(joints["A1_j4"])[0],
                               a1_lengths["forearm"] / 1000.0)
        self.assertAlmostEqual(self._origin_xyz(joints["A1_j5"])[0],
                               a1_lengths["wrist_roll"] / 1000.0)
        self.assertAlmostEqual(self._origin_xyz(joints["A1_j6"])[0],
                               a1_lengths["wrist_pitch"] / 1000.0)
        self.assertAlmostEqual(self._origin_xyz(joints["A1_tip_joint"])[0],
                               a1_lengths["tool"] / 1000.0)

        limit = joints["A1_j1"].find("limit")
        self.assertAlmostEqual(
            float(limit.attrib["upper"]),
            self.cfg["arms"][0]["joint_limits_deg"]["j1"]["max"] * math.pi / 180.0,
        )
        axis = joints["A1_j1"].find("axis").attrib["xyz"]
        self.assertEqual(axis, "0 0 1")

    def test_primitive_parts_and_fallbacks_without_stl(self) -> None:
        _out, report, root = self._generate()
        self.assertEqual(len(report["primitive_fallbacks"]), 28)
        self.assertEqual(report["meshes_used"], [])
        self.assertEqual(len(root.findall(".//mesh")), 0)
        self.assertGreater(report["total_mass_kg"], 0.0)
        self.assertAlmostEqual(
            report["total_mass_kg"], sum(report["arm_masses_kg"].values()), places=3
        )

        links = self._links(root)
        for arm_id in ("A1", "A2", "A3", "A4"):
            for link_id in ("base", "link1", "link2", "link3", "link4", "link5", "link6"):
                self.assertGreater(
                    len(links[f"{arm_id}_{link_id}"].findall("visual")), 0,
                    msg=f"{arm_id}_{link_id} has no visual",
                )

        for arm_id in ("A1", "A2", "A3", "A4"):
            link6 = links[f"{arm_id}_link6"]
            visual_names = {v.attrib.get("name") for v in link6.findall("visual")}
            self.assertIn("instrument_shaft", visual_names)
        for arm_id in ("A2", "A3"):
            self.assertIn(
                "piezo_fine_stage",
                {v.attrib.get("name") for v in links[f"{arm_id}_link6"].findall("visual")},
            )
        self.assertIn(
            "force_torque_sensor",
            {v.attrib.get("name") for v in links["A1_link6"].findall("visual")},
        )
        a4_names = {v.attrib.get("name") for v in links["A4_link6"].findall("visual")}
        self.assertIn("realsense_d405", a4_names)
        self.assertIn("laser_module", a4_names)

        all_visual_names = {
            v.attrib.get("name")
            for link in links.values()
            for v in link.findall("visual")
        }
        for arm_id in ("A1", "A2", "A3", "A4"):
            for index in range(1, 7):
                self.assertIn(f"{arm_id}_j{index}_harmonic_drive", all_visual_names)
                self.assertIn(f"{arm_id}_j{index}_motor", all_visual_names)

    def test_stl_replaces_visuals_but_not_collisions_by_default(self) -> None:
        stl = (
            "solid test\n"
            "facet normal 0 0 1\n"
            " outer loop\n"
            "  vertex 0 0 0\n"
            "  vertex 1 0 0\n"
            "  vertex 0 1 0\n"
            " endloop\n"
            "endfacet\n"
            "endsolid test\n"
        )
        _out, report, root = self._generate(mesh_files={"A1_link2.stl": stl})
        self.assertIn("A1_link2", report["meshes_used"])
        self.assertEqual(len(report["meshes_used"]), 1)
        self.assertEqual(len(report["primitive_fallbacks"]), 27)
        self.assertEqual(len(root.findall(".//mesh")), 1)

        link = self._links(root)["A1_link2"]
        mesh = link.find("./visual/geometry/mesh")
        self.assertIsNotNone(mesh)
        self.assertIn("A1_link2.stl", mesh.attrib["filename"])
        self.assertGreater(len(link.findall("collision")), 0)
        self.assertIsNone(link.find("./collision/geometry/mesh"))

    def test_collision_can_come_from_stl(self) -> None:
        stl = "solid test\nendsolid test\n"
        _out, _report, root = self._generate(
            mesh_files={"A1_link2.stl": stl}, collision_from_mesh=True
        )
        link = self._links(root)["A1_link2"]
        self.assertIsNotNone(link.find("./collision/geometry/mesh"))
        collision_geometries = [
            child.tag for collision in link.findall("collision")
            for child in collision.find("geometry")
        ]
        self.assertEqual(collision_geometries, ["mesh"])

    def test_editing_config_changes_urdf(self) -> None:
        cfg = copy.deepcopy(self.cfg)
        cfg["arms"][0]["joint_limits_deg"]["j1"]["max"] = 123.0
        cfg["arms"][0]["geometry"]["tube_outer_diameter_mm"] = 44.0
        cfg["arms"][0]["base_mount"]["xyz_mm"] = [111.0, 222.0, 333.0]
        _out, _report, root = self._generate(cfg=cfg)
        joints = self._joints(root)
        limit = joints["A1_j1"].find("limit")
        self.assertAlmostEqual(float(limit.attrib["upper"]), 123.0 * math.pi / 180.0)
        self.assertEqual(
            self._origin_xyz(joints["A1_mount"]), [0.111, 0.222, 0.333]
        )
        tube_link = self._links(root)["A1_link2"]
        tube = next(
            v for v in tube_link.findall("visual")
            if v.attrib.get("name") == "link_tube"
        )
        radius = float(tube.find("geometry/cylinder").attrib["radius"])
        self.assertAlmostEqual(radius, 0.022)

    @unittest.skipUnless(PY_BULLET_AVAILABLE, "PyBullet is not installed")
    def test_urdf_parses_in_pybullet(self) -> None:
        out, _report, _root = self._generate()
        client = p.connect(p.DIRECT)
        try:
            body = p.loadURDF(str(out), useFixedBase=True)
            self.assertEqual(p.getNumJoints(body), 32)
            revolute = [
                index for index in range(p.getNumJoints(body))
                if p.getJointInfo(body, index)[2] == p.JOINT_REVOLUTE
            ]
            self.assertEqual(len(revolute), 24)
        finally:
            p.disconnect(client)


if __name__ == "__main__":
    unittest.main()
