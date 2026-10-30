"""Stage-3 acceptance tests: 1 kHz loop, backlash, encoders and brakes."""

from __future__ import annotations

import unittest
from pathlib import Path

from neurosim.actuators import ArmActuatorSim, BacklashModel, quantize
from neurosim.config import load_config, resolve_arm_joints

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "robot_config.yaml"


class BacklashTests(unittest.TestCase):
    def test_dead_band_blocks_small_reversal(self) -> None:
        model = BacklashModel(half_gap_deg=0.5)
        model.output_deg(10.0)
        model.output_deg(12.0)  # moving up lags by half the gap
        first = model.output_deg(12.0)
        # Reverse by less than the full gap: the output must not move.
        second = model.output_deg(11.8)
        self.assertAlmostEqual(second, first, places=9)
        # Reverse beyond the gap: the output follows again, lagging the other way.
        third = model.output_deg(10.0)
        self.assertAlmostEqual(third, 10.0 + 0.5, places=9)

    def test_offset_stays_inside_half_gap(self) -> None:
        model = BacklashModel(half_gap_deg=0.5)
        for command in range(-60, 60):
            model.output_deg(float(command))
            self.assertLessEqual(abs(model.offset_deg), 0.5 + 1e-12)


class ActuatorSimTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config(CONFIG_PATH)
        cls.arms = {arm["id"]: arm for arm in cls.cfg["arms"]}

    def test_one_simulator_per_arm_with_six_joints(self) -> None:
        for arm_id, arm in self.arms.items():
            sim = ArmActuatorSim(self.cfg, arm)
            self.assertEqual(len(sim.joints), 6, arm_id)
            self.assertAlmostEqual(sim.dt_s, 0.001, places=9)

    def test_encoder_lsb_matches_config(self) -> None:
        cfg = self.cfg
        counts = cfg["error_budget"]["encoder_quantisation"]["counts_per_rev"]
        expected_lsb = 360.0 / counts
        arm = self.arms["A1"]
        sim = ArmActuatorSim(cfg, arm)
        for joint in sim.joints:
            self.assertAlmostEqual(
                float(joint.record["joint_encoder_lsb_deg"]), expected_lsb
            )

    def test_measured_values_are_quantised(self) -> None:
        sim = ArmActuatorSim(self.cfg, self.arms["A1"])
        sim.release_all_brakes()
        measured = sim.run([10.0, -20.0, 30.0, 0.0, 0.0, 0.0], duration_s=2.0)
        lsb = 360.0 / self.cfg["components"]["encoders"]["as5047p_joint_side"][
            "counts_per_rev"
        ]
        for value in measured:
            self.assertAlmostEqual(value / lsb, round(value / lsb), places=9)

    def test_tracks_command_within_backlash(self) -> None:
        sim = ArmActuatorSim(self.cfg, self.arms["A1"])
        sim.release_all_brakes()
        command = [5.0, -30.0, 60.0, 10.0, 20.0, -5.0]
        measured = sim.run(command, duration_s=3.0)
        backlash_deg = (
            self.cfg["error_budget"]["backlash"]["max_arcmin"] / 60.0
        )
        for actual, target in zip(sim.actual_deg(), command):
            self.assertLess(abs(actual - target), backlash_deg + 1e-6)

    def test_velocity_limit_is_respected(self) -> None:
        sim = ArmActuatorSim(self.cfg, self.arms["A1"])
        sim.release_all_brakes()
        arm = self.arms["A1"]
        max_speed = float(arm["max_joint_velocity_deg_s"])
        # Command a large step; the actual rate must never exceed the limit.
        previous = sim.actual_deg()
        for _ in range(500):
            sim.step([120.0] * 6)
            current = sim.actual_deg()
            for prev, curr in zip(previous, current):
                rate = abs(curr - prev) / sim.dt_s
                self.assertLessEqual(rate, max_speed + 1e-6)
            previous = current

    def test_brake_freezes_output(self) -> None:
        sim = ArmActuatorSim(self.cfg, self.arms["A1"])
        sim.release_all_brakes()
        sim.run([20.0] * 6, duration_s=0.5)
        frozen = sim.actual_deg()
        sim.engage_all_brakes()
        sim.run([-40.0] * 6, duration_s=0.5)
        self.assertEqual(sim.actual_deg(), frozen)
        sim.release_all_brakes()
        # 60 deg at the 60 deg/s safety limit needs a full second plus lag.
        sim.run([-40.0] * 6, duration_s=2.0)
        for actual, command in zip(sim.actual_deg(), [-40.0] * 6):
            self.assertLess(abs(actual - command), 1.0)

    def test_new_joints_start_at_home_with_brake_on(self) -> None:
        sim = ArmActuatorSim(self.cfg, self.arms["A2"])
        home = [float(v) for v in self.arms["A2"]["home_deg"]]
        self.assertEqual(sim.actual_deg(), home)
        for joint in sim.joints:
            self.assertTrue(joint.brake_engaged)

    def test_quantize_function(self) -> None:
        lsb = 360.0 / 16384
        self.assertAlmostEqual(quantize(0.4 * lsb, lsb), 0.0, places=12)
        self.assertAlmostEqual(quantize(0.6 * lsb, lsb), lsb, places=12)
        self.assertAlmostEqual(quantize(-0.6 * lsb, lsb), -lsb, places=12)


if __name__ == "__main__":
    unittest.main()
