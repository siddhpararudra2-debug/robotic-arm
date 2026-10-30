"""Stage 3: actuator-level simulation - the 1 kHz joint loop.

Per joint, models (all parameters from ``robot_config.yaml``):

- closed-loop position tracking with the configured velocity limit;
- harmonic-drive backlash as a direction-dependent dead band;
- 14-bit encoder quantisation on the joint side (and motor side);
- the power-off electromagnetic brake (engage/release delay, output frozen).

The simulator is deliberately simple: it is a kinematic loop, not a rigid-body
dynamic model. Stage 8 consumes its measured angles for the accuracy report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from .config import resolve_arm_joints


def quantize(value_deg: float, lsb_deg: float) -> float:
    """Round to the nearest encoder multiple (counts grid around zero)."""
    if lsb_deg <= 0.0:
        return value_deg
    return round(value_deg / lsb_deg) * lsb_deg


@dataclass
class BacklashModel:
    """Direction-dependent dead band between motor side and joint side."""

    half_gap_deg: float
    offset_deg: float = 0.0
    _last_commanded_deg: float | None = None

    def output_deg(self, commanded_deg: float) -> float:
        """Map the motor-side command through the dead band.

        Moving in one direction pushes the offset to one end of the gap; the
        output does not respond until the gap is traversed after a reversal.
        """
        if self._last_commanded_deg is not None:
            delta = commanded_deg - self._last_commanded_deg
            if delta > 0.0:
                # Moving up closes the gap toward the lower lag limit.
                self.offset_deg = max(-self.half_gap_deg, self.offset_deg - delta)
            elif delta < 0.0:
                # Moving down closes the gap toward the upper lag limit.
                self.offset_deg = min(self.half_gap_deg, self.offset_deg - delta)
        self._last_commanded_deg = commanded_deg
        return commanded_deg + self.offset_deg


@dataclass
class JointActuator:
    """One actuated joint: tracking, backlash, encoders and brake."""

    record: dict[str, Any]
    actual_deg: float = 0.0
    commanded_deg: float = 0.0
    backlash: BacklashModel = field(init=False)
    brake_engaged: bool = True
    _brake_timer_s: float = 0.0

    def __post_init__(self) -> None:
        backlash_deg = float(self.record["backlash_deg"])
        self.backlash = BacklashModel(half_gap_deg=backlash_deg / 2.0)
        home = float(self.record["home_deg"])
        self.actual_deg = home
        self.commanded_deg = home
        self.brake_engaged = True

    @property
    def max_speed_deg_s(self) -> float:
        return float(self.record["max_output_speed_deg_s"])

    def engage_brake(self) -> None:
        self.brake_engaged = True

    def release_brake(self) -> None:
        self.brake_engaged = False

    def step(self, dt_s: float, command_deg: float, gain_hz: float,
             max_step_deg: float) -> None:
        """Advance one control-loop tick.

        The brake freezes the output regardless of the command. Otherwise the
        output tracks the command through the backlash dead band with a
        first-order loop and the configured velocity limit.
        """
        self.commanded_deg = float(command_deg)
        if self.brake_engaged:
            return
        # Backlash acts between the motor shaft and the joint output.
        through_backlash = self.backlash.output_deg(self.commanded_deg)
        error_deg = through_backlash - self.actual_deg
        # First-order tracking with a per-tick slew clamp.
        rate = 2.0 * math.pi * gain_hz * error_deg
        rate = max(-self.max_speed_deg_s, min(self.max_speed_deg_s, rate))
        delta = rate * dt_s
        delta = max(-max_step_deg, min(max_step_deg, delta))
        self.actual_deg += delta

    def measured_joint_deg(self) -> float:
        """Joint-side encoder reading (quantised)."""
        encoder = self.record["joint_encoder"]
        return quantize(self.actual_deg, float(self.record["joint_encoder_lsb_deg"]))

    def measured_motor_side_deg(self) -> float:
        """Motor-side encoder reading mapped through the gear ratio."""
        lsb = float(self.record["motor_encoder_lsb_at_output_deg"])
        return quantize(self.actual_deg, lsb)


class ArmActuatorSim:
    """The 1 kHz loop for one arm's six joints."""

    def __init__(self, cfg: dict[str, Any], arm: dict[str, Any]) -> None:
        actuators_cfg = cfg.get("actuators", {})
        self.gain_hz = float(actuators_cfg.get("position_gain_hz", 20.0))
        self.max_step_deg = float(actuators_cfg.get("max_step_deg_per_loop", 0.5))
        brake = cfg["components"]["brakes"][arm["brake"]]
        self.brake_engage_s = float(brake.get("engage_time_ms", 50.0)) / 1000.0
        self.brake_release_s = float(brake.get("release_time_ms", 80.0)) / 1000.0
        self.dt_s = 1.0 / float(cfg["world"]["control_loop_hz"])
        self.joints = [JointActuator(rec) for rec in resolve_arm_joints(cfg, arm)]

    def step(self, commands_deg: Sequence[float]) -> list[float]:
        """Advance one tick; returns the encoder-measured joint angles."""
        if len(commands_deg) != len(self.joints):
            raise ValueError("need one command per joint")
        for joint, command in zip(self.joints, commands_deg):
            joint.step(self.dt_s, command, self.gain_hz, self.max_step_deg)
        return [joint.measured_joint_deg() for joint in self.joints]

    def run(self, commands_deg: Sequence[float], duration_s: float) -> list[float]:
        """Run the loop for a duration; returns the final measured angles."""
        steps = max(1, int(round(duration_s / self.dt_s)))
        measured: list[float] = []
        for _ in range(steps):
            measured = self.step(commands_deg)
        return measured

    def actual_deg(self) -> list[float]:
        return [joint.actual_deg for joint in self.joints]

    def measured_deg(self) -> list[float]:
        return [joint.measured_joint_deg() for joint in self.joints]

    def engage_all_brakes(self) -> None:
        for joint in self.joints:
            joint.engage_brake()

    def release_all_brakes(self) -> None:
        for joint in self.joints:
            joint.release_brake()
