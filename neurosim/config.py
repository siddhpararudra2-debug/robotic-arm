"""Load and validate ``robot_config.yaml``.

This module is deliberately small: it does NOT contain dimensions, limits or
part numbers. It only reads the YAML and derives a few values (unit
conversions, encoder resolution, torque after the gearbox). Every later stage
imports the same config so the simulation cannot silently drift from the spec.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when robot_config.yaml is missing data or is inconsistent."""


# Unit conversions (not machine specifications).
MM_PER_M = 1000.0
DEG_PER_RAD = 180.0 / math.pi


def load_config(path: str | Path) -> dict[str, Any]:
    """Read the YAML file, validate it, and return it as a dictionary."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError("Top level of robot_config.yaml must be a mapping.")

    # Private metadata so other modules can find meshes relative to the config.
    data["_config_path"] = str(path.resolve())
    data["_project_root"] = str(path.resolve().parent)

    validate_config(data)
    return data


def _need(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing required key {where}.{key}")
    return mapping[key]


def _need_number(mapping: dict[str, Any], key: str, where: str) -> float:
    value = _need(mapping, key, where)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where}.{key} must be a number, got {value!r}")
    return float(value)


def _need_positive(mapping: dict[str, Any], key: str, where: str) -> float:
    value = _need_number(mapping, key, where)
    if value <= 0:
        raise ConfigError(f"{where}.{key} must be > 0, got {value}")
    return value


def validate_config(cfg: dict[str, Any]) -> None:
    """Check structure and cross-references. Fail loudly with a useful path."""
    world = _need(cfg, "world", "config")
    for key in ("control_loop_hz", "vision_loop_hz"):
        _need_positive(world, key, "world")
    if world["vision_loop_hz"] not in world.get("vision_loop_options_hz", []):
        raise ConfigError(
            "world.vision_loop_hz must be one of world.vision_loop_options_hz"
        )
    safety = _need(world, "safety", "world")
    _need_positive(safety, "watchdog_timeout_ms", "world.safety")

    components = _need(cfg, "components", "config")
    for group in ("harmonic_drives", "motors", "brakes", "encoders"):
        _need(components, group, "components")

    encoders = components["encoders"]
    for enc_id, enc in encoders.items():
        bits = int(_need_number(enc, "bits", f"components.encoders.{enc_id}"))
        counts = int(_need_number(enc, "counts_per_rev", f"components.encoders.{enc_id}"))
        if counts != 2**bits:
            raise ConfigError(
                f"Encoder {enc_id}: counts_per_rev ({counts}) must equal 2**bits "
                f"({2**bits})."
            )

    # The fine stage travels in micrometres in the PARTS block.
    fine_stages = components.get("fine_stages", {})
    for stage_id, stage in fine_stages.items():
        if "travel_um" not in stage and "travel_mm" not in stage:
            raise ConfigError(f"Fine stage {stage_id} needs travel_um or travel_mm.")
        envelope = stage.get("envelope_xyz_mm", [])
        if len(envelope) != 3 or any(float(v) <= 0 for v in envelope):
            raise ConfigError(f"Fine stage {stage_id} needs envelope_xyz_mm = [x,y,z].")

    instrument = _need(cfg, "instrument", "config")
    _need_positive(instrument, "shaft_diameter_mm", "instrument")
    material_default = _need(instrument, "material_default", "instrument")
    if material_default not in cfg["materials"]:
        raise ConfigError(f"instrument.material_default {material_default!r} is unknown.")
    for material_id in instrument.get("material_options", []):
        if material_id not in cfg["materials"]:
            raise ConfigError(f"Unknown instrument material option {material_id!r}.")

    for camera_id, camera in components.get("cameras", {}).items():
        envelope = _need(camera, "envelope_mm", f"components.cameras.{camera_id}")
        for key in ("depth", "width", "height"):
            _need_positive(envelope, key, f"components.cameras.{camera_id}.envelope_mm")
        depth_range = _need(camera, "depth_range_mm", f"components.cameras.{camera_id}")
        if len(depth_range) != 2 or not 0 < depth_range[0] < depth_range[1]:
            raise ConfigError(
                f"components.cameras.{camera_id}.depth_range_mm must be [near, far]."
            )

    for laser_id, laser in components.get("lasers", {}).items():
        _need_positive(laser, "wavelength_nm", f"components.lasers.{laser_id}")
        _need_positive(laser, "max_power_mw", f"components.lasers.{laser_id}")
        if float(laser["max_power_mw"]) >= 1.0:
            raise ConfigError(
                f"Laser {laser_id}: PARTS requires Class 2 and power under 1 mW."
            )
        _need_positive(
            laser, "camera_filter_bandpass_nm", f"components.lasers.{laser_id}"
        )

    materials = _need(cfg, "materials", "config")
    if not materials:
        raise ConfigError("materials must not be empty.")

    cad = _need(cfg, "cad_meshes", "config")
    if cad.get("enabled") and ("{arm_id}" not in cad["filename_template"]
                              or "{link_id}" not in cad["filename_template"]):
        raise ConfigError(
            "cad_meshes.filename_template must contain {arm_id} and {link_id}."
        )

    error_budget = _need(cfg, "error_budget", "config")
    backlash = _need(error_budget, "backlash", "error_budget")
    if _need_number(backlash, "max_arcmin", "error_budget.backlash") < 0:
        raise ConfigError("error_budget.backlash.max_arcmin must be >= 0.")
    quant = _need(error_budget, "encoder_quantisation", "error_budget")
    quant_encoder = _need(encoders, quant["encoder"], "components.encoders")
    if int(quant["counts_per_rev"]) != int(quant_encoder["counts_per_rev"]):
        raise ConfigError(
            "error_budget.encoder_quantisation counts must match its encoder."
        )
    _need_number(error_budget, "calibration_offset_mm", "error_budget")
    tissue_shift = _need(error_budget, "tissue_shift", "error_budget")
    if _need_number(tissue_shift, "max_mm", "error_budget.tissue_shift") < 0:
        raise ConfigError("error_budget.tissue_shift.max_mm must be >= 0.")

    tracking = components.get("tracking", {}).get("ndi_polaris_vega_st")
    if tracking and _need_number(
        tracking, "accuracy_mm", "components.tracking.ndi_polaris_vega_st"
    ) < 0:
        raise ConfigError("Tracking accuracy must be >= 0.")

    leader = components.get("leader_handles", {})
    if leader:
        scale_range = leader.get("motion_scale_range")
        default_scale = leader.get("default_motion_scale")
        if (not isinstance(scale_range, list) or len(scale_range) != 2
                or not 0 < scale_range[0] <= scale_range[1]):
            raise ConfigError("leader_handles.motion_scale_range must be [min, max].")
        if not scale_range[0] <= default_scale <= scale_range[1]:
            raise ConfigError("default_motion_scale must lie inside motion_scale_range.")

    arms = _need(cfg, "arms", "config")
    if not isinstance(arms, list) or not arms:
        raise ConfigError("arms must be a non-empty list.")

    arm_ids = [arm.get("id") for arm in arms]
    if len(set(arm_ids)) != len(arm_ids):
        raise ConfigError(f"Arm IDs must be unique, got {arm_ids}")

    for arm in arms:
        _validate_arm(cfg, arm, components, materials)

    targets = cfg.get("performance_targets_mm", {})
    for arm_id, target in targets.items():
        if arm_id not in arm_ids:
            raise ConfigError(f"performance_targets_mm references unknown arm {arm_id}.")
        if target is not None and (not isinstance(target, (int, float)) or target <= 0):
            raise ConfigError(f"Target for {arm_id} must be a positive number or null.")

    targets_mm = cfg.get("targets_mm", {})
    for arm_id, xyz in targets_mm.items():
        if arm_id not in arm_ids:
            raise ConfigError(f"targets_mm references unknown arm {arm_id}.")
        if not isinstance(xyz, list) or len(xyz) != 3:
            raise ConfigError(f"targets_mm.{arm_id} must be [x, y, z] in mm.")


def _validate_arm(
    cfg: dict[str, Any],
    arm: dict[str, Any],
    components: dict[str, Any],
    materials: dict[str, Any],
) -> None:
    arm_id = arm.get("id", "<missing-id>")
    where = f"arms[{arm_id}]"

    for key in ("reach_mm", "payload_kg", "base_mount", "geometry",
                "actuator_groups", "encoders", "brake", "joint_axes",
                "joint_limits_deg", "home_deg", "max_joint_velocity_deg_s"):
        _need(arm, key, where)

    reach = _need_positive(arm, "reach_mm", where)
    _need_positive(arm, "payload_kg", where)
    _need_positive(arm, "max_joint_velocity_deg_s", where)

    base = arm["base_mount"]
    if len(base.get("xyz_mm", [])) != 3 or len(base.get("rpy_deg", [])) != 3:
        raise ConfigError(f"{where}.base_mount needs xyz_mm[3] and rpy_deg[3].")

    geometry = arm["geometry"]
    lengths = _need(geometry, "link_lengths_mm", f"{where}.geometry")
    length_keys = ("upper_arm", "forearm", "wrist_roll", "wrist_pitch", "tool")
    for key in length_keys:
        _need_positive(lengths, key, f"{where}.geometry.link_lengths_mm")
    length_sum = sum(float(lengths[key]) for key in length_keys)
    if abs(length_sum - reach) > 0.01:
        raise ConfigError(
            f"{where}: link lengths sum to {length_sum:g} mm but reach_mm is "
            f"{reach:g} mm. Adjust the five link lengths to match PARTS reach."
        )

    _need_positive(geometry, "tube_outer_diameter_mm", f"{where}.geometry")
    _need_positive(geometry, "tube_wall_mm", f"{where}.geometry")
    if geometry["tube_wall_mm"] * 2 >= geometry["tube_outer_diameter_mm"]:
        raise ConfigError(f"{where}: tube wall is thicker than its radius.")

    link_materials = _need(geometry, "link_materials", f"{where}.geometry")
    if len(link_materials) != 6:
        raise ConfigError(f"{where}.geometry.link_materials must have 6 entries.")
    for material_id in link_materials:
        if material_id not in materials:
            raise ConfigError(f"{where}: unknown material {material_id!r}.")

    axes = arm["joint_axes"]
    if len(axes) != 6 or any(axis not in ("x", "y", "z") for axis in axes):
        raise ConfigError(f"{where}.joint_axes must be exactly six of x/y/z.")

    limits = arm["joint_limits_deg"]
    home = arm["home_deg"]
    if len(home) != 6:
        raise ConfigError(f"{where}.home_deg must have 6 entries.")
    for index in range(1, 7):
        name = f"j{index}"
        limit = _need(limits, name, f"{where}.joint_limits_deg")
        low = _need_number(limit, "min", f"{where}.joint_limits_deg.{name}")
        high = _need_number(limit, "max", f"{where}.joint_limits_deg.{name}")
        if low >= high:
            raise ConfigError(f"{where}.{name}: min must be < max.")
        if not low <= float(home[index - 1]) <= high:
            raise ConfigError(f"{where}.{name}: home is outside its limits.")

    if len(limits) != 6:
        raise ConfigError(f"{where}.joint_limits_deg must define j1..j6.")

    groups = arm["actuator_groups"]
    covered: list[int] = []
    for group_index, group in enumerate(groups):
        group_where = f"{where}.actuator_groups[{group_index}]"
        joints = _need(group, "joints", group_where)
        covered.extend(joints)
        hd_id = _need(group, "harmonic_drive", group_where)
        motor_id = _need(group, "motor", group_where)
        if hd_id not in components["harmonic_drives"]:
            raise ConfigError(f"{group_where}: unknown harmonic drive {hd_id!r}.")
        if motor_id not in components["motors"]:
            raise ConfigError(f"{group_where}: unknown motor {motor_id!r}.")
    if sorted(covered) != [1, 2, 3, 4, 5, 6]:
        raise ConfigError(
            f"{where}: actuator_groups must cover joints 1-6 exactly once; "
            f"got {covered}."
        )

    brake_id = arm["brake"]
    if brake_id not in components["brakes"]:
        raise ConfigError(f"{where}: unknown brake {brake_id!r}.")
    if not cfg["world"]["safety"].get("brake_on_every_joint"):
        raise ConfigError("This project requires a brake on every joint.")

    for side in ("motor_side", "joint_side"):
        enc_id = _need(arm["encoders"], side, f"{where}.encoders")
        if enc_id not in components["encoders"]:
            raise ConfigError(f"{where}: unknown encoder {enc_id!r}.")
        if components["encoders"][enc_id].get("position") != side.replace("_", " "):
            raise ConfigError(
                f"{where}.encoders.{side} must reference an encoder whose "
                f"position is {side.replace('_', ' ')!r}."
            )

    fine_stage = arm.get("fine_stage")
    if fine_stage and fine_stage.get("component") not in components.get("fine_stages", {}):
        raise ConfigError(f"{where}: unknown fine stage {fine_stage.get('component')!r}.")

    force_sensor = arm.get("force_sensor")
    if force_sensor and force_sensor.get("component") not in components.get("force_sensor", {}):
        raise ConfigError(f"{where}: unknown force sensor {force_sensor.get('component')!r}.")

    end_effector = arm.get("end_effector")
    if end_effector:
        camera = end_effector.get("camera", {})
        laser = end_effector.get("laser", {})
        if camera and camera.get("component") not in components.get("cameras", {}):
            raise ConfigError(f"{where}: unknown camera {camera.get('component')!r}.")
        if laser and laser.get("component") not in components.get("lasers", {}):
            raise ConfigError(f"{where}: unknown laser {laser.get('component')!r}.")
        if len(camera.get("mount_offset_mm", [])) != 3:
            raise ConfigError(f"{where}: camera mount_offset_mm must be [x, y, z].")
        if len(laser.get("mount_offset_mm", [])) != 3:
            raise ConfigError(f"{where}: laser mount_offset_mm must be [x, y, z].")


def resolve_arm_joints(cfg: dict[str, Any], arm: dict[str, Any]) -> list[dict[str, Any]]:
    """Return six resolved joint records for one arm.

    Each record flattens the YAML references so later stages do not have to
    re-read the component catalogue: gear ratio, motor limits, brake, encoders,
    range, shape of the gearbox and so on are all in one place.
    """
    components = cfg["components"]
    arm_id = arm["id"]
    group_for_joint: dict[int, dict[str, Any]] = {}
    for group in arm["actuator_groups"]:
        for index in group["joints"]:
            group_for_joint[int(index)] = group

    joints: list[dict[str, Any]] = []
    for index in range(1, 7):
        group = group_for_joint[index]
        hd = components["harmonic_drives"][group["harmonic_drive"]]
        motor = components["motors"][group["motor"]]
        limit = arm["joint_limits_deg"][f"j{index}"]
        encoder_joint = components["encoders"][arm["encoders"]["joint_side"]]
        encoder_motor = components["encoders"][arm["encoders"]["motor_side"]]

        ratio = float(hd["ratio"])
        motor_speed_limit = (
            float(motor["max_speed_rpm"]) * 360.0 / 60.0 / ratio
        )
        joint_speed_limit = min(
            float(arm["max_joint_velocity_deg_s"]), motor_speed_limit
        )

        joints.append(
            {
                "index": index,
                "name": f"j{index}",
                "axis": arm["joint_axes"][index - 1],
                "lower_deg": float(limit["min"]),
                "upper_deg": float(limit["max"]),
                "home_deg": float(arm["home_deg"][index - 1]),
                "gear_id": group["harmonic_drive"],
                "gear": hd,
                "motor_id": group["motor"],
                "motor": motor,
                "brake_id": arm["brake"],
                "brake": components["brakes"][arm["brake"]],
                "motor_encoder_id": arm["encoders"]["motor_side"],
                "motor_encoder": encoder_motor,
                "joint_encoder_id": arm["encoders"]["joint_side"],
                "joint_encoder": encoder_joint,
                "ratio": ratio,
                "efficiency": float(hd["efficiency"]),
                "backlash_deg": float(hd["max_backlash_arcmin"]) / 60.0,
                "nominal_output_torque_nm": (
                    float(motor["nominal_torque_nm"]) * ratio * float(hd["efficiency"])
                ),
                "peak_output_torque_nm": (
                    float(motor["peak_torque_nm"]) * ratio * float(hd["efficiency"])
                ),
                "max_output_speed_deg_s": joint_speed_limit,
                "joint_encoder_lsb_deg": 360.0 / float(encoder_joint["counts_per_rev"]),
                "motor_encoder_lsb_at_output_deg": (
                    360.0 / (float(encoder_motor["counts_per_rev"]) * ratio)
                ),
                "arm_id": arm_id,
            }
        )
    return joints


def config_summary(cfg: dict[str, Any]) -> dict[str, Any]:
    """Human-friendly derived facts used by the CLI and tests."""
    arms = cfg["arms"]
    joints = [resolve_arm_joints(cfg, arm) for arm in arms]
    total_axes = sum(len(items) for items in joints)
    arm_reach = {
        arm["id"]: sum(arm["geometry"]["link_lengths_mm"][key] for key in (
            "upper_arm", "forearm", "wrist_roll", "wrist_pitch", "tool"
        ))
        for arm in arms
    }
    return {
        "project": cfg["meta"]["project"],
        "config_path": cfg.get("_config_path"),
        "arm_count": len(arms),
        "total_axes": total_axes,
        "arm_ids": [arm["id"] for arm in arms],
        "reach_mm": arm_reach,
        "performance_targets_mm": cfg["performance_targets_mm"],
        "backlash_deg": cfg["error_budget"]["backlash"]["max_arcmin"] / 60.0,
        "backlash_rad": (
            cfg["error_budget"]["backlash"]["max_arcmin"] * math.pi / (180.0 * 60.0)
        ),
        "joint_encoder_lsb_deg": (
            360.0 / cfg["components"]["encoders"][
                cfg["error_budget"]["encoder_quantisation"]["encoder"]
            ]["counts_per_rev"]
        ),
        "all_joints_braked": all(
            cfg["components"]["brakes"].get(arm["brake"], {}).get("type")
            == "power-off electromagnetic" for arm in arms
        ),
        "cad_meshes_enabled": bool(cfg["cad_meshes"]["enabled"]),
    }
