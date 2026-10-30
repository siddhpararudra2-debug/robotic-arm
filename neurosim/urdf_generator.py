"""Generate a single, shared-world URDF for the four-arm cell (Stage 0).

The URDF is a mechanical description only: four 6-DOF arms fixed to ``world``.
The phantom head, clamp and camera view are added in later stages.

Geometry comes from ``robot_config.yaml``. Joint housings and motors are drawn
as primitive parts; if a matching STL exists in the mesh directory, that STL
replaces the *visual* for that link. Collision geometry stays primitive unless
``cad_meshes.collision_from_mesh`` is enabled, because concave CAD meshes are a
poor collision representation for a real-time simulator.

Stage-0 inertias are deliberately simple lumped-cylinder estimates computed
from the configured materials and component masses. Replace them with CAD
inertias if the dynamics need to be quantitative.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Sequence
import xml.etree.ElementTree as ET

from .config import (
    ConfigError,
    config_summary,
    load_config,
    resolve_arm_joints,
)
from .urdf_geometry import (
    add_box_collision,
    add_box_visual,
    add_cylinder_collision,
    add_cylinder_visual,
    add_inertial,
    add_material,
    add_mesh_collision,
    add_mesh_visual,
    add_origin,
    combine_shaft_and_tip_mass,
    fmt,
    hollow_tube_mass,
    inertia_cylinder_along_x,
    inertia_cylinder_along_z,
    solid_cylinder_mass,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "robot_config.yaml"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "urdf" / "robot_cell.urdf"

# Structural mapping: the arm chain uses five config dimensions between axes.
# These names belong to the kinematic layout, not to a specific robot size.
TUBE_LINK_LENGTH_KEY = {
    2: "upper_arm",
    3: "forearm",
    4: "wrist_roll",
    5: "wrist_pitch",
}
JOINT_ORIGIN_LENGTH_KEY = {
    3: "upper_arm",
    4: "forearm",
    5: "wrist_roll",
    6: "wrist_pitch",
}
AXIS_VECTOR = {"x": "1 0 0", "y": "0 1 0", "z": "0 0 1"}


def _sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def _define_materials(root: ET.Element, cfg: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    """Create URDF materials; return lookups for physical and visual colours."""
    material_names: dict[str, str] = {}
    for material_id, spec in cfg["materials"].items():
        urdf_name = f"mat_{_sanitize(material_id)}"
        add_material(root, urdf_name, spec["color_rgba"])
        material_names[material_id] = urdf_name

    color_names: dict[str, str] = {}
    for color_id, rgba in cfg["visualization"]["colors_rgba"].items():
        urdf_name = f"vis_{_sanitize(color_id)}"
        add_material(root, urdf_name, rgba)
        color_names[color_id] = urdf_name
    return material_names, color_names


def _mesh_for_link(
    cfg: dict[str, Any],
    mesh_dir: Path,
    urdf_dir: Path,
    arm_id: str,
    link_id: str,
) -> dict[str, Any] | None:
    """Return mesh data if an STL exists for this arm/link, else None."""
    cad = cfg["cad_meshes"]
    if not cad.get("enabled") or link_id not in cad.get("link_ids", []):
        return None
    filename = cad["filename_template"].format(arm_id=arm_id, link_id=link_id)
    path = mesh_dir / filename
    if not path.is_file():
        return None
    relative = os.path.relpath(path, start=urdf_dir).replace(os.sep, "/")
    return {
        "filename": relative,
        "source": str(path),
        "scale": cad["scale"],
    }


def _module_mass(joint: dict[str, Any]) -> float:
    return (
        float(joint["gear"]["mass_kg"])
        + float(joint["motor"]["mass_kg"])
        + float(joint["brake"]["mass_kg"])
    )


def _add_joint_module(
    link: ET.Element,
    cfg: dict[str, Any],
    joint: dict[str, Any],
    color_names: dict[str, str],
    xyz_mm: Sequence[float],
    *,
    add_visual: bool,
    add_collision: bool,
) -> None:
    """Add a harmonic-drive housing and its side-mounted motor.

    The motor centre is offset sideways by ``motor_side_offset_factor *
    (R_housing + R_motor)`` so the two envelopes touch instead of intersecting.
    """
    hd = joint["gear"]
    motor = joint["motor"]
    axis = joint["axis"]
    hd_radius = float(hd["envelope_diameter_mm"]) / 2.0
    hd_length = float(hd["envelope_length_mm"])
    motor_radius = float(motor["envelope_diameter_mm"]) / 2.0
    motor_length = float(motor["envelope_length_mm"])
    label = f"{joint['arm_id']}_{joint['name']}"

    if add_visual:
        add_cylinder_visual(
            link,
            f"{label}_harmonic_drive",
            color_names["harmonic_drive"],
            radius_mm=hd_radius,
            length_mm=hd_length,
            xyz_mm=xyz_mm,
            axis=axis,
        )
    if add_collision:
        add_cylinder_collision(
            link,
            f"{label}_harmonic_drive",
            radius_mm=hd_radius,
            length_mm=hd_length,
            xyz_mm=xyz_mm,
            axis=axis,
        )

    # Choose a lateral offset direction that does not run along the link tube.
    offset_direction = {
        "z": (0.0, 1.0, 0.0),
        "y": (0.0, 0.0, 1.0),
        "x": (0.0, 0.0, 1.0),
    }[axis]
    distance = (
        float(cfg["visualization"]["motor_side_offset_factor"])
        * (hd_radius + motor_radius)
    )
    motor_xyz = tuple(
        float(xyz_mm[i]) + offset_direction[i] * distance for i in range(3)
    )

    if add_visual:
        add_cylinder_visual(
            link,
            f"{label}_motor",
            color_names["motor"],
            radius_mm=motor_radius,
            length_mm=motor_length,
            xyz_mm=motor_xyz,
            axis=axis,
        )
    if add_collision:
        add_cylinder_collision(
            link,
            f"{label}_motor",
            radius_mm=motor_radius,
            length_mm=motor_length,
            xyz_mm=motor_xyz,
            axis=axis,
        )


def _add_mesh_or_none(
    link: ET.Element,
    mesh: dict[str, Any] | None,
    color_name: str,
) -> bool:
    """Add a mesh visual if present. Return True when the visual was replaced."""
    if mesh is None:
        return False
    add_mesh_visual(
        link,
        "cad_mesh",
        color_name,
        filename=mesh["filename"],
        scale=mesh["scale"],
    )
    return True


def _add_tool_collisions(
    cfg: dict[str, Any],
    link: ET.Element,
    *,
    tool_length_mm: float,
    tool_parts: dict[str, Any],
) -> None:
    """Collision geometry mirrors the simple tool visuals, not the STL."""
    instrument = cfg["instrument"]
    shaft_radius = float(instrument["shaft_diameter_mm"]) / 2.0
    add_cylinder_collision(
        link, "instrument_shaft", radius_mm=shaft_radius,
        length_mm=tool_length_mm, xyz_mm=(tool_length_mm / 2.0, 0, 0), axis="x",
    )
    for item in tool_parts["collisions"]:
        if item["kind"] == "box":
            add_box_collision(
                link, item["name"], size_mm=item["size"], xyz_mm=item["xyz"]
            )
        else:
            add_cylinder_collision(
                link, item["name"], radius_mm=item["radius"],
                length_mm=item["length"], xyz_mm=item["xyz"], axis="x",
            )


def _tool_parts_from_config(
    cfg: dict[str, Any], arm: dict[str, Any]
) -> dict[str, Any]:
    """Collect sensor/fine-stage parts that sit on the tool link."""
    parts: list[dict[str, Any]] = []
    collisions: list[dict[str, Any]] = []
    end_mass = float(arm["payload_kg"])

    fine_stage = arm.get("fine_stage")
    if fine_stage:
        stage = cfg["components"]["fine_stages"][fine_stage["component"]]
        size = [float(v) for v in stage["envelope_xyz_mm"]]
        center = (size[0] / 2.0, 0.0, 0.0)
        parts.append({"kind": "box", "name": "piezo_fine_stage", "size": size, "xyz": center})
        collisions.append(parts[-1])
        end_mass += float(stage["mass_kg"])

    force_sensor = arm.get("force_sensor")
    if force_sensor:
        sensor = cfg["components"]["force_sensor"][force_sensor["component"]]
        radius = float(sensor["envelope_diameter_mm"]) / 2.0
        length = float(sensor["envelope_length_mm"])
        center = (length / 2.0, 0.0, 0.0)
        parts.append({
            "kind": "cylinder", "name": "force_torque_sensor",
            "radius": radius, "length": length, "xyz": center,
        })
        collisions.append(parts[-1])
        end_mass += float(sensor["mass_kg"])

    end_effector = arm.get("end_effector")
    if end_effector:
        camera_cfg = end_effector.get("camera")
        if camera_cfg:
            camera = cfg["components"]["cameras"][camera_cfg["component"]]
            envelope = camera["envelope_mm"]
            size = [
                float(envelope["depth"]),
                float(envelope["width"]),
                float(envelope["height"]),
            ]
            center = tuple(float(v) for v in camera_cfg["mount_offset_mm"])
            parts.append({"kind": "box", "name": "realsense_d405", "size": size, "xyz": center})
            collisions.append(parts[-1])

        laser_cfg = end_effector.get("laser")
        if laser_cfg:
            laser = cfg["components"]["lasers"][laser_cfg["component"]]
            radius = float(laser["envelope_diameter_mm"]) / 2.0
            length = float(laser["envelope_length_mm"])
            center = tuple(float(v) for v in laser_cfg["mount_offset_mm"])
            parts.append({
                "kind": "cylinder", "name": "laser_module",
                "radius": radius, "length": length, "xyz": center,
            })
            collisions.append(parts[-1])

    return {
        "parts": parts,
        "collisions": collisions,
        "end_mass_kg": end_mass,
    }


def _add_tool_visuals(
    link: ET.Element,
    cfg: dict[str, Any],
    arm: dict[str, Any],
    tool_length_mm: float,
    color_names: dict[str, str],
) -> None:
    instrument = cfg["instrument"]
    shaft_radius = float(instrument["shaft_diameter_mm"]) / 2.0
    add_cylinder_visual(
        link, "instrument_shaft", color_names["instrument"],
        radius_mm=shaft_radius, length_mm=tool_length_mm,
        xyz_mm=(tool_length_mm / 2.0, 0, 0), axis="x",
    )

    for item in _tool_parts_from_config(cfg, arm)["parts"]:
        if item["kind"] == "box":
            color = (
                color_names["piezo_stage"] if item["name"] == "piezo_fine_stage"
                else color_names["camera"]
            )
            add_box_visual(
                link, item["name"], color, size_mm=item["size"], xyz_mm=item["xyz"]
            )
        else:
            if item["name"] == "laser_module":
                color = color_names["laser"]
            elif item["name"] == "force_torque_sensor":
                color = color_names["force_sensor"]
            else:
                color = color_names["instrument"]
            add_cylinder_visual(
                link, item["name"], color,
                radius_mm=item["radius"], length_mm=item["length"],
                xyz_mm=item["xyz"], axis="x",
            )


def _add_revolute_joint(
    root: ET.Element,
    joint: dict[str, Any],
    parent: str,
    child: str,
    origin_xyz_mm: Sequence[float],
) -> None:
    joint_el = ET.SubElement(
        root,
        "joint",
        {"name": f"{joint['arm_id']}_{joint['name']}", "type": "revolute"},
    )
    ET.SubElement(joint_el, "parent", {"link": parent})
    ET.SubElement(joint_el, "child", {"link": child})
    add_origin(joint_el, origin_xyz_mm)
    ET.SubElement(joint_el, "axis", {"xyz": AXIS_VECTOR[joint["axis"]]})
    ET.SubElement(
        joint_el,
        "limit",
        {
            "lower": fmt(joint["lower_deg"] * math.pi / 180.0),
            "upper": fmt(joint["upper_deg"] * math.pi / 180.0),
            "effort": fmt(joint["peak_output_torque_nm"]),
            "velocity": fmt(joint["max_output_speed_deg_s"] * math.pi / 180.0),
        },
    )


def _add_fixed_joint(
    root: ET.Element,
    name: str,
    parent: str,
    child: str,
    xyz_mm: Sequence[float],
    rpy_deg: Sequence[float] = (0, 0, 0),
) -> None:
    joint_el = ET.SubElement(root, "joint", {"name": name, "type": "fixed"})
    ET.SubElement(joint_el, "parent", {"link": parent})
    ET.SubElement(joint_el, "child", {"link": child})
    add_origin(joint_el, xyz_mm, rpy_deg)


def _add_base_link(
    root: ET.Element,
    cfg: dict[str, Any],
    arm: dict[str, Any],
    joints: list[dict[str, Any]],
    mesh: dict[str, Any] | None,
    material_names: dict[str, str],
    color_names: dict[str, str],
) -> float:
    arm_id = arm["id"]
    link = ET.SubElement(root, "link", {"name": f"{arm_id}_base"})
    column_length = float(arm["base_column_mm"])
    j1 = joints[0]
    column_radius = float(j1["gear"]["envelope_diameter_mm"]) / 2.0
    material_id = arm["geometry"]["link_materials"][0]
    density = float(cfg["materials"][material_id]["density_kg_m3"])
    mass = (
        solid_cylinder_mass(density, column_radius, column_length)
        + _module_mass(j1)
    )
    ixx, iyy, izz = inertia_cylinder_along_z(mass, column_radius, column_length)
    add_inertial(
        link, mass_kg=mass, com_xyz_mm=(0, 0, column_length / 2.0),
        ixx=ixx, iyy=iyy, izz=izz,
    )

    collision_from_mesh = bool(mesh and cfg["cad_meshes"]["collision_from_mesh"])
    _add_mesh_or_none(link, mesh, material_names[material_id])
    if collision_from_mesh:
        add_mesh_collision(link, "cad_mesh", filename=mesh["filename"], scale=mesh["scale"])

    if mesh is None:
        add_cylinder_visual(
            link, "base_column", material_names[material_id],
            radius_mm=column_radius, length_mm=column_length,
            xyz_mm=(0, 0, column_length / 2.0), axis="z",
        )
    if not collision_from_mesh:
        add_cylinder_collision(
            link, "base_column", radius_mm=column_radius,
            length_mm=column_length, xyz_mm=(0, 0, column_length / 2.0), axis="z",
        )

    _add_joint_module(
        link, cfg, j1, color_names, (0, 0, column_length),
        add_visual=mesh is None, add_collision=not collision_from_mesh,
    )
    return mass


def _add_yoke_link(
    root: ET.Element,
    cfg: dict[str, Any],
    arm: dict[str, Any],
    joints: list[dict[str, Any]],
    mesh: dict[str, Any] | None,
    material_names: dict[str, str],
    color_names: dict[str, str],
) -> float:
    """Link 1 is the short shoulder yoke; it carries joint 2."""
    arm_id = arm["id"]
    link = ET.SubElement(root, "link", {"name": f"{arm_id}_link1"})
    j2 = joints[1]
    mass = _module_mass(j2)
    radius = float(j2["gear"]["envelope_diameter_mm"]) / 2.0
    length = float(j2["gear"]["envelope_length_mm"])
    ixx, iyy, izz = inertia_cylinder_along_x(mass, radius, length)
    add_inertial(link, mass_kg=mass, com_xyz_mm=(0, 0, 0), ixx=ixx, iyy=iyy, izz=izz)

    material_id = arm["geometry"]["link_materials"][0]
    collision_from_mesh = bool(mesh and cfg["cad_meshes"]["collision_from_mesh"])
    _add_mesh_or_none(link, mesh, material_names[material_id])
    if collision_from_mesh:
        add_mesh_collision(link, "cad_mesh", filename=mesh["filename"], scale=mesh["scale"])
    _add_joint_module(
        link, cfg, j2, color_names, (0, 0, 0),
        add_visual=mesh is None, add_collision=not collision_from_mesh,
    )
    return mass


def _add_tube_link(
    root: ET.Element,
    cfg: dict[str, Any],
    arm: dict[str, Any],
    joints: list[dict[str, Any]],
    link_index: int,
    length_mm: float,
    mesh: dict[str, Any] | None,
    material_names: dict[str, str],
    color_names: dict[str, str],
) -> float:
    """Links 2-5: tube from the proximal joint to the next joint module."""
    arm_id = arm["id"]
    link = ET.SubElement(root, "link", {"name": f"{arm_id}_link{link_index}"})
    tube_radius = float(arm["geometry"]["tube_outer_diameter_mm"]) / 2.0
    wall = float(arm["geometry"]["tube_wall_mm"])
    material_id = arm["geometry"]["link_materials"][link_index - 1]
    density = float(cfg["materials"][material_id]["density_kg_m3"])
    distal_joint = joints[link_index]  # link 2 carries j3, ..., link 5 carries j6
    mass = (
        hollow_tube_mass(density, tube_radius, wall, length_mm)
        + _module_mass(distal_joint)
    )
    ixx, iyy, izz = inertia_cylinder_along_x(mass, tube_radius, length_mm)
    add_inertial(
        link, mass_kg=mass, com_xyz_mm=(length_mm / 2.0, 0, 0),
        ixx=ixx, iyy=iyy, izz=izz,
    )

    collision_from_mesh = bool(mesh and cfg["cad_meshes"]["collision_from_mesh"])
    _add_mesh_or_none(link, mesh, material_names[material_id])
    if collision_from_mesh:
        add_mesh_collision(link, "cad_mesh", filename=mesh["filename"], scale=mesh["scale"])
    if mesh is None:
        add_cylinder_visual(
            link, "link_tube", material_names[material_id],
            radius_mm=tube_radius, length_mm=length_mm,
            xyz_mm=(length_mm / 2.0, 0, 0), axis="x",
        )
        _add_joint_module(
            link, cfg, distal_joint, color_names, (length_mm, 0, 0),
            add_visual=True, add_collision=False,
        )
    if not collision_from_mesh:
        add_cylinder_collision(
            link, "link_tube", radius_mm=tube_radius, length_mm=length_mm,
            xyz_mm=(length_mm / 2.0, 0, 0), axis="x",
        )
        _add_joint_module(
            link, cfg, distal_joint, color_names, (length_mm, 0, 0),
            add_visual=False, add_collision=True,
        )
    return mass


def _add_tool_link(
    root: ET.Element,
    cfg: dict[str, Any],
    arm: dict[str, Any],
    mesh: dict[str, Any] | None,
    material_names: dict[str, str],
    color_names: dict[str, str],
) -> float:
    arm_id = arm["id"]
    link = ET.SubElement(root, "link", {"name": f"{arm_id}_link6"})
    tool_length = float(arm["geometry"]["link_lengths_mm"]["tool"])
    instrument = cfg["instrument"]
    shaft_radius = float(instrument["shaft_diameter_mm"]) / 2.0
    shaft_material = instrument["material_default"]
    density = float(cfg["materials"][shaft_material]["density_kg_m3"])
    shaft_mass = solid_cylinder_mass(density, shaft_radius, tool_length)
    tool_parts = _tool_parts_from_config(cfg, arm)
    mass, com_x, ixx, iyy, izz = combine_shaft_and_tip_mass(
        shaft_mass, tool_parts["end_mass_kg"], shaft_radius, tool_length
    )
    add_inertial(
        link, mass_kg=mass, com_xyz_mm=(com_x, 0, 0),
        ixx=ixx, iyy=iyy, izz=izz,
    )

    material_id = arm["geometry"]["link_materials"][5]
    collision_from_mesh = bool(mesh and cfg["cad_meshes"]["collision_from_mesh"])
    _add_mesh_or_none(link, mesh, material_names[material_id])
    if collision_from_mesh:
        add_mesh_collision(link, "cad_mesh", filename=mesh["filename"], scale=mesh["scale"])

    if mesh is None:
        _add_tool_visuals(link, cfg, arm, tool_length, color_names)
    if not collision_from_mesh:
        _add_tool_collisions(
            cfg, link, tool_length_mm=tool_length, tool_parts=tool_parts
        )
    return mass


def generate_urdf(
    cfg: dict[str, Any],
    output_path: str | Path | None = None,
    mesh_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Generate the URDF and return a small report dictionary."""
    project_root = Path(cfg["_project_root"])
    out_path = Path(output_path) if output_path else DEFAULT_OUTPUT_PATH
    if not out_path.is_absolute():
        out_path = project_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cad = cfg["cad_meshes"]
    mesh_root = Path(mesh_dir) if mesh_dir else Path(cad["directory"])
    if not mesh_root.is_absolute():
        mesh_root = project_root / mesh_root

    root = ET.Element("robot", {"name": _sanitize(cfg["meta"]["project"]).lower()})
    material_names, color_names = _define_materials(root, cfg)
    ET.SubElement(root, "link", {"name": "world"})

    report: dict[str, Any] = {
        "output": str(out_path.resolve()),
        "arm_count": len(cfg["arms"]),
        "revolute_joints": 0,
        "links": 1,
        "meshes_used": [],
        "primitive_fallbacks": [],
        "arm_masses_kg": {},
    }
    total_mass = 0.0

    for arm in cfg["arms"]:
        arm_id = arm["id"]
        joints = resolve_arm_joints(cfg, arm)
        base_link_name = f"{arm_id}_base"

        base_mesh = _mesh_for_link(cfg, mesh_root, out_path.parent, arm_id, "base")
        _record_mesh(report, arm_id, "base", base_mesh)
        mass = _add_base_link(
            root, cfg, arm, joints, base_mesh, material_names, color_names
        )
        mass += _add_arm_chain(
            root, cfg, arm, joints, mesh_root, out_path.parent,
            material_names, color_names, report,
        )
        total_mass += mass
        report["arm_masses_kg"][arm_id] = round(mass, 4)

        mount = arm["base_mount"]
        _add_fixed_joint(
            root, f"{arm_id}_mount", "world", base_link_name,
            mount["xyz_mm"], mount["rpy_deg"],
        )
        tip_name = f"{arm_id}_tip"
        tip = ET.SubElement(root, "link", {"name": tip_name})
        tip_radius = float(cfg["instrument"]["shaft_diameter_mm"]) / 2.0
        add_cylinder_visual(
            tip, "tool_tip_marker", color_names["instrument"],
            radius_mm=tip_radius, length_mm=tip_radius * 2.0,
            xyz_mm=(0, 0, 0), axis="x",
        )
        # The tip is a marker frame, not a physical part: a tiny inertial
        # keeps PyBullet from inventing a default 1 kg mass for the link.
        add_inertial(
            tip, mass_kg=0.001, com_xyz_mm=(0, 0, 0),
            ixx=1e-9, iyy=1e-9, izz=1e-9,
        )
        tool_length = float(arm["geometry"]["link_lengths_mm"]["tool"])
        _add_fixed_joint(
            root, f"{arm_id}_tip_joint", f"{arm_id}_link6", tip_name,
            (tool_length, 0, 0),
        )

    report["total_mass_kg"] = round(total_mass, 4)
    report["revolute_joints"] = sum(len(arm["joint_axes"]) for arm in cfg["arms"])
    report["links"] = len(root.findall("link"))

    ET.indent(root, space="  ")
    header = (
        '<?xml version="1.0"?>\n'
        "<!-- Generated by neurosim.urdf_generator from robot_config.yaml.\n"
        "     Four 6-DOF arms share the 'world' frame. Stage-0 visuals use STL\n"
        "     meshes when present, otherwise config-sized primitive parts.\n"
        "     Inertia values are lumped estimates, not CAD inertias. -->\n"
    )
    out_path.write_text(header + ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
    return report


def _record_mesh(
    report: dict[str, Any], arm_id: str, link_id: str, mesh: dict[str, Any] | None
) -> None:
    label = f"{arm_id}_{link_id}"
    if mesh is None:
        report["primitive_fallbacks"].append(label)
    else:
        report["meshes_used"].append(label)


def _add_arm_chain(
    root: ET.Element,
    cfg: dict[str, Any],
    arm: dict[str, Any],
    joints: list[dict[str, Any]],
    mesh_root: Path,
    urdf_dir: Path,
    material_names: dict[str, str],
    color_names: dict[str, str],
    report: dict[str, Any],
) -> float:
    """Add links 1-6 and joints 1-6 for one arm. Returns link mass (no base)."""
    arm_id = arm["id"]
    lengths = arm["geometry"]["link_lengths_mm"]
    masses: list[float] = []

    for index in range(1, 7):
        link_id = f"link{index}"
        mesh = _mesh_for_link(cfg, mesh_root, urdf_dir, arm_id, link_id)
        _record_mesh(report, arm_id, link_id, mesh)

        if index == 1:
            mass = _add_yoke_link(
                root, cfg, arm, joints, mesh, material_names, color_names
            )
        elif 2 <= index <= 5:
            length_key = TUBE_LINK_LENGTH_KEY[index]
            mass = _add_tube_link(
                root, cfg, arm, joints, index,
                float(lengths[length_key]), mesh, material_names, color_names,
            )
        else:
            mass = _add_tool_link(
                root, cfg, arm, mesh, material_names, color_names
            )
        masses.append(mass)

        if index == 1:
            parent = f"{arm_id}_base"
            origin = (0, 0, float(arm["base_column_mm"]))
        elif index == 2:
            parent = f"{arm_id}_link1"
            origin = (0, 0, 0)
        else:
            length_key = JOINT_ORIGIN_LENGTH_KEY[index]
            parent = f"{arm_id}_link{index - 1}"
            origin = (float(lengths[length_key]), 0, 0)

        _add_revolute_joint(
            root, joints[index - 1], parent, f"{arm_id}_link{index}", origin
        )

    return sum(masses)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG_PATH),
        help="Path to robot_config.yaml (default: project root).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output URDF path (default: <project>/urdf/robot_cell.urdf).",
    )
    parser.add_argument(
        "--mesh-dir", default=None,
        help="Override CAD mesh directory from the config.",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"Config error: {exc}")
        return 2

    report = generate_urdf(cfg, args.out, args.mesh_dir)
    summary = config_summary(cfg)
    print(f"Generated: {report['output']}")
    print(
        f"  {report['arm_count']} arms | {report['revolute_joints']} revolute axes | "
        f"{report['links']} URDF links | {report['total_mass_kg']:.2f} kg (lumped inertia)"
    )
    print(
        f"  CAD meshes used: {len(report['meshes_used'])} | "
        f"primitive fallbacks: {len(report['primitive_fallbacks'])}"
    )
    if report["meshes_used"]:
        print("  Meshes: " + ", ".join(report["meshes_used"]))
    targets = ", ".join(
        f"{arm}={target:g} mm" for arm, target in summary["performance_targets_mm"].items()
        if target is not None
    )
    print(f"  Accuracy targets: {targets}")
    print(f"  Next: inspect the URDF, then Stage 1 adds the phantom head and clamp.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
