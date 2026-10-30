"""Tiny helpers for writing URDF XML elements.

Keeping the XML mechanics here lets the generator module read like the robot
build itself: base column, link tube, harmonic-drive housing, motor, sensor.
All dimensions passed to these helpers are in millimetres and converted once
to metres, as URDF requires.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Iterable, Sequence

MM_TO_M = 0.001

# URDF cylinders point along local +z. These rotations align that axis with a
# robot joint axis (or with +x for a link tube).
AXIS_RPY_DEG = {
    "x": (0.0, 90.0, 0.0),
    "y": (-90.0, 0.0, 0.0),
    "z": (0.0, 0.0, 0.0),
}


def fmt(value: float) -> str:
    """Compact, deterministic number formatting for XML attributes."""
    text = f"{float(value):.10g}"
    return "0" if text == "-0" else text


def add_origin(
    element: ET.Element,
    xyz_mm: Sequence[float] = (0.0, 0.0, 0.0),
    rpy_deg: Sequence[float] = (0.0, 0.0, 0.0),
) -> None:
    ET.SubElement(
        element,
        "origin",
        {
            "xyz": " ".join(fmt(v * MM_TO_M) for v in xyz_mm),
            "rpy": " ".join(fmt(v * 3.141592653589793 / 180.0) for v in rpy_deg),
        },
    )


def add_material(root: ET.Element, name: str, rgba: Sequence[float]) -> None:
    material = ET.SubElement(root, "material", {"name": name})
    ET.SubElement(material, "color", {"rgba": " ".join(fmt(v) for v in rgba)})


def _add_geometry(
    parent: ET.Element,
    kind: str,
    *,
    radius_mm: float = 0.0,
    length_mm: float = 0.0,
    size_mm: Sequence[float] | None = None,
    mesh_filename: str | None = None,
    mesh_scale: Sequence[float] | None = None,
) -> None:
    geometry = ET.SubElement(parent, "geometry")
    if kind == "cylinder":
        ET.SubElement(
            geometry,
            "cylinder",
            {
                "radius": fmt(radius_mm * MM_TO_M),
                "length": fmt(length_mm * MM_TO_M),
            },
        )
    elif kind == "box":
        if not size_mm or len(size_mm) != 3:
            raise ValueError("box geometry needs size_mm = [x, y, z]")
        ET.SubElement(
            geometry,
            "box",
            {"size": " ".join(fmt(v * MM_TO_M) for v in size_mm)},
        )
    elif kind == "sphere":
        ET.SubElement(geometry, "sphere", {"radius": fmt(radius_mm * MM_TO_M)})
    elif kind == "mesh":
        if not mesh_filename:
            raise ValueError("mesh geometry needs a filename")
        attributes = {"filename": mesh_filename}
        if mesh_scale:
            attributes["scale"] = " ".join(fmt(v) for v in mesh_scale)
        ET.SubElement(geometry, "mesh", attributes)
    else:
        raise ValueError(f"Unsupported geometry kind: {kind}")


def add_cylinder_visual(
    link: ET.Element,
    name: str,
    color_name: str,
    *,
    radius_mm: float,
    length_mm: float,
    xyz_mm: Sequence[float] = (0, 0, 0),
    axis: str = "z",
) -> None:
    visual = ET.SubElement(link, "visual", {"name": name})
    add_origin(visual, xyz_mm, AXIS_RPY_DEG[axis])
    _add_geometry(visual, "cylinder", radius_mm=radius_mm, length_mm=length_mm)
    ET.SubElement(visual, "material", {"name": color_name})


def add_box_visual(
    link: ET.Element,
    name: str,
    color_name: str,
    *,
    size_mm: Sequence[float],
    xyz_mm: Sequence[float] = (0, 0, 0),
) -> None:
    visual = ET.SubElement(link, "visual", {"name": name})
    add_origin(visual, xyz_mm, (0, 0, 0))
    _add_geometry(visual, "box", size_mm=size_mm)
    ET.SubElement(visual, "material", {"name": color_name})


def add_mesh_visual(
    link: ET.Element,
    name: str,
    color_name: str,
    *,
    filename: str,
    scale: Sequence[float],
) -> None:
    visual = ET.SubElement(link, "visual", {"name": name})
    add_origin(visual)
    _add_geometry(visual, "mesh", mesh_filename=filename, mesh_scale=scale)
    ET.SubElement(visual, "material", {"name": color_name})


def add_cylinder_collision(
    link: ET.Element,
    name: str,
    *,
    radius_mm: float,
    length_mm: float,
    xyz_mm: Sequence[float] = (0, 0, 0),
    axis: str = "z",
) -> None:
    collision = ET.SubElement(link, "collision", {"name": name})
    add_origin(collision, xyz_mm, AXIS_RPY_DEG[axis])
    _add_geometry(collision, "cylinder", radius_mm=radius_mm, length_mm=length_mm)


def add_box_collision(
    link: ET.Element,
    name: str,
    *,
    size_mm: Sequence[float],
    xyz_mm: Sequence[float] = (0, 0, 0),
) -> None:
    collision = ET.SubElement(link, "collision", {"name": name})
    add_origin(collision, xyz_mm, (0, 0, 0))
    _add_geometry(collision, "box", size_mm=size_mm)


def add_mesh_collision(
    link: ET.Element,
    name: str,
    *,
    filename: str,
    scale: Sequence[float],
) -> None:
    collision = ET.SubElement(link, "collision", {"name": name})
    add_origin(collision)
    _add_geometry(collision, "mesh", mesh_filename=filename, mesh_scale=scale)


def add_inertial(
    link: ET.Element,
    *,
    mass_kg: float,
    com_xyz_mm: Sequence[float],
    ixx: float,
    iyy: float,
    izz: float,
) -> None:
    if mass_kg <= 0:
        raise ValueError(f"inertial mass must be positive, got {mass_kg}")
    inertial = ET.SubElement(link, "inertial")
    add_origin(inertial, com_xyz_mm)
    ET.SubElement(inertial, "mass", {"value": fmt(mass_kg)})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": fmt(ixx),
            "ixy": "0",
            "ixz": "0",
            "iyy": fmt(iyy),
            "iyz": "0",
            "izz": fmt(izz),
        },
    )


def solid_cylinder_mass(density_kg_m3: float, radius_mm: float, length_mm: float) -> float:
    radius_m = radius_mm * MM_TO_M
    length_m = length_mm * MM_TO_M
    return density_kg_m3 * 3.141592653589793 * radius_m**2 * length_m


def hollow_tube_mass(
    density_kg_m3: float, outer_radius_mm: float, wall_mm: float, length_mm: float
) -> float:
    outer = outer_radius_mm * MM_TO_M
    inner = (outer_radius_mm - wall_mm) * MM_TO_M
    length = length_mm * MM_TO_M
    return density_kg_m3 * 3.141592653589793 * (outer**2 - inner**2) * length


def inertia_cylinder_along_x(mass: float, radius_mm: float, length_mm: float) -> tuple[float, float, float]:
    """Principal inertia of a solid cylinder whose symmetry axis is local +x."""
    r = radius_mm * MM_TO_M
    length = length_mm * MM_TO_M
    ixx = 0.5 * mass * r * r
    iyy_izz = mass * (3.0 * r * r + length * length) / 12.0
    return ixx, iyy_izz, iyy_izz


def inertia_cylinder_along_z(mass: float, radius_mm: float, length_mm: float) -> tuple[float, float, float]:
    """Principal inertia of a solid cylinder whose symmetry axis is local +z."""
    r = radius_mm * MM_TO_M
    length = length_mm * MM_TO_M
    ixx_iyy = mass * (3.0 * r * r + length * length) / 12.0
    izz = 0.5 * mass * r * r
    return ixx_iyy, ixx_iyy, izz


def combine_shaft_and_tip_mass(
    shaft_mass: float,
    tip_mass: float,
    radius_mm: float,
    length_mm: float,
) -> tuple[float, float, float, float, float]:
    """Combine a rod (COM mid-length) with a point payload at the tip.

    Returns (total_mass, com_x_mm, ixx, iyy, izz). The payload is on the local
    x-axis, so it adds no xx inertia but does add y/z inertia at its lever arm.
    """
    r = radius_mm * MM_TO_M
    length = length_mm * MM_TO_M
    total = shaft_mass + tip_mass
    if total <= 0:
        raise ValueError("tool link has no mass")
    com = (shaft_mass * 0.5 * length + tip_mass * length) / total
    ixx = 0.5 * shaft_mass * r * r
    shaft_iyy = shaft_mass * (3.0 * r * r + length * length) / 12.0
    iyy = (
        shaft_iyy
        + shaft_mass * (0.5 * length - com) ** 2
        + tip_mass * (length - com) ** 2
    )
    return total, com / MM_TO_M, ixx, iyy, iyy
