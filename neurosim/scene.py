"""Stage 1: assemble the full simulation scene in PyBullet.

Builds, entirely from ``robot_config.yaml``:

- the generated robot URDF (four arms fixed to the ``world`` frame);
- the phantom head as an ellipsoid (``phantom.head_radii_mm``);
- the rigid head clamp (posts + a ring of box segments);
- the operating table.

No dimension, color or count may be hardcoded here: everything is read from
the config so the scene cannot drift from the specification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pybullet as p
except ImportError:  # keeps this module importable before PyBullet is installed
    p = None

from .config import MM_PER_M, load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "robot_config.yaml"
DEFAULT_URDF_PATH = PROJECT_ROOT / "urdf" / "robot_cell.urdf"

REVOLUTE_JOINT_TYPES = {p.JOINT_REVOLUTE} if p is not None else set()


@dataclass
class Scene:
    """Handles and lookup tables for one assembled PyBullet scene."""

    client_id: int
    robot_id: int
    head_id: int
    clamp_ids: list[int]
    table_id: int
    ground_id: int = -1
    origin_line_ids: list[int] = field(default_factory=list)
    arm_joint_indices: dict[str, list[int]] = field(default_factory=dict)
    link_index_by_name: dict[str, int] = field(default_factory=dict)
    arm_of_link: dict[int, str] = field(default_factory=dict)
    tip_line_ids: dict[str, list[int]] = field(default_factory=dict)
    tip_text_ids: dict[str, int] = field(default_factory=dict)
    target_ids: dict[str, int] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Small geometry helpers (all inputs/outputs in metres).
# -----------------------------------------------------------------------------

def _ellipsoid_mesh(
    radii_m: tuple[float, float, float],
    segments: tuple[int, int] = (24, 16),
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    """Return (vertices, triangles) for a UV-sphere squashed to an ellipsoid."""
    nu, nv = segments
    a, b, c = radii_m
    u = np.linspace(0.0, 2.0 * math.pi, nu, endpoint=False)
    v = np.linspace(0.0, math.pi, nv, endpoint=True)
    vertices: list[tuple[float, float, float]] = []
    for vj in v:
        for ui in u:
            vertices.append(
                (
                    a * math.sin(vj) * math.cos(ui),
                    b * math.sin(vj) * math.sin(ui),
                    c * math.cos(vj),
                )
            )
    triangles: list[tuple[int, int, int]] = []
    for j in range(nv):
        for i in range(nu):
            i_next = (i + 1) % nu
            v00 = j * nu + i
            v01 = j * nu + i_next
            v10 = (j + 1) * nu + i
            v11 = (j + 1) * nu + i_next
            if j > 0:  # skip degenerate triangle at the poles
                triangles.append((v00, v01, v11))
            if j < nv - 1:
                triangles.append((v00, v11, v10))
    return vertices, triangles


def _rgba(color: list[float] | None, fallback: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(color) if color else fallback


# -----------------------------------------------------------------------------
# Scene builders.
# -----------------------------------------------------------------------------

def _build_head(cfg: dict[str, Any]) -> int:
    """Create the static phantom head ellipsoid."""
    phantom = cfg["phantom"]
    center = [v / MM_PER_M for v in phantom["head_center_mm"]]
    radii = [v / MM_PER_M for v in phantom["head_radii_mm"]]
    vertices, triangles = _ellipsoid_mesh(tuple(radii))
    flat_indices = [index for triangle in triangles for index in triangle]
    color = _rgba(phantom.get("head_color_rgba"), (0.93, 0.84, 0.78, 1.0))

    collision = p.createCollisionShape(
        p.GEOM_MESH, vertices=vertices, indices=flat_indices
    )
    visual = p.createVisualShape(
        p.GEOM_MESH,
        vertices=vertices,
        indices=flat_indices,
        rgbaColor=color,
    )
    # mass 0 -> static body; the clamp holds the phantom rigid anyway.
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=center,
    )


def _build_clamp(cfg: dict[str, Any]) -> list[int]:
    """Create the static head clamp: vertical posts plus a ring of boxes."""
    phantom = cfg["phantom"]
    clamp = phantom["head_clamp"]
    color = _rgba(clamp.get("color_rgba"), (0.16, 0.17, 0.19, 1.0))
    center = [v / MM_PER_M for v in phantom["head_center_mm"]]
    ring_radius = float(clamp["ring_radius_mm"]) / MM_PER_M
    ring_z = float(clamp["ring_center_z_mm"]) / MM_PER_M
    ring_height, ring_thickness = (
        float(v) / MM_PER_M for v in clamp["ring_cross_section_mm"]
    )
    post_radius = float(clamp["post_diameter_mm"]) / MM_PER_M / 2.0
    post_count = int(clamp["post_count"])
    segment_count = int(clamp["segment_count"])
    table_top_z = float(phantom["table"]["top_surface_z_mm"]) / MM_PER_M

    post_top_z = ring_z + ring_height / 2.0
    post_length = post_top_z - table_top_z
    post_shape = p.createCollisionShape(
        p.GEOM_CYLINDER, radius=post_radius, height=post_length
    )
    post_visual = p.createVisualShape(
        p.GEOM_CYLINDER, radius=post_radius, length=post_length, rgbaColor=color
    )
    post_z = (table_top_z + post_top_z) / 2.0
    bodies: list[int] = []
    for index in range(post_count):
        angle = 2.0 * math.pi * index / post_count + math.pi / post_count
        bodies.append(
            p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=post_shape,
                baseVisualShapeIndex=post_visual,
                basePosition=(
                    ring_radius * math.cos(angle),
                    ring_radius * math.sin(angle),
                    post_z,
                ),
            )
        )

    segment_length = 2.0 * math.pi * ring_radius / segment_count
    segment_shape = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=(segment_length / 2.0, ring_thickness / 2.0, ring_height / 2.0),
    )
    segment_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=(segment_length / 2.0, ring_thickness / 2.0, ring_height / 2.0),
        rgbaColor=color,
    )
    for index in range(segment_count):
        angle = 2.0 * math.pi * index / segment_count
        orientation = p.getQuaternionFromEuler((0.0, 0.0, angle + math.pi / 2.0))
        bodies.append(
            p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=segment_shape,
                baseVisualShapeIndex=segment_visual,
                basePosition=(
                    ring_radius * math.cos(angle),
                    ring_radius * math.sin(angle),
                    ring_z,
                ),
                baseOrientation=orientation,
            )
        )
    return bodies


def _build_table(cfg: dict[str, Any]) -> int:
    """Create the static operating table under the phantom."""
    table = cfg["phantom"]["table"]
    color = _rgba(table.get("color_rgba"), (0.25, 0.28, 0.32, 1.0))
    size_x, size_y = (float(v) / MM_PER_M for v in table["top_size_mm"])
    thickness = float(table["top_thickness_mm"]) / MM_PER_M
    top_z = float(table["top_surface_z_mm"]) / MM_PER_M
    half_extents = (size_x / 2.0, size_y / 2.0, thickness / 2.0)
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    visual = p.createVisualShape(
        p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color
    )
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=(0.0, 0.0, top_z - thickness / 2.0),
    )


def _build_ground_plane(cfg: dict[str, Any], client_id: int) -> int:
    """Create a large ground plane sized from the operating table."""
    table = cfg["phantom"]["table"]
    size_x, size_y = (float(v) / MM_PER_M for v in table["top_size_mm"])
    half = (max(size_x, size_y) * 1.2, max(size_x, size_y) * 1.2, 0.01)
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half,
        rgbaColor=(0.16, 0.18, 0.20, 1.0),
    )
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=(0.0, 0.0, -half[2]),
        physicsClientId=client_id,
    )


def _add_origin_axes(client_id: int, length_m: float = 0.15) -> list[int]:
    """Draw red/green/blue X/Y/Z debug lines at the world origin."""
    origin = (0.0, 0.0, 0.0)
    axis_colors = (
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, 1.0),
        (0.0, 0.0, 1.0, 1.0),
    )
    axis_dirs = (
        (length_m, 0.0, 0.0),
        (0.0, length_m, 0.0),
        (0.0, 0.0, length_m),
    )
    line_ids: list[int] = []
    for color, direction in zip(axis_colors, axis_dirs):
        line_ids.append(
            p.addUserDebugLine(
                origin, direction, color, physicsClientId=client_id
            )
        )
    return line_ids


def _setup_camera(client_id: int, cfg: dict[str, Any]) -> None:
    """Position the GUI camera to frame the phantom head and all four arms."""
    phantom = cfg["phantom"]
    center = [v / MM_PER_M for v in phantom["head_center_mm"]]
    p.resetDebugVisualizerCamera(
        cameraDistance=1.3,
        cameraYaw=40.0,
        cameraPitch=-22.0,
        cameraTargetPosition=center,
        physicsClientId=client_id,
    )


def _build_target_markers(cfg: dict[str, Any], client_id: int) -> dict[str, int]:
    """Draw one small sphere per arm at its configured target point (targets_mm)."""
    targets = cfg.get("targets_mm", {})
    color_by_arm = {
        "A1": (0.90, 0.20, 0.20, 1.0),
        "A2": (0.20, 0.90, 0.20, 1.0),
        "A3": (0.20, 0.50, 0.90, 1.0),
        "A4": (0.90, 0.70, 0.20, 1.0),
    }
    shape = p.createVisualShape(
        p.GEOM_SPHERE, radius=0.006, rgbaColor=(0.5, 0.5, 0.5, 1.0)
    )
    ids: dict[str, int] = {}
    for arm_id, target_mm in targets.items():
        rgba = color_by_arm.get(arm_id, (0.7, 0.7, 0.7, 1.0))
        # Recreate the shape per arm so each can have its own color.
        shape = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.006, rgbaColor=rgba
        )
        pos = [float(v) / MM_PER_M for v in target_mm]
        body = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=shape,
            basePosition=pos,
            physicsClientId=client_id,
        )
        ids[arm_id] = body
    return ids


def _index_robot(scene: Scene, cfg: dict[str, Any]) -> None:
    """Fill the joint/link lookup tables from the loaded URDF."""
    robot = scene.robot_id
    arm_ids = {arm["id"] for arm in cfg["arms"]}
    scene.link_index_by_name["world"] = -1
    scene.arm_of_link[-1] = "world"
    for joint_index in range(p.getNumJoints(robot, physicsClientId=scene.client_id)):
        info = p.getJointInfo(robot, joint_index, physicsClientId=scene.client_id)
        joint_name = info[1].decode()
        link_name = info[12].decode()
        scene.link_index_by_name[link_name] = joint_index
        arm_id = link_name.split("_", 1)[0]
        if arm_id in arm_ids:
            scene.arm_of_link[joint_index] = arm_id
        if info[2] in REVOLUTE_JOINT_TYPES:
            parts = joint_name.rsplit("_", 1)
            if len(parts) == 2 and parts[1] in ("j1", "j2", "j3", "j4", "j5", "j6"):
                scene.arm_joint_indices.setdefault(parts[0], []).append(joint_index)


def update_debug_visuals(scene: Scene, cfg: dict[str, Any]) -> None:
    """Draw per-frame tool-tip indicators and live position text."""
    client_id = scene.client_id
    robot_id = scene.robot_id

    # Remove the previous frame's debug items.
    for line_ids in scene.tip_line_ids.values():
        for line_id in line_ids:
            p.removeUserDebugItem(line_id, physicsClientId=client_id)
    for text_id in scene.tip_text_ids.values():
        p.removeUserDebugItem(text_id, physicsClientId=client_id)
    scene.tip_line_ids.clear()
    scene.tip_text_ids.clear()

    arms = {arm["id"]: arm for arm in cfg["arms"]}
    for arm_id, arm in arms.items():
        link_name = f"{arm_id}_link6"
        link_index = scene.link_index_by_name.get(link_name)
        if link_index is None:
            continue
        link_state = p.getLinkState(robot_id, link_index, physicsClientId=client_id)
        pos = link_state[0]
        orn = link_state[1]

        tool_length_m = (
            float(arm["geometry"]["link_lengths_mm"]["tool"]) / MM_PER_M
        )
        _, tip_end = p.multiplyTransforms(
            pos, orn, (0.0, 0.0, 0.0), (tool_length_m, 0.0, 0.0)
        )

        # Line from link 6 out to the tool tip.
        line_id = p.addUserDebugLine(
            pos,
            tip_end,
            (1.0, 0.25, 0.25, 1.0),
            lineWidth=2.5,
            physicsClientId=client_id,
        )
        # Thick cap at the tip so it is easy to see.
        cap_id = p.addUserDebugLine(
            tip_end,
            tip_end,
            (1.0, 0.25, 0.25, 1.0),
            lineWidth=6.0,
            physicsClientId=client_id,
        )
        scene.tip_line_ids[arm_id] = [line_id, cap_id]

        # Floating text with the tip position in millimetres.
        text = (
            f"{arm_id} tip: "
            f"({tip_end[0] * 1000.0:.1f}, {tip_end[1] * 1000.0:.1f}, "
            f"{tip_end[2] * 1000.0:.1f}) mm"
        )
        text_id = p.addUserDebugText(
            text,
            (tip_end[0], tip_end[1], tip_end[2] + 0.025),
            textColorRGB=(1.0, 1.0, 1.0),
            textSize=1.1,
            physicsClientId=client_id,
        )
        scene.tip_text_ids[arm_id] = text_id

        # Draw a line from the current tip to its configured target, if any.
        target_mm = cfg.get("targets_mm", {}).get(arm_id)
        if target_mm is not None:
            target_pos = [float(v) / MM_PER_M for v in target_mm]
            err_id = p.addUserDebugLine(
                tip_end,
                target_pos,
                (0.3, 0.9, 0.3, 1.0),
                lineWidth=1.5,
                physicsClientId=client_id,
            )
            scene.tip_line_ids.setdefault(arm_id, []).append(err_id)
            err_mm = float(
                np.linalg.norm(
                    np.array(tip_end) - np.array(target_pos)
                )
                * MM_PER_M
            )
            target_text = f"{arm_id} err: {err_mm:.2f} mm"
            target_text_id = p.addUserDebugText(
                target_text,
                (
                    (tip_end[0] + target_pos[0]) / 2.0,
                    (tip_end[1] + target_pos[1]) / 2.0,
                    (tip_end[2] + target_pos[2]) / 2.0 + 0.02,
                ),
                textColorRGB=(0.3, 0.9, 0.3),
                textSize=1.0,
                physicsClientId=client_id,
            )
            scene.tip_text_ids[f"{arm_id}_target"] = target_text_id


def build_scene(
    cfg: dict[str, Any],
    gui: bool = False,
    urdf_path: str | Path | None = None,
) -> Scene:
    """Connect to PyBullet and assemble the complete scene from the config."""
    if p is None:
        raise RuntimeError(
            "PyBullet is not installed - run 'python -m pip install pybullet'."
        )
    client_id = p.connect(p.GUI if gui else p.DIRECT)
    if client_id < 0:
        raise RuntimeError("Could not connect to PyBullet.")
    gravity = float(cfg["world"]["gravity_m_s2"])
    p.setGravity(0.0, 0.0, gravity, physicsClientId=client_id)
    p.setTimeStep(1.0 / float(cfg["world"]["control_loop_hz"]), physicsClientId=client_id)

    urdf = Path(urdf_path) if urdf_path else DEFAULT_URDF_PATH
    if not urdf.is_file():
        p.disconnect(physicsClientId=client_id)
        raise FileNotFoundError(
            f"{urdf} not found - run 'python generate_urdf.py' first."
        )
    robot_id = p.loadURDF(
        str(urdf), useFixedBase=True, physicsClientId=client_id
    )

    scene = Scene(
        client_id=client_id,
        robot_id=robot_id,
        head_id=_build_head(cfg),
        clamp_ids=_build_clamp(cfg),
        table_id=_build_table(cfg),
        ground_id=_build_ground_plane(cfg, client_id),
        origin_line_ids=_add_origin_axes(client_id),
        target_ids=_build_target_markers(cfg, client_id),
    )
    _index_robot(scene, cfg)
    reset_to_home(scene, cfg)
    p.performCollisionDetection(physicsClientId=client_id)
    if gui:
        _setup_camera(client_id, cfg)
    return scene


def reset_to_home(scene: Scene, cfg: dict[str, Any]) -> None:
    """Drive every revolute joint to its arm's configured home position."""
    for arm in cfg["arms"]:
        joint_indices = scene.arm_joint_indices[arm["id"]]
        for index, joint_index in enumerate(joint_indices):
            angle_rad = math.radians(float(arm["home_deg"][index]))
            p.resetJointState(
                scene.robot_id,
                joint_index,
                targetValue=angle_rad,
                targetVelocity=0.0,
                physicsClientId=scene.client_id,
            )


# -----------------------------------------------------------------------------
# Collision queries.
# -----------------------------------------------------------------------------

def _robot_link_indices(scene: Scene) -> list[int]:
    return [
        index for index in scene.arm_of_link
        if scene.arm_of_link[index] != "world"
    ]


def arm_arm_contacts(
    scene: Scene, distance_m: float = 0.0005
) -> list[tuple[str, str]]:
    """Return named link pairs from *different* arms that touch or come close."""
    robot = scene.robot_id
    links = _robot_link_indices(scene)
    contacts: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for a_index, a in enumerate(links):
        for b in links[a_index + 1:]:
            if scene.arm_of_link[a] == scene.arm_of_link[b]:
                continue
            points = p.getClosestPoints(
                robot, robot, distance=distance_m,
                linkIndexA=a, linkIndexB=b,
                physicsClientId=scene.client_id,
            )
            if points:
                name_a = _link_name(scene, a)
                name_b = _link_name(scene, b)
                pair = tuple(sorted((name_a, name_b)))
                if pair not in seen:
                    seen.add(pair)
                    contacts.append((name_a, name_b))
    return contacts


def contacts_with_body(
    scene: Scene, body_id: int, distance_m: float = 0.0005
) -> list[str]:
    """Return the robot link names that touch or come close to ``body_id``."""
    robot = scene.robot_id
    names: list[str] = []
    for link_index in _robot_link_indices(scene):
        points = p.getClosestPoints(
            robot, body_id, distance=distance_m,
            linkIndexA=link_index, physicsClientId=scene.client_id,
        )
        if points:
            names.append(_link_name(scene, link_index))
    return names


def _link_name(scene: Scene, link_index: int) -> str:
    if link_index == -1:
        return "world"
    for name, index in scene.link_index_by_name.items():
        if index == link_index:
            return name
    return f"link#{link_index}"


def close(scene: Scene) -> None:
    """Disconnect the physics client, cleaning up debug visuals first."""
    client_id = scene.client_id
    for line_ids in scene.tip_line_ids.values():
        for line_id in line_ids:
            p.removeUserDebugItem(line_id, physicsClientId=client_id)
    for text_id in scene.tip_text_ids.values():
        p.removeUserDebugItem(text_id, physicsClientId=client_id)
    for line_id in scene.origin_line_ids:
        p.removeUserDebugItem(line_id, physicsClientId=client_id)
    p.disconnect(physicsClientId=client_id)
