"""Debug: compare all-pairs getClosestPoints(robot, robot) vs per-link calls."""
from __future__ import annotations

from neurosim.config import load_config
from neurosim.scene import build_scene, close, _link_name
from neurosim.layout import _arm_link_indices
import pybullet as p

CONFIG = "c:\\Users\\rudra\\Desktop\\neurosurgical_robot\\robot_config.yaml"


def main() -> None:
    from neurosim.config import load_config as lc
    cfg = lc(CONFIG)
    scene = build_scene(cfg, gui=False)
    robot = scene.robot_id
    arm_of_link = scene.arm_of_link

    # Set all arms to home.
    from neurosim.layout import _set_arm_joints
    for arm in cfg["arms"]:
        _set_arm_joints(scene, cfg, arm["id"], [float(v) for v in arm["home_deg"]])
    p.performCollisionDetection(physicsClientId=scene.client_id)

    # All-pairs query.
    all_pairs = p.getClosestPoints(robot, robot, distance=0.0005, physicsClientId=scene.client_id)
    print(f"all-pairs close points: {len(all_pairs)}")
    print(f"len(pt)={len(all_pairs[0]) if all_pairs else 'n/a'}")
    for i, pt in enumerate(all_pairs[:6]):
        print(f"  pt[{i}] len={len(pt)} pt[7]={pt[7]!r} type={type(pt[7]).__name__} full={pt}")
    cross_arm = []
    for pt in all_pairs:
        la, lb = pt[2], pt[3]
        dist = pt[8] if len(pt) > 8 else pt[7]
        oa, ob = arm_of_link.get(la), arm_of_link.get(lb)
        if oa != ob and oa != "world" and ob != "world" and dist <= 0.0005:
            cross_arm.append((la, oa, lb, ob, dist))
    print(f"cross-arm close pairs (dist<=0.0005): {len(cross_arm)}")
    for c in cross_arm[:30]:
        print("  ", c)

    # Per-link query for A1 links vs all other-arm links, like OLD code.
    arm_of_link = scene.arm_of_link
    a1_links = _arm_link_indices(scene, "A1")
    other_links = [(i, o) for i, o in arm_of_link.items() if o not in ("A1", "world")]
    print(f"\nA1 links: {a1_links}")
    print(f"other-arm links: {other_links}")
    count = 0
    for li in a1_links:
        for oi, owner in other_links:
            pts = p.getClosestPoints(robot, robot, distance=0.0005,
                                     linkIndexA=li, linkIndexB=oi,
                                     physicsClientId=scene.client_id)
            if pts:
                count += 1
                print(f"  A1 link#{li} ({_link_name(scene,li)}) vs {owner} link#{oi} ({_link_name(scene,oi)}): {len(pts)} pts")
    print(f"per-link A1-vs-other close pairs: {count}")

    # Verify getClosestPoints(robot, robot, linkIndexA=li) reliability:
    # does linkIndexA-only return only genuinely-close links?
    print("\n=== linkIndexA-only query for A1 link0 ===")
    pts = p.getClosestPoints(robot, robot, distance=0.0005, linkIndexA=0, physicsClientId=scene.client_id)
    print(f"  {len(pts)} close points for link#0")
    for pt in pts[:15]:
        print("   ", pt[2], "->", pt[3], "arm", arm_of_link.get(pt[3]), "dist", pt[7])

    close(scene)


if __name__ == "__main__":
    main()
