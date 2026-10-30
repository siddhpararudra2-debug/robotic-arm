# Neurosurgical 4-arm robot: staged simulator

Stage 1 is complete: the Stage-0 config + URDF generator, plus a PyBullet scene
with the phantom head, head clamp and operating table.
The kinematics, control, error models and metrics arrive in Stages 2-9. Each
stage stops here and waits for you to say **"next"**.

## Project structure

```
neurosurgical_robot/
|-- robot_config.yaml        single source of truth (PARTS + marked assumptions)
|-- generate_urdf.py         convenience launcher
|-- run_simulation.py        Stage-1 scene launcher (--gui for the 3D viewer)
|-- neurosim/
|   |-- config.py            YAML loader, validation and derived values
|   |-- urdf_geometry.py     small URDF/XML and inertia helpers
|   |-- urdf_generator.py    builds the 4-arm URDF
|   `-- scene.py             Stage-1 PyBullet scene assembly
|-- tests/
|   |-- test_stage0.py       Stage-0 acceptance tests
|   `-- test_stage1.py       Stage-1 scene tests
|-- meshes/                  optional STL files (naming rules below)
|-- urdf/                    generated output
|   `-- robot_cell.urdf
`-- requirements.txt
```

Python 3.10+ is required. The current environment already has Python 3.12.

## Quick start

From `neurosurgical_robot/`:

```bash
python -m pip install -r requirements.txt
python generate_urdf.py
python -m unittest discover -s tests -v
python run_simulation.py          # headless scene check
python run_simulation.py --gui    # 3D viewer (press Q in the window to quit)
```

Expected generator output:

```
4 arms | 24 revolute axes | 33 URDF links | ~15.8 kg (lumped inertia)
CAD meshes used: 0 | primitive fallbacks: 28
Accuracy targets: A1=0.5 mm, A2=0.1 mm, A3=0.1 mm
```

The generated file is `urdf/robot_cell.urdf`.

## What Stage 1 adds

`run_simulation.py` assembles the PyBullet scene:

- the robot URDF loaded with the `world` link fixed;
- the phantom head as a static ellipsoid sized by
  `phantom.head_center_mm` / `phantom.head_radii_mm`;
- the rigid head clamp: `post_count` posts plus a `segment_count`-segment
  ring, sized by `phantom.head_clamp`;
- the operating table from `phantom.table`;
- every arm driven to its `home_deg` pose.

It then reports collisions at the home poses: arm-vs-arm, robot-vs-clamp,
robot-vs-table and robot-vs-head (tool links may rest near the phantom; other
links may not). New scene dimensions are marked `ASSUMPTION` in the YAML.

## What Stage 0 actually builds

The URDF has one root link, `world`, and four independent arms fixed to it
through `A1_mount` ... `A4_mount`. Each arm has exactly six revolute joints:

```
world -> mount -> base (column) -> j1 (yaw)
      -> link1 -> j2 (pitch)
      -> link2 (upper arm) -> j3 (pitch)
      -> link3 (forearm) -> j4 (roll)
      -> link4 (wrist_roll) -> j5 (pitch)
      -> link5 (wrist_pitch) -> j6 (roll)
      -> link6 (tool/sensors) -> tip (fixed frame at tool length)
```

The five link dimensions (`upper_arm`, `forearm`, `wrist_roll`,
`wrist_pitch`, `tool`) must add up to each arm's PARTS reach. The config
validator enforces that.

Each revolute joint includes:

- joint axis from `joint_axes`;
- position limits from `joint_limits_deg`;
- effort limit derived from motor peak torque x gear ratio x efficiency;
- velocity limit = the smaller of the arm safety speed and motor speed/ratio;
- a URDF collision model for the real parts.

Primitive fallback visuals show:

- the base column and harmonic-drive housings;
- side-mounted motor cylinders (Maxon EC-i 40 / EC-i 30 / EC 20 flat);
- aluminium and carbon-fibre link tubes;
- the 5 mm instrument shafts;
- the PI-class piezo XYZ stage on A2/A3;
- the ATI Nano17 force/torque sensor on A1;
- the Intel RealSense D405 camera and 650 nm laser module on A4.

Brake hardware is in the config and in the joint mass estimate, but a URDF
cannot represent a power-off brake. That behaviour is simulated in Stage 3.
Inertias are simple lumped-cylinder estimates, not CAD inertias; this is called
out in the generated URDF header and is sufficient for Stage 0. The A2/A3
tendon-driven wrists are represented as ordinary revolute wrists for now; the
tendon routing is recorded in the config but is not drawn in this stage.
Component envelopes (motors, gearboxes, camera, piezo stage) are approximate
sizes marked `ASSUMPTION` unless a PARTS dimension was available.

## Where the numbers live

`robot_config.yaml` contains every value from the PARTS block:

- arm count, reaches, payloads and wrist types;
- Harmonic Drive and Maxon motor assignment per joint;
- 100:1 gear ratios, under-1-arcmin backlash, brakes on every joint;
- two AS5047P 14-bit / 16384-count encoders per joint;
- 1 kHz control loop, 30-60 Hz vision loop, CAN-FD, EPOS4 / ODrive S1;
- AS5047P, D405, laser and filter, Nano17, NDI tracker, leader handles;
- piezo travel, shaft diameter and material choices;
- Pilz relay / e-stop / watchdog safety settings;
- complete error budget and accuracy targets.

Values not supplied in PARTS are marked `ASSUMPTION` in the YAML. The most
important ones to replace with measured data are motor masses, motor torque
limits, gearbox masses/envelopes and the arm base layout. The error models and
stage-8 conclusions should not be treated as hardware predictions until those
are verified.

## How to test Stage 0 by hand

1. Run `python generate_urdf.py`.
2. Open `urdf/robot_cell.urdf` in a URDF viewer (RViz/`check_urdf` if your ROS
   installation has it, PyBullet, or any URDF inspector). You should see four
   arms around the world origin and 24 rotatable joints.
3. Run the automated tests:
   ```bash
   python -m unittest discover -s tests -v
   ```
   The suite checks the PARTS values, chain lengths, joint limits, 24 axes,
   all 28 fallback links, STL replacement, collision fallback and that editing
   the YAML changes the URDF. If PyBullet is installed, it also loads the URDF
   and verifies 24 revolute joints.
4. Change one value in `robot_config.yaml`, for example the A1 j1 upper limit,
   regenerate, and search for it in the URDF. This demonstrates that the build
   is config-driven.

Stage-2 layout review should verify that these provisional base positions can
actually reach the target points without arm-arm collisions.

## Adding your CAD (STL) meshes

Place files in `meshes/` using exactly this pattern:

```
{arm_id}_{link_id}.stl
```

Examples: `A1_base.stl`, `A1_link2.stl`, `A4_link6.stl`.

Allowed link IDs: `base`, `link1`, `link2`, `link3`, `link4`, `link5`,
`link6`.

Mesh conventions:

- units are metres by default (set `cad_meshes.scale` to `[0.001, 0.001,
  0.001]` if your CAD is exported in millimetres);
- origin at the proximal joint axis of that link;
- local +x points down the link (the direction in which the next joint lies);
- the mesh replaces all primitive visuals for that link, so it should include
  the joint housing and motor that belong to that link;
- collision geometry remains primitive by default. Set
  `cad_meshes.collision_from_mesh: true` only if the STLs are suitable for
  collision.

If a mesh is missing, the generator silently falls back to the primitive parts;
the generator prints a count of used meshes and fallbacks.

## What is intentionally not here yet

- Stage 2: FK/IK, workspace and collision checks.
- Stage 3: gear backlash, encoder quantization, brakes and 1 kHz actuator loop.
- Stage 4: laser spot, D405 noise model and triangulation.
- Stage 5: visual servoing, leader-follower scaling, tremor filter, piezo stage.
- Stage 6: calibration offset, tissue shift and force noise.
- Stage 7: joint, collision, e-stop, watchdog, force and speed safety tests.
- Stage 8: tip-error logs against the 0.5 mm / 0.1 mm targets and plots.
- Stage 9 (optional): ROS 2 / ros2_control port.

The base mounts and home poses are Stage-0 layout assumptions. They are
deliberately in the YAML so Stage 2 can adjust them after reach and collision
checks rather than reworking code.
