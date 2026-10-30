# CAD mesh drop-in folder

Put STL files here named `<arm>_<link>.stl`, for example:

- `A1_base.stl`
- `A1_link1.stl` ... `A1_link6.stl`
- `A2_link3.stl`
- `A4_link6.stl`

The generator (`python generate_urdf.py`) detects them automatically. If a file
is missing, that link uses config-sized primitive parts instead.

Rules:

- metres by default; if your CAD is in millimetres set
  `cad_meshes.scale: [0.001, 0.001, 0.001]` in `robot_config.yaml`;
- link-frame origin at the proximal joint axis;
- local +x along the link toward the next joint;
- the mesh should include the joint housing and motor for that link;
- collision stays primitive unless `cad_meshes.collision_from_mesh: true`.

After adding files, regenerate the URDF and check the "CAD meshes used" line.
