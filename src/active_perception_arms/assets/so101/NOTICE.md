# SO-101 model provenance and modifications

`so101_arm.xml` and `assets/` are vendored from
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
`robotstudio_so101` at commit `71f066ad0be9cd271f7ed58c030243ef157af9f4`
(2026-07-04), Apache-2.0 (see `LICENSE`, `UPSTREAM_README.md`).

Modifications relative to upstream `so101.xml` (renamed `so101_arm.xml`):

1. Removed scene-level `<option>` and `<visual>` elements — simulation options
   are owned by the mjlab `SimulationCfg` of the composed scene.
2. `wrist_cam`: replaced `resolution/sensorsize/focal` with `fovy="71"`.
   The MuJoCo-Warp batched renderer consumes `fovy`; 71° matches the
   sim2real-tuned wrist-camera FOV of the squint reference setup
   (aalmuzairee/squint, MIT) and is what the legacy benchmark modes (top /
   lateral / search / stack / awning / plug) were trained with. The upstream
   intrinsics (`resolution="1920 1080" sensorsize="0.00576 0.00324"
   focal="0.0036 0.0036"`, i.e. 48.46° vertical FOV) are restored at
   spec-build time for the empty-hand modes (`transfer`, `lidbox`) on both
   arms via `make_so101_spec(camera_fovy=WRIST_CAM_FOVY_MENAGERIE)`, rendered
   as 4:3 frames (128x96 by default; horizontal ~61.9°, the LeRobot 640x480
   crop). See `so101_constants.py` (`WRIST_CAM_FOVY_MENAGERIE`).
3. Added site `wrist_cam_site` co-located and co-oriented with `wrist_cam`
   (the site's −z axis is the camera optical axis). Used as the IK frame for
   the camera arm.
4. Gripper collision classes (`collision_gripper`, `collision_gripper_mesh`):
   sliding friction raised 1.0 → 2.0 (squint's gripper material), torsional /
   rolling components unchanged.

5. Link materials recolored from TheRobotStudio yellow (1 0.82 0.12) to
   white (0.9 0.9 0.92) for visual distinction from the yellow task objects.
