# Asset provenance

The SO-101 XML, mesh assets, original LICENSE, NOTICE.md, and UPSTREAM_README.md
were copied from the user-provided `old/active_perception_arms` repository.
Its notice identifies MuJoCo Menagerie commit
`71f066ad0be9cd271f7ed58c030243ef157af9f4` (Apache-2.0).

The generic SO101Chain kinematics implementation was reused from that project's
`robots/so101_kinematics.py`. The new environments do not import or depend on the
old checkout. Changes to the camera FOV, materials, collision groups, and scene
geometry are applied by the new scene builder, keeping the vendored XML intact.

Simulation stack: MjLab 1.4.0 / RSL-RL 5.2.0, with stable MuJoCo 3.9.0 and the
MuJoCo Warp commit used by the earlier project. This intentionally uses a pinned
stack rather than tracking MjLab main. `uv.lock` records full dependency versions.

The copied `NOTICE.md` describes legacy FOV/material settings. In this project
**all** rendered cameras use 48.45549° vertical FOV, 96×72 pixels by default, and
yellow arm materials. Gripper friction and the camera mounting geometry remain
those of the supplied model. The optical intervention panel has collision disabled;
robot links, plug/socket, cube, tray, cubby, and table have physical collisions.
The plug is deliberately pregrasped with a weld; this task tests insertion,
not grasp acquisition, cable dynamics, or electrical connection.

For pushing only, the scene builder adds a visible 8 mm radius, 40 mm long rigid
cylindrical tip below the gripper TCP (5 g). The robot XML itself stays unchanged.
