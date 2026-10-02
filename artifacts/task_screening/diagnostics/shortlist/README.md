# Videos of the three shortlisted prototypes

Each scene has a **1920×1080 outside view at 25 fps** and a synchronized comparison
of the wrist, validation-selected fixed camera, and scripted camera arm. Sensor
images are 96×72, enlarged for viewing. Both robot arms are white.

| Scene | Outside video | Camera comparison |
|---|---|---|
| Shrouded connector insertion prototype | [Watch](shuttered_insertion_overview.mp4) | [Watch](shuttered_insertion_sensors.mp4) |
| Two-site seating prototype | [Watch](two_site_seating_overview.mp4) | [Watch](two_site_seating_sensors.mp4) |
| Open-slot pushing prototype | [Watch](open_slot_push_overview.mp4) | [Watch](open_slot_push_sensors.mp4) |

These show the **current visibility prototypes**, with scripted native MuJoCo camera
motion and prescribed hand/object query states. They do not show completed
contact-dependent connector insertion, snap-fit assembly, or a learned policy.
The camera returns to its home pose before each hand/scene phase change.

All three use seed 41000, the first screen test seed. The one-episode diagnostics
here describe the recordings; use the [main study report](../../../../docs/TASK_SCREENING.md)
for the multi-episode comparisons and task-readiness limitations.

[Recorded servo checks](rendered_checks.json) · [Encoding and playback checks](video_validation.json)

To reproduce:

```bash
.venv/bin/python -u -m active_perception_arms.screening_diagnostics \
  --report artifacts/task_screening/pixel_screen/screening.json \
  --output artifacts/task_screening/diagnostics/shortlist --episodes 1 \
  --candidates shuttered_insertion two_site_seating open_slot_push
```
