# Study protocol and interpretation

The hypothesis is conditional: moving the camera may help when relevant information
changes or is hidden. Precision alone does not establish a need for camera motion.
Remembering, waiting, and finding a good initial view are competing explanations.
A successful script only establishes mechanical feasibility.

The [pre-training task screen](TASK_SCREENING.md) records the current shortlist,
visibility experiments and rejected explanations. It supersedes the tentative
physical-task suggestions below: ordinary tray-to-cubby transfer is too weak a
positive test, and the proposed contact-dependent seating variants still require
mechanical validation.

The October 5 correction in that report restores the original hidden-prong plug
as a primary candidate. The earlier centered-pin/marker prototypes did not test
the legacy task's unknown prong offset, and their retreat result cannot be used
to dismiss its reported active-camera benefit. Best-static and initial-inspection
comparisons for the original task remain open. The [faithful task port](PLUG_RESTORATION.md)
is now implemented and mechanically checked. The [seed-0 cluster pilots](../artifacts/cluster_pilot/README.md)
are completed exploratory runs; sparse success and evaluation-repeat divergence
preclude a statistically supported sensing comparison.

The [literature review](LITERATURE_POSITIONING.md) audits the four supplied papers
and related active-VLA/memory work. It narrows the prospective contribution to the
measured value of fresh physical observations under strong sensing and memory
controls; spare-arm vision, RL gaze, random occlusion, and memory are established.

## Comparisons

Keep randomization, resolution/FOV, control rate, reward, elapsed-time budget,
training transitions, encoder, and seeds matched. The first second gives every
condition the same manipulation pause. Separate the best searched external camera
from an external camera placed at the camera arm's starting pose. Include overhead
in the search, even if it is difficult for the spare arm to reach.

1. Wrist, searched static, and searched wrist+static establish strong conventional
   sensing baselines. Select views using validation success across three training
   seeds; report search size and total training cost.
2. Initial-only camera movement tests whether selecting a view is sufficient.
3. Scheduled movement tests moving viewpoints without online camera feedback.
4. Active movement tests closed-loop, observation-dependent camera control.

Use the same held-out reset seeds and evaluation batch size. Report each training
seed's results and uncertainty across seeds, in addition to per-checkpoint Wilson
intervals over episodes. Preserve zero improvements and regressions in summaries.
The 26-view grid is a finite search, not proof of a globally optimal fixed camera.

## Mechanism controls

- **Continued camera movement:** freeze camera commands after the initial second,
  keep images live. Compare with separately trained initial-only policies; the
  test-only freeze also introduces distribution shift.
- **Continued external feedback:** hold the external RGB frame after a chosen time,
  leaving wrist feedback live. Combine with camera freeze for an initial-snapshot
  condition, and acknowledge the changed input distribution.
- **Online camera adaptation:** replay camera commands recorded on different
  held-out episodes. Compare on the same destination seed with a normal active
  evaluation. Replay may be unsafe or collide; retain these outcomes.
- **Memory:** train feedforward and recurrent actors. Resetting GRU state at every
  evaluation step is an additional intervention, not a substitute for retraining.
- **Training-time benefit:** cross-evaluate each active-trained actor with camera
  freeze/replay. Compare to policies trained with static or initial-only sensing
  at matched budgets. These tests alone do not perfectly isolate representation
  learning from policy distribution shift; do not claim they do.

## Occlusion and stale information

`clean`: no panel. `static`: panel present throughout. `phase`: panel present
from 3 to 7 seconds. `random`: onset sampled uniformly from 2–5 seconds and duration
from 1–4 seconds; lateral side is randomized. These timers are unavailable to the
actor except indirectly through images. The phase condition is tied to time, not
to privileged task completion; a slow learned policy may encounter it in a
different manipulation phase. The optical panel deliberately has no contact.
Physical fixture walls and the hand produce additional natural occlusion.

For the restored plug, clean is the default: its occlusion comes from the plug
body and hand. Its 3.5-second horizon means the old 3–7 second panel schedule is
not the main plug experiment. Panel modes remain explicit supplementary controls.

Clean episodes are essential controls. The random occluder can disappear
before the 12-second deadline, so waiting is a valid strategy; do not remove that
possibility to force a positive result. Compare completion time and camera travel
as well as success. Add shorter deadlines or longer occlusion only as explicit,
matched experiment variants, with their sensitivity results reported.

`--perturb-push` applies a 0.025 N lateral force for 0.12 seconds at t=4 seconds.
It tests the possibility that old object information becomes stale. It is a small,
fixed-time intervention, not proof that memory cannot suffice; measure the actual
object displacement and visibility before interpreting the result. For stronger
experiments, randomize disturbance time/magnitude in both train and test with a
predeclared distribution rather than tuning to maximize the active-policy gap.

## Current task limitations

- Plug: pregrasped rigid fixture, no cable or insertion force sensing. The 2 mm
  clearance is a starting setting; rerun both physics sanity and pixel-resolution
  checks if changing it. The socket has fixed orientation and XY uniform in [0, 50] mm². Four
  hidden offsets of ±15 mm change the required body position. The wrist has a
  small direct cue for one variant at the checked reset pose; three remain
  pixel-identical there. See the restoration audit for the measured scope. Success is not an electrical connection test.
- Transfer: the supplied camera housing needs a tall open cubby for access. The
  wrist has a strong view of grasping; movement may offer little advantage on clean
  episodes. The walls are physical and can exclude some searched static views.
- Push: a visible 8 mm radius rigid tip is mounted below the closed gripper; it
  avoids unstable side impacts from bare jaw edges. Current geometry intentionally provides a simple negative-control task
  where wrist memory/feedback may be enough. The optional disturbance is small.
- Randomization is limited to object/fixture XY and panel timing/side. No domain
  randomization of optics, mass, friction, or lighting has been claimed or added.
- Active control has more action dimensions and different embodiment than a fixed
  camera. Report these differences. Static cameras are permitted anywhere in the
  published grid, rather than being artificially limited to reachable arm poses.
- Camera motion has a small action penalty through the common reward. It has no
  separate learned safety layer. Both robots have collision geometry; adding
  hardware requires calibrated limits, collision checks, and real-camera validation.

Suggested physical follow-ups are plug insertion and tray-to-cubby transfer under
an externally moved screen. This repository supplies simulation and cluster jobs;
it does not deploy policies or control physical hardware.
