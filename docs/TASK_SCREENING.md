# Task shortlist and pre-training screen — 2026-10-02

**October 7 update:** the [original hidden-prong task is now restored](PLUG_RESTORATION.md).
The dated audit below records the earlier mismatch; the October 2 prototype
geometry remains isolated in the screening harness for reproducibility.

## Correction: the original hidden-prong plug is a different task — 2026-10-05

**Keep the original hidden-prong plug insertion as a primary study candidate.**
The October 2 prototype screen below did not evaluate that task. Applying its
hand-retreat finding to the original plug, or concluding that the original plug
needs a new contact-dependent mechanism to be promising, was incorrect.

Inspection of the user-specified legacy checkout,
`/Users/pembe/Desktop/Projects/ActivePerception/old/active_perception_arms`, establishes
the following distinction:

- Original `specs.py` defines four plugs with identical 64×42×30 mm bodies and
  two prongs shifted by ±15 mm along x or y. The body occludes the offset from
  above. `_reset_plug` sets the required plug-origin position to socket center
  minus the hidden offset. The offset therefore changes the correct action.
- The original position-only manipulation controller holds grasp orientation.
  Raising the hand moves the wrist camera and held plug together; it does not
  reveal the underside in the way a hand lift reveals our independent marker.
  Physical contact probing remains an alternative information channel.
- Original variants have equalized mass/inertia to remove an identified
  variant-dependent sag cue. The legacy sensors disable shadows. These details
  are part of the original perception problem and must survive a faithful port.
- This checkout's `scenes.py` instead builds one centered rectangular pin. Its
  screening harness hides the socket and queries a separate green marker. Neither
  its 522-view search nor its retreat experiment tests the original four-prong-
  layout identification problem. The currently published prototype videos also
  depict that different geometry.

The later legacy `docs/WORK_LOG.md`, "Closing 2026-09-30", reports the following
balanced historical re-evaluation: 3 training seeds, 400 episodes per cell.

| Legacy condition | Reported equal-seed mean success |
|---|---:|
| Wrist only | 80.7% |
| Fixed camera at home + wrist | 60.1% |
| Active camera + wrist | 99.5% |
| Active-trained policy with camera frozen | 67.1% |

These are **reported legacy results**, not new runs or independently reproduced
measurements. The task README's older "pl4 pending" paragraph is superseded by
this later work log. The raw evaluation/checkpoint files referenced by the log
are absent from the supplied legacy checkout. The log records camera collisions
in 4% of episodes for active seed 0 and none for seeds 1–2; the latter still lose
success when their camera is frozen. Freezing also changes the trained policy's
inputs/behavior distribution, so it does not establish that every alternative
policy requires motion.

The unresolved comparison is **best searched fixed sensing and initial inspection
with memory**, not whether the original task already has promising active-camera
evidence. The legacy `docs/LAYOUT_DECISION_MEMO.md`, lines 44–56, reports lower
fixed viewpoints that see all four variants in sampled socket cells. Its static
training search was still a next step. The hidden prong offset is constant during
an episode, so inspecting it once and retaining that information is a meaningful
baseline. Our broad search over the different marker prototype cannot settle
either comparison for the original task.

Source locations in the legacy checkout: `tasks/insertion/specs.py:210` and
`:268`, `tasks/insertion/mdp/events.py:849`, `tasks/insertion/insertion_env_cfg.py:680`,
`docs/WORK_LOG.md:262`, and `docs/LAYOUT_DECISION_MEMO.md:44`. Task source paths are
under `src/active_perception_arms/`. No legacy files, physics, policies or current
environment implementation were modified during this inspection. A faithful
hidden-prong port remains outstanding; the screen below must not substitute for it.

## October 2 alternative designs and prototype results

The following shortlist and measurements are retained as results about the
**new marker/enclosure prototypes only**. They do not demote or replace the
original hidden-prong plug benchmark described above.

**Prioritize shrouded connector insertion and contact-dependent seating checks.
Keep open-slot pushing as a negative control.** The screen supports a reason to
change viewpoint in some geometries. It does **not** establish that a learned,
adaptive camera policy will beat a prescribed scan, or that camera motion improves
manipulation success. In particular, lifting the hand defeats the occlusion in the
current prototypes. That result materially changes the proposed task definitions.

No training, optimizer updates, Slurm submission, or hardware execution occurred.
The existing three MjLab/Warp tasks remain unchanged. The new code is a separate
native MuJoCo screening harness, not three newly validated RL environments.

## Recommended study tasks

### 1. Shrouded connector insertion with engagement checked under load

The manipulator inserts a keyed connector into a recessed, slightly floating
receptacle. It must align the connector and confirm the seating gap or retaining
clip while maintaining engagement. The hand and connector obscure the top; side
access changes when nearby clutter or an external obstruction moves. Use a
physically realizable shroud with two side windows in the eventual apparatus.

- **Information needed:** relative alignment after first contact, then the final
  seating/retention state. Initial socket position alone may become stale after
  receptacle movement or contact-induced slip.
- **Why move a camera:** the useful side view can change after approach, while the
  wrist remains above the grasped connector.
- **Critical qualification:** withdrawal must actually change or unload the state
  being assessed. If a robot can back off, look, and reinsert reliably, that is a
  legitimate baseline. Do not prohibit retreat merely to manufacture a benefit.
- **Physical goal for the next implementation:** keyed pin aligned within the
  specified clearance, correct insertion depth and latch state, sustained for a
  settling interval. The current green marker is not this contact model.
- **Controls:** rigid versus floating receptacle; static versus changing side
  occlusion; no slip versus slip; long versus short observation budgets; hand
  retreat and probing; higher image resolution. Start with the same two-arm
  platform, clearance, action rate and sensing channels for every policy.

This is the strongest **candidate to develop next**, rather than a task that has
already passed the manipulation-success gate. The prototype rotates a housing
between prescribed query states; a deployable task needs physical shutter/clutter
motion and contact-generated state changes instead of these interventions.

### 2. Sequential press-fit or snap-fit seating at two inspection locations

The manipulator seats a component at a source fixture, then seats it at a second
fixture or engages a second tab whose inspection direction differs. For a physical
study, a two-tab cover or two retaining clips is preferable to ordinary tray-to-cubby
transfer: the queried information is a local seating/engagement state, not merely
the already visible object center.

- **Information needed:** the result of each seating operation, generated after
  contact at that location. Both operations must matter to final success.
- **Why move a camera:** source and destination windows can require different
  views. The prototype uses two stationary enclosures, about 16 cm apart, rather
  than relocating one enclosure between queries.
- **What this tests well:** continued movement between manipulation phases.
- **What it does not yet test well:** online camera adaptation. A known phase route
  performs as well as the sampled-view oracle in the main visibility screen.
- **Physical goal for the next implementation:** both prescribed fits or tabs
  mechanically engaged; blind pressing, self-alignment and probing must remain
  allowed and be evaluated. A seating indicator alone must not define success.
- **Controls:** predictable phase order versus randomized obstruction; rigid versus
  compliant seating; initial-only observation; phase schedule with camera stow
  before hand relocation; retreat/regrasp. Only use a loaded snap-fit variant if
  unloading genuinely changes the relevant state on the real mechanism.

The present held-plug proxy does not implement snap-fit contact or a grasp-transfer
sequence. Treat this as a movement control and a second candidate for the physical
study, subject to mechanical validation.

### 3. Open-slot pushing as an intentional negative control

Push a block along an open trough with the existing rigid pushing tip. Include
unpredictable object displacement and temporary occlusion as variants. The wrist
has direct access to the task in the screened geometry. A camera-motion benefit
should not be expected merely because pushing is precise or the object moves.

Keep this task in the three-task simulated study to test unnecessary camera motion,
time/action cost, and robustness. Do not describe it as a positive active-perception
benchmark. Ordinary open transfer is similarly weak evidence and should remain a
reference rather than the main physical demonstration.

## What was executed

Six visibility prototypes use the supplied white SO-101 assets, physical enclosure
walls and matched 48.46-degree FOV. Insertion/seating prototypes use the actual held
plug footprint; the pushing control uses the actual pushing tip and block.

Each episode has an initial observation and two subsequent information queries.
For persistent-state controls, earlier observations remain valid. For fresh-state
variants, an independently sampled offset is revealed after each phase change;
the previous offset is not valid evidence about the new one. Offset changes do not
leak into the prescribed hand joint positions. These are explicit information
interventions, not measurements of contact-induced slip in a physics rollout.

The main screen uses **24 validation seeds (12000–12023), 48 held-out screen seeds
(41000–41047), 522 fixed poses per candidate, and 32 sampled camera-arm endpoints**.
The fixed search includes overhead, a broad azimuth/height/radius grid, every active
endpoint, and 104 local refinements around four spatially separated validation
leaders. Both the refinement and selected view use validation data only. This is
a substantial finite search, not a proof of the global best static pose.

The endpoint catalog comes from an IK grid with position/aim checks and collision
checks against the other arm and environment. It is not the whole reachable
workspace. The fixed camera is allowed outside that workspace. Both arms and their
optical geometry remain present; the spare arm is parked for fixed-camera checks.

Visibility means at least **three marker pixels**, computed by rays through image
pixel centers at 96×72. Tests compare these counts with OpenGL segmentation.
Counterfactual images move only the marker by 6 mm, retaining the hand and scene,
and report both direct marker pixels and any RGB changes, including indirect
effects such as shadows. Three pixels establish neither reliable pose estimation
nor the necessity of different manipulation actions.

## Main results: both information queries observable

Percentages below are **visibility-proxy coverage, not task success**. “Initial
oracle” can choose one fixed view with knowledge of both future query states, so it
is an intentionally strong upper bound for selecting an initial view. “Moving
oracle” chooses among the 32 sampled endpoints separately at each query; it is not
an upper bound over continuous motion. Phase routes and scans are selected on
validation. Endpoint scores below omit travel time.

| Prototype | Wrist + valid memory | Searched fixed only | Searched fixed + wrist | Initial oracle, any fixed pose + wrist | Repeated two-view scan + wrist | Moving endpoint oracle + wrist |
|---|---:|---:|---:|---:|---:|---:|
| Open reference | 93.8% | 100% | 100% | 100% | 100% | 100% |
| Static recess, unchanged state | 93.8% | 100% | 93.8% | 100% | 93.8% | 100% |
| Static recess, fresh offset | 0% | 52.1% | 52.1% | 97.9% | 60.4% | 72.9% |
| Changing side access, fresh offset | 0% | 16.7% | 16.7% | 43.8% | 64.6% | 70.8% |
| Two-site seating checks | 0% | 0% | 0% | 0% | 83.3% | 83.3% |
| Open-slot pushing | 100% | 100% | 100% | 100% | 100% | 100% |

Each episode contributes 2.08 percentage points. These exploratory samples do not
establish a statistically reliable advantage for adaptation. Static-only and
static+wrist select their views separately: a validation tie can select a poorer
held-out view for the combined sensor. Do not interpret that as harm from adding
wrist sensing. In particular, the static recess is a negative result because a
searched static-only view already covers every episode.

The initial oracle restricted to camera-arm endpoints covers 35.4% for changing
side access, compared with 70.8% for independently selected endpoints. This is
evidence for potentially useful **continued movement** in the proxy. However, a
repeated scan already covers 64.6%, leaving only three episodes of endpoint-oracle
headroom over that stronger baseline. For two-site seating, one predefined view
per phase already achieves the same 83.3% as the endpoint oracle.

## Controls that limit the interpretation

**Retreat:** raising the hand 45 mm reveals both queried states in all 48 episodes
of every candidate, with no detected hand/environment penetrations deeper than
1 mm along the sampled lift. This is a major competing strategy. The present
prototypes therefore do not establish a need for a second moving camera when the
manipulator may change its own viewpoint. Contact-dependent task definitions above
are proposals requiring physical validation, not a result already demonstrated.

**Waiting:** a persistent housing does not clear simply because time passes. When
the housing is removed as an optimistic temporary-occlusion control, fixed+wrist
recovers both fresh queries in 47/48 changing-access episodes. This is an endpoint
check after removal, not a timed waiting-policy evaluation. Do not claim a benefit
over waiting without measuring matched completion-time budgets and real occluder
durations.

**Motion and schedules:** joint paths are sampled at at most 0.035 rad per command,
with the existing 25 Hz control rate. A continuous blind scan can see useful
intermediate views absent from the endpoint catalog: it covers 75% of the changing-
access episodes, exceeding the sampled-endpoint oracle's 70.8%. Its median cycle
is 3.0 s. The privileged route's median duration to its chosen endpoint is 0.96 s;
that is a planning diagnostic, not a measured speed advantage of a visual policy.

For two-site seating, blindly visiting the next site's view too early can hit the
manipulator. A phase schedule that stows before hand relocation recovers 83.3%
coverage; its sampled paths are feasible in 46/48 episodes. A naive scan's collision
failures are not evidence for adaptive perception. Selected native servo videos
also check actual dynamics; the unstowed two-site failure is retained in the media
manifest. Sampled path checks are not continuous swept-volume guarantees or a
hardware safety validation.

The corrected two-site native demonstration stows before each prescribed hand
relocation, records zero camera collision frames, and sees the marker during every
phase. This one-seed correction verifies that an orderly camera route can execute;
it does not validate the eventual snap-fit/transfer mechanics.

**Resolution:** at matched 192×144, the changing-access fixed+wrist score is 22.9%,
the repeated scan 97.9%, and the endpoint oracle 100%. Two-site fixed+wrist rises
from 0% to 45.8%; the phase schedule and endpoint oracle both reach 100%. The original
two-site zero therefore mixes occlusion and insufficient pixel detail. Higher
resolution also gives the fresh-offset/static-aperture initial endpoint oracle
100%, reinforcing the conclusion that good initial positioning can suffice there.

**Memory:** without memory, the open reference's selected fixed+wrist view covers
33.3% of both later queries; retaining the initial unchanged state raises coverage
to 100%. Fresh offsets invalidate that particular old observation by construction.
This validates the information-epoch bookkeeping, not the effectiveness of a GRU
or the impossibility of learning a useful dynamics model in a real task.

## Gates before launching comparative RL

1. Implement the shortlisted connector/clip mechanics with a fully observed
   diagnostic controller. Demonstrate real contact success across perturbed
   states; prevent neither probing nor retreat unless the physical task demands it.
2. Construct counterfactual physical states that require different corrective
   actions. Verify that wrist+searched-static histories cannot resolve them soon
   enough, while a feasible camera path can. Marker visibility alone is insufficient.
3. Run an actual image-based estimator/controller, plus initial-only, memory,
   waiting, retreat and phase/scan baselines. Evaluate camera travel and elapsed
   time without tuning deadlines to produce a desired success gap.
4. Only then register the surviving variants in MjLab/Warp and run matched RL
   budgets using the existing cluster infrastructure. Keep clean/open negatives.
   Separately test freeze, image hold, replay, recurrent versus feedforward actors,
   and active-trained policies under frozen-camera evaluation.

The current evidence justifies a **shortlist and a falsifiable next experiment**,
not GPU training on three purportedly active-camera-dependent tasks. Joint visual
control already has precedent, including [Cheng et al., 2018](https://proceedings.mlr.press/v87/cheng18a.html).
Viewpoint selection for manipulation also appears in [Observe Then Act](https://arxiv.org/abs/2409.14891).
The study's contribution should remain the controlled comparison and separation of
mechanisms, rather than the existence of a camera arm or recurrent actor.

## Reproduction and artifacts

```bash
.venv/bin/python -m active_perception_arms.task_screening \
  --episodes 24 --test-episodes 48 --validation-seed 12000 --test-seed 41000 \
  --no-render --output artifacts/task_screening/pixel_screen
.venv/bin/python -m active_perception_arms.screening_diagnostics \
  --report artifacts/task_screening/pixel_screen/screening.json \
  --episodes 48 --output artifacts/task_screening/diagnostics
.venv/bin/python -m active_perception_arms.task_screening \
  --episodes 24 --test-episodes 48 --validation-seed 12000 --test-seed 41000 \
  --width 192 --height 144 --no-render \
  --candidates recess_with_slip shuttered_insertion two_site_seating \
  --output artifacts/task_screening/resolution_192
```

The screen runs on CPU without a renderer when `--no-render` is given. Diagnostics
use native MuJoCo OpenGL for the 1920×1080 external videos and 96×72 sensor images.
The scripts never invoke a trainer. See the [media and raw reports](../artifacts/task_screening/README.md),
[progress log](../PROGRESS.md), and [test log](../artifacts/tests.log).
