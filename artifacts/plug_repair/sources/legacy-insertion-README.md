# Insertion task modes

Modes of the two-arm insertion environment (see the repo README for the base
task). This file documents the two benchmark modes: **`stack`** (block-on-block
placement behind a see-around barrier) and **`plug`** — both designed so that
the manipulator's own wrist camera *structurally cannot* observe the
task-relevant state, making the external camera arm necessary.

## `stack` — three-condition occlusion benchmark (geometry of 2026-07-31)

Place grasped **block A** onto free-standing, pushable **block B**. Success =
A's bottom-center within tolerance of B's LIVE top-center; 5 s episodes;
reward `-log(100*d + 1)`, `+1000` on success. Both robot bases are the
canonical placements (manipulator + `SIDE_CAM_BASE_POS` camera arm,
`modes.py`).

> **History**: the 2026-07-30 geometry (single deterministic plank at 40–50 %
> of the camera distance, min-sep 8 cm, 2 mm tolerance) crushed the condition
> separation (wrist 0.495 / static 0.520 / active 0.560; no tau ≤ 15 mm
> separated them). Diagnosis: (L1) the plank's pose deterministically encoded
> B (invertible to ±2 cm along a known ray, visible to every camera) and
> (L2) B leaked into the wrist view at reset and through the 4–10 cm approach
> band. The 2026-07-31 redesign fixes L1 (fake planks) and the reset part of
> L2 (min-sep 0.10); the approach-band leak is accepted pending the probe.

### Geometry (constants in `so101_constants.py`, spec in `specs.py`)

| element | value | why |
|---|---|---|
| block A | 60 × 60 × 30 mm slab + holder1 grasp collar on top (`make_stack_block_a_spec`), origin at slab bottom-center | BIGGER than B: held under the top-down wrist camera it fills the view corridor, so B is invisible from the wrist view during descent (the old A — the 30 mm peg — leaked B around it). Collar at the exact peg position ⇒ `GRIP_*` constants + weld relpose unchanged (hold/transport probes 100 %/100 %, drift 0.3 mm; `scripts/check_stack_grasp.py`) |
| block B | 30 mm cube, mass 60 g, free body | sampled over `[-0.03, 0.06] × [-0.03, 0.08]`, ≥ **10 cm** from A (JOINT (A, B) rejection; kills the reset-frame wrist leak, measured 0.35 → 0.085; documented cost: A↔B anticorrelation ≈ −0.7, B|A residual ≈ 3.1 cm RMS, B-box center depleted) |
| barrier 1 (true plank) | vision-only plank **3 cm wide × 21 cm tall**, on the camera-home→B sightline at **55–65 %** of the horizontal camera distance | from the frozen side home B's FULL silhouette is blocked for 100 % of B positions (vert margin 24–48 mm, lat 8.2–9.8 mm), yet a clear in-gimbal-box pose to B always exists (clear-pose share 35–56 %). The old 7 × 10 cm plate at fixed 0.07 m blocked 0 % — no occlusion pressure |
| barriers 2–4 (fake planks) | 3 identical planks on sightlines of **fake B samples from the same law** (same frac, same yaw), unconstrained per fake | a single plank's pose no longer identifies B (true plank is an exchangeable 1-of-4; marginal TV 0.014 ≈ noise). Episode-level solvability: plank sets that wall off every in-box view of B are resampled whole (round-1 infeasibility 0.6 %, fallback ~0) |

All verified in `scripts/check_stack_geometry.py` (gates (a), (b1), (b2), (c)
— the Monte-Carlo runs through the REAL sampler `events.sample_stack_fake_planks`)
plus rendered diagnostics (`scripts/render_stack_diagnostics.py`).

### Three-condition protocol

All `--insertion-mode stack --im-size 64`, 1024 envs, 2000 iters:

- **C1 wrist_only** — `--task vision --views wrist` (3-ch image, 3-D action)
- **C2 static_cam** — `--task vision` (dual [6,64,64], camera arm frozen at home)
- **C3 active_cam** — `--task active_vision` (dual + camera-pose obs, 8-D action)

Eval (`scripts/eval_stack_three_conditions.py`): ≥ 200 episodes/cell, 50 envs,
`episode_length_s=5.0`, success = terminated & !timeout, Wilson-95; C3 also
re-run with the gimbal action zeroed (frozen camera); C2/C3 record a
geometric-LOS visibility proxy and camera-motion stats;
`scripts/eval_stack_tolerance_sweep.py` records per-episode min |A_bottom −
B_top| for post-hoc tolerance rescoring. A `--task proprio` oracle run
calibrates the dynamics/control ceiling per tolerance.

### Finding: active perception as self-curriculum (st5, 2026-08-04)

The identity redesign (decoy cubes + a green mark on the true B's +y face,
2.5 s budget, 96 px) produced an airtight C1/C3 separation — wrist-only
**0.285** [0.227, 0.351] vs active **1.000** [0.981, 1.000] — but the
channel-attribution suite shows the active policy does NOT use the camera
arm at execution time: frozen-camera **0.950**, cam-arm-view-blacked
**0.955**, while suppressing only green-dominant pixels in the WRIST stream
collapses it to **0.215** (the wrist-trained policy is unchanged by the same
mask: 0.330 → 0.320, and barely uses its image at all — full blackout costs
it 6 points). The mark leaks to the wrist camera as a peripheral grazing
cue (its world-fixed optical axis (−0.54, 0, −0.84) reaches +y faces at
≥ 37° azimuthal incidence at the FOV edge), and the active-TRAINED policy
reads it there: 12-step straight-line solves, camera motion ~0.

The C1↔C3 gap is therefore a **learnability gap, not an execution-time
information gap**: the camera arm's easy frontal view of the mark early in
training bootstraps the "green = target" concept (a discovery jump 0.14 →
0.73 between iterations 100–250), which then distills into the cheap wrist
channel; the wrist-only condition, facing the same eval-time information,
never bootstraps (plateau ~0.31 across 2000 iterations). Active perception
acted as a self-curriculum for a passive skill — a teacher-student
distillation the policy performed on itself. The green-mask attribution
cells (`scripts/eval_stack_channel_attribution.py`) are a permanent part of
the eval suite.

A follow-up design (identity mark readable ONLY by the arm: a louvered
"snorkel" hood; `scripts/check_stack_snorkel.py`) was gate-checked and
STOPPED before training: the direction-space bound is achievable (99.7% of
episodes have a wrist-safe, plank-clear arm ray) but passive angular
filtering at 3 cm cube scale measured out at {wrist-leak 0 px ↔ arm
coverage 21–34%} / {coverage 58–90% ↔ leak 1.7–4.8 px} across ten tuned
variants — never both. The benchmark's execution-time causal claim needs a
mechanically different identity carrier; the curriculum finding stands on
its own.

**Status:** st5 checkpoints + full attribution under
`results/stack/three_conditions/`; the full 3×3 is on hold pending a task
variant whose identity channel is arm-exclusive at execution time.

## `plug` — hidden prong offset

<p align="center">
  <img src="media/plug_scene.jpg" width="640"
       alt="Manipulator holding the plug over the socket plate, camera arm at its viewing home">
</p>

A wall plug is scripted-grasped by the manipulator (same collar grasp and weld
as the pegs) and must be inserted into a two-hole socket plate on the floor.
The plug's **prong pair hangs off-center** beneath the body: four variants
(pair center shifted ±15 mm along x or y), assigned round-robin per env via
the mesh-variant machinery. All variants share an identical body whose
footprint covers every prong position, so **from above the variants are
indistinguishable** — the manipulator's wrist camera sees the same white top
face no matter which plug it holds. Only the camera arm sees the prongs.

| side (external view) | bottom (never visible to the wrist camera) |
|---|---|
| ![variants side](media/plug_variants_side.jpg) | ![variants bottom](media/plug_variants_bottom.jpg) |

![socket and alignment](media/socket_and_alignment.jpg)

*Left to right: socket plate; top view; plug correctly offset over the socket;
seated at the success pose.*

![policy views](media/plug_policy_views.jpg)

*What the policy sees: camera-arm view (256 px render), manipulator wrist view
(only the plug's top face — the prongs and their offset are invisible), and
the actual 64 px camera-arm observation.*

### Why this forces external perception

- The policy has no yaw control (the grasp weld pins orientation), so the
  hidden state is a pure xy offset of the prong pair.
- The socket's holes are always at the plate center ± the prong spacing:
  seeing the plate reveals *where the socket is* but nothing about *which
  plug is held*. Wrist-cam shadows are off in the render sensors.
- Success requires placing the plug origin at `socket_center − offset`
  (within the standard 2 mm), i.e. reading the offset from the camera-arm
  image. A blind policy can only guess 1-of-4 (≈25 % ceiling).
- The y-offsets lie along the side camera's optical axis (depth); resolving
  them hinges on the near-side prong pair peeking below the body — a signal
  whose size the camera controls by moving (see "Camera geometry" below).
  The x-offsets are resolved by wrist-view contact probing (their prongs are
  structurally occluded from the camera arm; see below).

### Geometry (constants in `specs.py`)

| element | value | note |
|---|---|---|
| body | 64 × 42 × 30 mm, origin at bottom-center | covers all prong layouts; holder1 collar on top (z 30–60 mm) as for pegs |
| prongs | 8 × 8 mm section, 12 mm long, 22 mm apart | per-variant **meshes** (offset baked into vertices) |
| pair offsets | (±15, 0), (0, ±15) mm | `PLUG_PRONG_OFFSETS`, round-robin `env i → variant i % 4` |
| socket plate | 100 × 70 × 14 mm, exact 5-box decomposition | mocap entity, placed over the base hole box `[0, 0.05]²` each reset |
| holes | 12 mm square at ±11 mm, through | 2 mm/side clearance (pegs run 1.25 mm; the plug swings slightly on the compliant weld during transport, and 12 mm prongs / 14 mm plate keep ≈9 mm tip clearance in flight) |
| success | origin within 2 mm of `socket − offset`, z = 16 mm | prongs ≥10 mm engaged; reuses `insertion_reward` / `insertion_success` unchanged — the reset event folds the offset into `state.target_pos`, so the privileged critic knows it and the actor must perceive it |

### Validation gates

`Scripted probe` (ramped traverse → settle → slow descend → gentle press),
16 envs, all 4 variants:

- **oracle** (knows the per-env offset): **16/16** success, two seeds
- **blind** (aims at the socket center): **0/16**

Note when probing manually: success TERMINATES the episode and the env
auto-resets inside `env.step()` — accumulate the step's `terminated` return,
not a post-step `insertion_success(env)` read, which sees the fresh episode.

### Camera geometry (2026-07-30 fix: the "vertical variation" failure)

The pre-fix evidence (64 px smoke run + `results/plug_insertion/*.json`):
xm/xp ≈ 100 % but **ym 9/160, yp 5/160** — the y-offset variants were
unsolved. Two mechanisms, both verified by rendered diagnostics
(`scripts/diagnose_plug_camera.py`, outputs under
`results/plug_insertion/three_conditions/diagnostics/`):

1. **Depth axis**: from the side base, world-y is the camera's optical axis;
   ym/yp project onto the vertical image direction with almost no lateral
   signal. (The task has **no z-variation** — the socket sits on the floor;
   "vertical variation" = these depth-axis variants.)
2. **Slab occlusion**: a 12 mm prong `d` mm behind the body's near-bottom edge
   is visible only below `atan(12/d)` depression. From the fixed side base
   (reach floor ≈ z 0.175) the yp pair (`d` = 2 mm) peeks below the body; the
   ym pair (`d` = 32 mm) and both x-pairs (`d` = 17 mm) never do — at ANY
   resolution (0–1 changed px even at 256 px). So the only camera-visible
   per-variant signal is **yp presence vs ym absence**, and its pixel size
   (∝ 1/distance) is the design knob.

Fix (bases unchanged; `PLUG_*` in `modes.py`):

| element | value | evidence |
|---|---|---|
| gimbal home | `(0, 0.19, 0.22, −1.42, 0.837)` — D ≈ 0.25 m, aimed at the workspace | 43/43 probe reach (unit test); servo gate max 0.5 mm / 0.14° |
| gimbal box | home ± the standard half-ranges `(±5, ±5, ±4.5 cm, ±0.42, ±0.27 rad)` | close/lateral poses reach D ≈ 0.20 m |
| action box | +y headroom trimmed `0.095 → 0.076` (task needs ≤ 0.0735) | standoff: worst inter-arm clearance −9.9 mm @ 4/43 poses vs the accepted stack geometry's −15.8 mm @ 2/43 |
| im_size | **96** (all conditions) | at 64 px the yp notch is ~1 px (5.8–6.7 mm/px at home) — the magnitude that provably failed to learn; at 96 px: home 4–8 changed px (partial), close/left 8–13 px (decisive) |

Discriminability gate (rendered, 96 px, 3×3 socket grid × 4 variants):
ym-vs-yp resolvable (≥ 6 changed px + yp-present/ym-absent) for **every** grid
cell from at least one reachable pose — PASS; static home partial by design
(worst cell 4 px), so camera motion carries the decisive margin.

### The sag side channel (2026-08-01 fix)

The seed-0 probing autopsy (`scripts/analyze_plug_probing.py`; results under
`results/plug_insertion/three_conditions/diagnostics/`) showed that **no
policy probed**: every condition placed the plug in ONE attempt at ~0.5 s,
including variants whose prongs are provably never camera-visible. The leak:
per-variant geom-derived inertials made the plug **swing differently on the
compliant weld during transport** (up to 74 mm / 17.5° pose divergence between
variants; `scripts/check_plug_sag.py`) — an instant side channel readable from
proprio and the close-range wrist view. Each seed-0 policy memorized a
different 3-of-4 subset of this "sag dictionary" (which explains static <
wrist and the frozen-active/wrist ym/yp mirror in the pre-fix numbers).

Fixes:

- **Equalized inertials**: all four variants share the centered-prong
  reference mass/CoM/inertia (`make_plug_spec`, explicit inertial). Gates:
  compiled inertials bit-identical across variants (unit test); free-air
  transport pose divergence across variants **0.0006 mm / 0.06°** (was
  74 mm / 17.5°; `scripts/check_plug_sag.py` — PASS). Variant-dependent prong
  CONTACT (socket presses, floor grazing at low hover) remains — that is the
  legitimate probing channel, now available only through actual interaction.
- **2.5 s episode budget** (`InsertionModeCfg.episode_length_s`; other modes
  keep 5 s): measured on the seed-0 policies, genuine placements finish
  ≤ 1.1 s (p90) while grind-and-retry probing and drag-until-visible
  substitutes need > 2 s.

### Three-condition protocol (`scripts/plug_three_conditions_queue.sh`)

All `--insertion-mode plug --im-size 96`, 1024 envs, 2000 iters, seeds 0/1/2.
Run names: `pl3_<cond>_s<seed>` (pre-sag-fix, superseded); the fixed task is
first validated by a **probe pair** `pl4_wrist_s0` / `pl4_active_s0`
(`scripts/plug_sag_fix_probe.sh`) whose eval + probing autopsy act as the
decision gate before a full relaunch:

- **C1 wrist_only** — `--task vision --views wrist` (3-ch image, 3-D action)
- **C2 static_cam** — `--task vision` (dual [6,96,96], camera arm frozen at home)
- **C3 active_cam** — `--task active_vision` (dual + camera-pose obs, 8-D action)

Eval (`scripts/eval_plug_three_conditions.py`): ≥ 200 episodes/cell with
**per-variant counts** (≥ 50 each — the xm/xp-vs-ym/yp split is the story),
64 envs (multiple of 4 keeps the round-robin balanced), episodes capped at
the task budget (2.5 s), success = terminated & !timeout, Wilson-95; C3
re-run with the gimbal action zeroed (frozen camera); C2/C3 record the
geometric prong-visibility proxy (`scripts/plug_los.py`) and camera-motion
stats. Plots + self-contained report:
`results/plug_insertion/three_conditions/report.html`
(`scripts/build_plug_three_conditions_report.py`).

Expected structure post-fix: (i) C1 ≈ x-variants via contact only; (ii) C2
partial (home notch on yp); (iii) C3 high via genuine viewpoint use.
**Numbers: PENDING — the pl4 probe pair fills in the report.** Seed-0
PRE-fix numbers (5 s, sag channel open): wrist 0.940 / static 0.862 / active
1.000 (frozen 0.974) — the active policy solved all four variants, but
largely through the sag dictionary, so they overstate perception quality.

### HISTORICAL: smoke-training results and camera ablations (64 px, `orig` placement)

Proprio oracle saturates at 100% (600 iters, 1024 envs). Single-frame
active-vision (500 iters, 512 envs) solves the x-offset variants (~100%) but
not the y-offset variants (~4%) — the y offsets lie along the camera's
optical axis (depth), exactly the component a near-static viewpoint cannot
resolve. Eval-time ablations of that checkpoint (160 eps/bar, Wilson 95%):

![ablations](../../../results/plug_insertion/ablation_bars_64px.png)

Freezing the camera arm's *motion* costs 18–82 points, so the learned camera
behavior is load-bearing; but blanking the camera-arm *image* barely dents the
x-variants — part of that competence flows through the wrist view + proprio
(contact-probing against the 2 mm/side clearance, and possibly the per-variant
CoM shift leaking through servo sag). Blanking the wrist view collapses
everything. Channel-zeroing is OOD for the CNN, so these are behavioral lower
bounds, not exact attribution.

### Camera-arm placements (HISTORICAL — ablation concluded 2026-07)

> The `--plug-cam` / `--stack-cam` placement flags were removed after the
> placement ablation concluded: both benchmark tasks (stack, plug) now share
> ONE canonical camera-arm placement — the side view (`SIDE_CAM_BASE_POS`,
> same riser level as the manipulator; see `modes.py`). The numbers below are
> the record of that ablation; reproduce via git history if needed.

The inherited cam0 home views the workspace from 0.35 m — at 64 px the ±15 mm
prong offset is a ~2 px signal. The ablation compared calibrated placements
(each swept for 100% gimbal-box IK reach; boxes trimmed so the two arms keep
a standoff at their command-box extremes):

![placements](media/plug_cam_placements.jpg)

| name | base | camera home (view dist) | note |
|---|---|---|---|
| `orig` | (−0.30, −0.44) | cam0, 0.35 m | control; the base task's placement |
| `front` | (0.42, 0.025), facing −x | (0.17, 0.025, 0.16), 0.20 m | on the manipulator's y-line, arms face each other; image axes flip: prong y-offsets become lateral, x-offsets become depth. Needs `njmax` 2048 (the face-to-face zero-pose interpenetrates at model build) |
| `side` | (−0.18, −0.36) | (−0.055, −0.115, 0.16), 0.21 m | the cam0 direction, moved ~40% closer |

Pair with `--im-size 256` to give the CNN enough pixels on the prongs.

**Placement comparison (256 px, 256 envs, 700 iters, seed 0; full analysis
under `results/plug_insertion/`, untracked):** variant-balanced success
`orig` 39% > `side` 26% > `front` 22%, all still rising at 700 iters. Each
policy locked onto a different variant subset, so this is a
short-training-budget snapshot, not a geometry verdict. The ablation suite
(160 eps/cell) shows sharply different strategies: `orig` depends on both the
camera-arm image and its motion (xp 61→8% when frozen); `side` is the most
motion-dependent (26→3% frozen — the strongest active-perception signature);
`front` barely uses its camera despite the closest view — its success flows
through wrist/contact strategies. Blanking the wrist view kills every policy
(it carries the fine servo signal). Longer runs and `--frame-history` are the
natural follow-ups.

### Implementation notes

- mjlab's `VariantEntityCfg` batches **only mesh geometry** per world.
  Primitive geoms must be identical across variants — differing primitive
  *positions* pass validation and are silently collapsed to the template
  variant. Hence the prongs are per-variant meshes with the offset baked into
  the vertices (`geom_pos`/inertia are then compiled per variant and
  scattered per world).
- Train / roll out via the standard task ids with `--insertion-mode plug`,
  e.g. `uv run python -m active_perception_arms.rl.train_insertion --task
  active_vision --insertion-mode plug --num-envs 512 --im-size 64`.

## `hinsert` — horizontal insertion into a recessed bore

A receptacle board stands vertically on the table with a **recessed square
bore** on its face; the manipulator, already holding a cylindrical peg, must
insert it **horizontally**. The design object is the bore's **visibility cone**:
a bore of half-width `r_clear` sunk `d_recess` below the face plane exposes its
floor only to viewpoints within `atan(r_clear / d_recess)` of the insertion
axis. `d_recess` is a **mode parameter** (`InsertionModeCfg.d_recess`), not a
constant — sweeping it is how the recess-depth curve is produced.

Two readings are required, and they come from viewpoints that move apart as the
board yaw randomises:

1. **bore pose** — legible only from inside the cone (bore-floor pixels; the
   floor is the only geom in its colour, so the mask is unambiguous);
2. **peg-in-hand** — the peg's shaft hangs off the collar axis by a hidden
   ±6 mm, baked into four per-variant meshes exactly like the plug's prongs.
   6 mm exceeds the bore's 2 mm clearance, so an uncorrected peg strikes the
   face. Inertials are equalised across variants (the plug's sag lesson).

### The geometry is forced by the arm, not chosen

The SO-101's **jaw azimuth equals `−shoulder_pan` exactly** (verified to 1e-15
over 10k random configurations), so a horizontal jaw always lies in the vertical
**shoulder-pan plane**. With the repo's 3-DoF position action the aim must be a
function of the TCP alone — anything else would leak the board pose into the IK
— so the insertion axis is radial and the **board yaw IS the board's azimuth on
an arc**. Board yaw and board *lateral* position are one degree of freedom, not
two; depth (arc radius) and bore height are independent.

`radial_center` (new `InsertionModeCfg` field, `(0, 0)` for every legacy mode)
measures that radial about the **pan axis** rather than the base body origin.
This is not cosmetic: a base-origin reference puts the bore axis
`38.8 mm · |sin(az)|` off the pan plane, which the TCP can absorb only to about
±9 mm — capping the usable azimuth near 0.25 rad, which is exactly
`LATERAL_HOLE_AZ`. With the pan-axis reference the insertion-pose IK is **exact
(0.00 mm)** over the full ±60° yaw range.

### Geometry (constants in `so101_constants.py`, specs in `specs.py`)

| element | value | why |
|---|---|---|
| board yaw | `±60°` uniform (`HINSERT_BOARD_AZ`) | the primary randomisation; rotates the cone axis per episode |
| face radius | `0.320–0.350` m from the **pan axis** | set BY the measured 100%-reachable horizontal-jaw band `r ∈ [0.220, 0.325]`: a closer board makes the 5 cm approach unreachable |
| bore | `r_clear` 12 mm half-width, `d_recess` 30 mm default | cone half-angle 21.8° at the default |
| peg | 10 mm-radius cylinder, shaft over `z ∈ [0, 0.030]`, holder1 collar over `[0.030, 0.060]` | `GRIP_Z_OFFSET` / `GRIP_ANGLE_*` / the grasp-weld relpose transfer verbatim |
| camera base | `(−0.151, −0.235)`, yaw `+1.571` | swept; see below |
| lens | Menagerie so101 intrinsics (48.46°, 4:3) | new mode → current camera era, like `transfer` / `lidbox` |
| success | axis within 5°, tip past 10 mm, **held 10 steps** | first mode with a non-instantaneous success criterion |

The peg is **cylindrical** on purpose: the radial IK leaves peg roll to the
kinematics and the policy has no roll authority, so a keyed cross-section would
be unsolvable. The bore is **square** so the panel decomposes exactly into six
boxes with no convex-hull approximation.

### Why the camera base could not be inherited

Unlike the benchmark tasks this mode cannot reuse `SIDE_CAM_BASE_POS`. The bore
axis points *away* from the manipulator, so the viewing cone runs back **through
the manipulator's own workspace**: an in-cone camera sits 10–25 cm in front of
the face, i.e. barely outside the shoulder, and must be laterally offset to
clear the arm — which lies in the vertical pan plane *containing* the cone axis.
A base directly behind the arm scores 100% on camera reach and then fails on
collision (−63 mm penetration on 42 of 60 sampled pairs); a base to the **side**
approaches from off the pan plane and does not. Swept over 7 radii × 32 azimuths
× 4 recess depths in `scripts/sweep_hinsert_camera_base.py`.

### Gates

Run, not trained. See `results/new_tasks/hinsert/`.

```bash
MUJOCO_GL=egl uv run python scripts/gate_hinsert_reach.py            # gate 1
MUJOCO_GL=egl uv run python scripts/gate_hinsert_self_occlusion.py   # gate 2
MUJOCO_GL=egl uv run python scripts/gate_hinsert_exclusivity.py      # gates 3+4
MUJOCO_GL=egl uv run python scripts/render_hinsert_matrix.py         # gate 5
MUJOCO_GL=egl uv run python scripts/probe_hinsert_insertion.py       # solvability
MUJOCO_GL=egl uv run python scripts/benchmark_hinsert_throughput.py
```

`scripts/calibrate_workspace.py` does **not** cover this mode, for the same
reason it does not cover `lateral`: it gates jaw-down grasp modes and its
z-clamp assumes the peg hangs below the TCP. `gate_hinsert_reach.py` is the
equivalent for the radial formulation.

## `lidlatch` — latch-released spring lid, hinged toward the camera

The `lidbox` scene with the lid opened by a **mechanism** instead of the arm.
Motivation: the graded box-yaw gates
(`results/new_tasks/lidbox/graded_yaw_gates/GATE_NOTES.md`) showed that a lid
hinged toward the camera arm (box yaw 180°) cannot be opened by the SO-101
gripper — the fixed jaw leads and jams, the plate props on the wrist at
85–96°, and no retreat path disengages it — while that hinge side is exactly
the one that roofs the interior from the camera home. Releasing a spring
removes the arm from the plate's swing, so every yaw bin becomes openable and
the measured camera-regime gradient can be used.

### Geometry (constants `LIDLATCH_*` in `so101_constants.py`, spec `make_lidbox_spec(latch=True)`)

| element | value | why |
|---|---|---|
| latch | joint equality `lid_latch` pinning the hinge at 0, active in the spec (mjwarp's per-env reset restores it), nearly rigid impedance (`LIDLATCH_EQ_SOLIMP`) | released per env through `eq_active`, the grasp-weld mechanism; with the default impedance the spring pre-load would hold the "closed" lid several degrees ajar (measured 0.005° with the rigid setting) |
| spring | torsion spring `LIDLATCH_HINGE_STIFFNESS` 0.04 N m/rad, springref **at the stop**; damping 0.008 | pre-load 0.091 N m vs 0.021 N m gravity at closed; the released lid passes 90° in 0.29 s and sits on the stop at 0.53 s with 0.6° overshoot, and re-opens after being knocked down to 20°/60° (CPU tests) |
| stop | 110° (`LIDLATCH_LID_STOP`, = the lidbox stop) | the only stop at which the **static camera stays blind for every cube spawn**: worst cube px at 128×96 over 5 centers × 7 cube offsets — 110°: 0, 120°: 22 (cubes on the hinge/camera side peek over the plate), 130°: 67 (whole cube ≈ 50 px). The {0,180} gate's "130° = active-only" row was computed on the lb3 centers 3–6 cm farther from the camera; here the home looks down more steeply and sees over the plate (`videos_stop130/` shows it) |
| button | cylinder r 8 mm × 10 mm on a 4 cm ledge outside the **free-edge** wall, center 3 cm beyond the wall face; release when the TCP is within 1.5 cm of its top center (`LIDLATCH_PRESS_RADIUS`) | contact-free trigger, the magnetic-capture convention; the visual button is collidable; box-lid contacts stay excluded so the plate sweeps over the ledge freely |
| swing clearance | `LIDLATCH_JAW_CLEAR` 4 cm: a TCP that far beyond the lid edge is outside the plate's whole swing | the fixed jaw juts 4.8 cm toward the lid (+y) above z +0.04 and the wrist 5 cm at z +0.10..0.17; a press straight over a near button leaves the jut over the plate and the released lid stalls on it at ~7° (measured). The jaw-clear press pose (4 cm out, tips on the button plane) is 1.2 cm from the button top, inside the trigger sphere |
| lid | 14 × 14 cm plate, **no lip, no knob** | both were affordances for the manual opener |
| box yaw | 180° (`LIDLATCH_YAW_SET`) | hinge on the +y (camera) edge: the open lid roofs the interior from the home; the graded set is the target once the button is reachable at every bin |
| spawn | cx [−0.02, 0.01] × cy [−0.05, −0.01] (`box_center` / `box_jitter` on the mode) | the lb3 x-range; the y-range that keeps the jaw-clear line (cy − 0.11) inside the reachable action box — 3 cm nearer the camera than lb3, so the yaw-180 camera rows must be **re-gated** on it |
| TCP spawn | (−0.02, −0.165, 0.08) ± 5 mm (`LIDLATCH_SPAWN_TCP`, `tcp_spawn` on the mode) | on the free-edge side, beyond the jaw-clear line of every center: the lidbox hover sits OVER the box and a springing lid lands on it |
| action box | (−0.05, −0.17, 0.015) .. (0.04, 0.107, 0.085) | 6×6×4 jaw-down IK grid < 2 mm and button hovers ≤ 1 mm (tests); y −0.185 loses the +x corner at z 0.09; no lift-under-lip corridor, so z-max drops to 0.085 |
| reward | the lidbox staged reward + `1[latched] · 0.35 · max(0, log(100 d0+1) − log(100 d_button+1))` | the press term's maximum (0.97) stays below the +1/step lid term the released lid pays, so releasing never scores below hovering at the button |
| metrics | the lidbox set + `latch_released_rate`, `release_step` | |

Everything else — cameras (Menagerie intrinsics), bases, gimbal home/box, the
cube and its magnetic capture, success (captured + lifted 1 cm above the rim),
the 4 s budget — is the lidbox mode's.

### Known limits (to gate before training)

* The button sits on the free edge, which at yaws 45/90 lies inside the
  manipulator's reach hole (`GATE_NOTES.md` 0.4): the graded yaw set is not
  reachable yet; a second button or a side-wall placement is the follow-on.
* A policy that presses straight over the button and lingers gets a lid
  stalled on its jaw until it backs off 1 cm; the jaw-clear pose is inside
  the trigger sphere, so the clean press exists, but it is the policy's to
  find. `scripts/probe_lidlatch.py` measures `over` / `retreat` / `clear`
  on the GPU physics.
* The spawn rectangle moved 3 cm toward the camera relative to lb3, which
  is why the gate's 130° "active-only" reading did not transfer (measured
  above; the stop is 110° for that reason). Still open: whether a LOW gimbal
  pose sees into the box at 110° on this rectangle — on the lb3 centers the
  earlier gate found no pose that does (a wrist-only roof, i.e. no
  static-vs-active separation from the lid alone). Run the best-pose sweep
  of the yaw gates on this rectangle before relying on an active-only story;
  if it stays a roof, the levers are a lower gimbal box or the wall/lid
  geometry, not the stop.
* **GPU only.** mujoco-warp's CPU device blows the whole world up (NaN in
  every dof within ~10 steps) whenever the lid hinge carries a spring — a
  static latched scene, any integrator, any lid inertia; the spring-free
  `lidbox` is fine and the GPU runs the same scenes clean (bisected
  2026-09-03). Keep the latch mode's env tests, probes and training on
  CUDA; the env-level test is `skipif`-gated accordingly.

```bash
uv run pytest tests/test_empty_hand_modes.py -k lidlatch            # spec, physics, config, GPU env
MUJOCO_GL=egl uv run python scripts/probe_lidlatch.py               # batched GPU probe (+ --video)
MUJOCO_GL=egl uv run python scripts/render_lidlatch_videos.py       # scene | static camera | wrist videos + sheets
uv run python -m active_perception_arms.rl.train_insertion --insertion-mode lidlatch ...
```
