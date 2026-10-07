# Literature review and project positioning — 2026-10-06

The project remains worth investigating, but its contribution needs to be narrower
than a demonstration that a second arm can improve vision. The strongest candidate
is a controlled study of **when fresh, physically acquired observations improve
contact-rich manipulation enough to justify their cost**, after accounting for
camera placement, resolution, initial inspection, memory, and simpler recovery
strategies. This is a proposed research direction, not an established novelty or
a positive experimental result.

## Sources and scope

The four supplied PDFs were read through their methods, experiments, limitations,
and appendices. Page numbers below are PDF page numbers. Important tables and the
EyeRobot 2.0 rendering diagram were also checked visually.

| Local version | Paper | Main evidence locations |
|---|---|---|
| `2506.10968v2.pdf`, 18 pages | [Eye, Robot: Learning to Look to Act with a BC-RL Perception-Action Loop](https://arxiv.org/abs/2506.10968v2) | pp. 4–9; appendices pp. 15–18 |
| `2602.01939v3.pdf`, 8 pages | [Towards Exploratory and Focused Manipulation with Bimanual Active Perception](https://arxiv.org/abs/2602.01939v3) | pp. 4–7, Tables III–V |
| `2609.37292v1.pdf`, 20 pages | [Recovering the View: Benchmarking Physical Active Vision for Occlusion Recovery in Robotic Manipulation](https://arxiv.org/abs/2609.37292v1) | pp. 3–9; appendices pp. 13–20 |
| `2610.03710v1.pdf`, 25 pages | [EyeRobot 2.0: Active Gaze for Precise Manipulation without Wrist Cameras](https://arxiv.org/abs/2610.03710v1) | pp. 3–9; appendices pp. 10–19, especially Table 5 |

The local PDFs are the source of the detailed four-paper audit. Additional primary
sources were checked online to avoid mistaking an omission in these four papers
for a field-wide gap. That supplementary check was targeted, not an exhaustive
review. The user-supplied PDFs remain untracked; they were not added to Git.

## What the four papers establish, and what they leave open

### EyeRobot: learned gaze already works, with informative negative results

EyeRobot learns a physical pan/tilt camera policy with reinforcement learning and
a manipulation policy with behavior cloning. Its EyeGym replays 360-degree video
of demonstrations while varying gaze; the camera is rewarded through manipulation
action-prediction accuracy. It studies five physical tasks, with wrist/external
comparisons on eraser placement and E-stop reaching. Therefore, learning camera
motion with RL, gaze changing with task phase, and joint perception–action
training are already demonstrated.

The eraser results are particularly relevant. Table 1, p. 7, reports **100% for
wrist sensing versus 60% for EyeRobot** on ordinary pick-and-place; after the
placement target is perturbed, the reported ordering reverses to **10% wrist,
40% wrist+external, and 100% EyeRobot**. These are their task-specific results,
not estimates for our system. They already provide evidence for the conditional
hypothesis that active sensing becomes valuable when relevant information changes.

The paper explicitly identifies lack of translational motion parallax as its
primary limitation (p. 9). Rotating its eye cannot expose a surface hidden behind
an obstruction along all available rays from that camera center. Its servoing
errors are on the order of centimeters, so it does not establish millimeter-scale
contact alignment under such occlusion.

It does not report a searched external-camera placement study or an
initial-inspection-plus-memory comparison. This is a scope distinction, not an
argument that its baselines are careless: it matches architecture and data and
tests camera configurations. Appendix D, p. 16, gives the eye a 10-step history
and makes the manipulation policy effectively single-frame. It would be wrong to
say the entire system has no memory. The active-visual-pretraining ablation also
studies training effects; training benefits are not untouched territory.

**Implication:** retain wrist sensing as a strong baseline and retain negative
controls. Physical translation around hidden contact geometry is more relevant to
our apparatus than repeating panoramic object search.

### EFM/BAP: the spare-camera-arm idea is directly covered

This paper is the closest hardware overlap. One arm supplies an informative wrist
view while the other manipulates; force/torque sensing supports contact. The ten
physical tasks already include USB light insertion, charger insertion, cable
matching, placement, and pushing. Demonstrated camera motion is learned through
imitation. Table IV evaluates ACT, Diffusion Policy, GR-MG, and **π0**.

Its preliminary comparison, Table III on p. 5, varies whether the active view
contains the manipulated area and the operating end effector. For example,
Cup-Hang rises from 23.3% when neither is captured to 90% when both are captured.
This establishes the value of useful visual context, but does not compare against
an external camera selected by a disclosed placement search. Nor does it separate
one initial inspection from continued movement and memory using retrained controls.

Fine manipulation remains difficult even with BAP: Table IV reports π0 at 23.3%
on Light-Plug and 16.7% on Charger-Plug. Table V raises GR-MG Light-Plug success
from 20.0% to 36.7% by adding force/torque. These results support studying how
visual and contact information complement one another; they do not establish that
vision alone is the bottleneck or that RL would solve it.

**Implication:** neither SO-101 deployment, a spare arm for vision, plug insertion,
nor attaching a VLA is sufficient novelty. Our contact tasks need to distinguish
missing visual evidence from force-control or mechanical failure. Probing is a
legitimate alternative, and actual available force/proprioceptive channels must be
reported.

### BAVO-Bench/A-FAR: random occlusion and causal visibility are already covered

BAVO-Bench trains on stage-aligned external occlusion and evaluates clean,
stage-aligned, and randomly timed occlusion across five simulated tasks. Its
main policies all operate in an active-camera setting. A-FAR uses robot-base
point clouds and training-only 4D relational distillation. Physical deployment
is demonstrated on an observation arm and a manipulation arm.

Its Camera-attributable Visibility Gain (CVG, p. 5) is a substantive causal
control: it compares the realized camera pose with the event-start camera pose
while rendering **the same current scene state**. We should reuse or cite this
idea, not claim that attributing visibility recovery to camera motion is new.

Tables 1–2 on p. 8 report A-FAR at 63.7% success for Stage Occlusion and 18.7%
for Random-time Occlusion. π0.5 achieves the highest average CVG, 34.42 percentage
points, yet only 4.0% random-time task success. The paper itself discusses the gap
between acquiring a useful view and exploiting it for manipulation (p. 9).
Consequently, neither random-time recovery, VLA evaluation, nor the observation
that visibility does not guarantee success is available as our central novelty.

The remaining comparisons are different: best searched fixed+wrist, initial
inspection with memory, scheduled movement, and observation-dependent movement
at matched physical cost. CVG compares a camera with its previous position; it
does not establish task-success gains over policies trained for the best found
static configuration.

Several qualifications matter when using its numbers. π0.5 has one multitask
training run; the other main policies have three per-task seeds. A-FAR uses
RGB-D geometry, so the comparison does not isolate model scale alone. Random-time
can contain more occlusion events than Stage; the authors explicitly acknowledge
that this is a stronger condition rather than a timing-only intervention.
Occluders are placed to block the current camera-to-target ray and then remain
fixed until the next event (Appendix A). The physical section gives feasibility
rollouts, not a quantitative hardware comparison against optimized fixed sensing.

**Implication:** simply adding moving screens and reporting visibility recovery
would overlap heavily. A sharper study manipulates whether the hidden state
actually changed, independently of whether its image became occluded.

### EyeRobot 2.0: precision does not imply a need to translate a camera

The critical implementation detail is that this system uses a **fixed stereo
camera**. Section 3.1, p. 4, synthesizes fixation by warping the two images around
their optical centers; Section 4, p. 6, describes the static hardware mounting.
It allocates detail through foveated crops and represents actions in a
fixation-relative frame. The paper explicitly leaves head/neck movement outside
its scope (p. 9).

This is strong evidence that better use of an existing view can improve precise
manipulation without a translating camera arm. Its seven physical tasks include
straw insertion and marker capping; its aggregate physical performance is 66.5%
in Table 2. Foveation and stereo ablations reduce that result. The simulation
ablation in Table 5, p. 18, drops from 74% to 53% without fixation-relative
actions, showing that action representation contributes substantially.

It also contains a relevant control that must not be overlooked: restricting
fixation to the best single object produces 69% simulation success versus 74%
for full target selection. Constant-object gaze can still track a moving object;
it is not a frozen camera pose or an initial-information-only policy. Nonetheless,
it already asks a meaningful version of whether ongoing target switching helps.

The manipulation policy is single-frame; the gaze policy has short gaze-direction
history (p. 11). There is no initial-inspection-plus-longer-memory study. The
reported setting trains task-specific policies rather than a general VLA. Main
simulation results use the best of three seeds, with all seed statistics supplied
in the appendix; physical evaluation randomly selects one of three trained seeds.
Its substantial physical validation should be acknowledged alongside those limits.

**Implication:** add an informative fixed-camera processing baseline. A cheap
low-resolution full-frame baseline is insufficient for a strong physical-motion
claim. Adaptive cropping can recover detail in observed rays; it cannot measure a
surface hidden from both original stereo viewpoints.

## Additional overlap beyond the four supplied papers

[ActiveArena](https://arxiv.org/html/2609.24124v1) already studies 35 tasks,
memory writing/capacity, planning, and active perception with VLA variants. It
also has a fixed-viewpoint variant and physical experiments. Its fixed variant
removes active-view action dimensions, rather than reporting an external-camera
placement optimization. Its broad evidence-acquisition/memory framing means we
cannot claim that combining memory and active perception is new.

[TAVIS](https://arxiv.org/html/2605.07943v1) already offers paired active-head versus
fixed-camera comparisons on the same demonstrations, task-conditional gains,
distribution shifts, and a teleoperation-data-bias experiment. Thus neither a
matched comparison nor separating some training-data effects from observation
effects is novel by itself.

[SaPaVe](https://arxiv.org/abs/2603.12193) already learns semantic camera control
and manipulation with a VLA and a decoupled camera action component.
[ActiveScale](https://active-scale.github.io/) combines a VLA, observation history,
camera-pose supervision, substantial human/robot training data, and an independently
actuated camera. A camera adapter or adding history to a VLA is not an unoccupied
direction.

The underlying principles are older still: [Cheng et al.](https://proceedings.mlr.press/v87/cheng18a.html)
study RL for manipulation with active vision under occlusion, and
[Learning to Look](https://arxiv.org/abs/2410.18964) studies information-seeking via
policy factorization. We should present value-of-information reasoning as a
foundation, not invent a novelty claim for the principle.

## A defensible project question

**When does new physical visual evidence change the correct manipulation action,
and when is obtaining it worth more than using memory or simpler sensing?**

The original hypothesis remains sensible. The literature establishes much of its
broad motivation. A contribution would require new controlled findings about a
specific unresolved boundary, or a method that demonstrably improves that boundary.
More baselines alone do not guarantee a publishable result.

For our platform, separate these situations:

| Information state | Example | Main competing explanation |
|---|---|---|
| Present but too small in the image | Tiny visible insertion feature | Higher resolution or adaptive cropping suffices |
| Hidden initially, stable afterwards | Original hidden prong offset | Inspect once and remember; or choose a better fixed view |
| Previously observed, unchanged during occlusion | Stable fixture temporarily blocked | Memory or waiting suffices |
| Previously observed, then changed | Compliant receptacle shifts after contact | A fresh observation or contact probe may be needed |
| Newly determined by an interaction | Latch actually engages or remains partly seated | Inspect the result before the next consequential action |

The informative experiment varies **occlusion and state change separately**. Use
the same apparatus with visible/occluded and unchanged/changed conditions. Changes
should come from a physically credible mechanism or a clearly labeled controlled
disturbance. Do not make a camera move necessary by forbidding ordinary recovery
actions or by giving baselines an unsuitable field of view.

This decomposition distinguishes more pixels, a better initial pose, retained
information, stale information, and action-relevant measurement. It also makes
negative results useful for choosing an actual sensing system.

## Relevance to language and foundation models

The relevant robotic models are vision-language models and vision-language-action
models, not text-only LLMs in isolation. They may improve recognition, instructions,
priors, planning, and recovery. They can also learn active sensing, as the papers
above show. Our project should support and evaluate such policies rather than
frame them as incapable of perception or reasoning.

There is a precise limit worth exploiting experimentally. If two equally plausible
physical states produce the same complete available observation history but
require different next actions, additional computation on that same history
cannot identify which state occurred. A prior can guide a guess; memory can
recover earlier evidence; a new camera view or contact interaction can distinguish
the states. This is a conditional information argument, not an empirical claim
that today's VLAs fail our tasks. The experiment must verify its premises.

The hidden-prong plug provides a concrete version: visually identical bodies have
different offsets underneath, requiring different alignment targets. But contact,
proprioception, repeated attempts, a low fixed view, and prior inspection might
provide the missing evidence. A supposed four-way 25% bound would apply only to a
balanced, indistinguishable, one-shot decision with mutually exclusive correct
actions—not to whole episodes that permit informative interaction. It must not be
used to dismiss the legacy wrist policy's reported higher success.

A useful foundation-model experiment compares the same adapted pretrained policy
under searched static sensing, initial inspection with history, a scheduled camera,
and an adaptive camera. Include fixed-view adaptive crops. Verify that the model
can solve an adequately observed version first; poor SO-101 adaptation is not
evidence against foundation models. Supply camera poses with historical images
where appropriate and match adaptation data and observation budgets. No training
of a new foundation model is required.

One optional method is a small controller that chooses whether and where to inspect
based on the expected improvement in the next manipulation decision, observation
history, and camera-motion cost. Compare it to simple periodic and event-triggered
inspection. Uncertainty-triggered sensing and modular camera controllers are
established ideas; a proposed implementation needs its own demonstrated advance.
An LLM's verbal confidence alone is not a calibrated measure of sensing value.

## Concrete experiment sequence

1. **Restore and validate the original hidden-prong task first.** Preserve the
   four offsets, equalized inertials, sensor details, and intended action space.
   The current centered-pin/green-marker implementation is a different task.
   Establish feasible insertion with a privileged-state reference controller so
   failures in mechanics are not misattributed to sensing.
2. **Run the strongest simple sensing comparisons.** Use wrist with history;
   searched fixed+wrist with history and adequate detail; initial camera
   positioning followed by live images from the held pose; an inspection followed
   by retained history with no further external information; a prescribed scan;
   and full feedback-based camera control. The initial-positioning and
   inspection-memory conditions answer different questions and should not be
   conflated. Use the current recurrent actor as a practical starting point.
3. **Add one controlled extension only if needed to answer a distinct question.**
   A compliant receptacle can shift after contact; a clip/seating mechanism can
   reveal a newly determined success state. Pair each with an unchanged-state
   control. Natural occlusion and changed state must both be measured. These
   extensions are proposals, not completed environments.
4. **Evaluate one pretrained VLA after the environment passes these checks.**
   Repeat the core informative contrasts, rather than an enormous model leaderboard.
   This tests whether the conclusion survives stronger visual/semantic priors.
5. **Validate on physical tasks where the sensing tradeoff survives.** The original
   plug and one contact-generated state-change task are sensible candidates.
   Select clearance and loads that the hardware can reliably execute when informed.
   Keep the existing open pushing scene as a negative control.

No jobs should be launched as part of this review; the user's no-training
instruction remains in force.

## Evidence needed for a convincing result

Select the fixed view on validation episodes and evaluate on held-out layouts;
describe it as the **best found under a disclosed search budget**, not a proven
global optimum. Include plausible overhead and low side placements. A two-fixed-
camera configuration is a valuable practical cost comparison because dedicating
an arm has an opportunity cost; it is a separate sensing-budget condition.

Report success, time to successful completion, failed-attempt timeouts, camera
travel, contact/collision outcomes, and inspection count. Show the success–time
tradeoff rather than hiding it inside a single hand-chosen motion penalty. Report
training-seed variation as well as uncertainty over paired test episodes.

For a timing-only test, match the number, severity, and duration of occlusion
events. To compare sensor configurations fairly, predefine shared world-space
occluder trajectories; a policy-dependent camera-targeting occluder is a separate
adversarial experiment. Keep waiting, retreat, regrasping where available, and
contact probing as legitimate strategies with their measured costs.

Use fresh-image, held-image, memory-reset, and camera-freeze interventions as
diagnostics. An active-trained policy failing when frozen establishes dependence
of that policy, not necessity for all policies. Retrain restricted conditions and
cross training/execution sensing regimes where feasible. Compare camera motion
with fresh versus withheld imagery to check whether measured gains depend on its
visual information rather than incidental physical contact. These interventions
still need distribution-shift qualifications.

In addition to visibility, measure whether acquired evidence correctly identifies
the hidden variable and improves the next action. Use simulator labels only for
training supervision or evaluation, never as unreported policy input. Provide
paired states in which similar wrist observations require different corrections,
then test whether an acquired view resolves the ambiguity. This connects perception
to manipulation more directly than target pixel count.

## Decision criteria and current evidence

The [legacy audit](TASK_SCREENING.md#correction-the-original-hidden-prong-plug-is-a-different-task--2026-10-05)
records reported historical success of 99.5% active, 80.7% wrist, 60.1% home-fixed,
and 67.1% frozen-active. These results are promising, but raw checkpoints and
evaluation data were absent from that checkout, and they were not reproduced here.
The legacy layout memo also identifies lower fixed views that reveal all four
variants. Best-static training and initial-inspection-with-memory remain unresolved.

Continue with the adaptive-camera claim if new measurements show a meaningful
success/time advantage over those stronger controls, and identify a reproducible
reason for it. If one inspection matches continued active sensing, report that
result and pursue efficient inspection rather than continuous movement. If a
searched fixed camera with adequate image detail matches performance, recommend
that configuration for this setting. If contact probing resolves the ambiguity
more cheaply, characterize that sensory tradeoff. None of those outcomes validates
a universal claim that an active camera is required.

The credible contribution would be an experimentally supported account of when
to place, inspect, remember, or reacquire—and, if supported, an efficient controller
that improves the measured tradeoff on both a compact policy and a pretrained VLA.
This is potentially useful for assembly, connector servicing, and inspection
during manipulation. Its publication strength depends on the evidence, not on
the age or size of the policy model.
