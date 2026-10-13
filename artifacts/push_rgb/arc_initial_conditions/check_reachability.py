"""Static IK boundary screen for the arc; separate from rollout feasibility."""
import json
from pathlib import Path
import numpy as np
import torch
from active_perception_arms.config import Experiment, SOURCE_XY
from active_perception_arms.native import NativeEnv
from active_perception_arms.manipulation import CartesianIK

torch.set_num_threads(1)
e = NativeEnv(Experiment(task='push', task_revision='push_v2', manip_control='cartesian', render_sensors=False, reset_xy_jitter=.02))
ik = CartesianIK(e.model, 'push', dtype=torch.float64)
points, descriptors = [], []
for dx in [-.02, 0, .02]:
    for dy in [-.02, 0, .02]:
        obj = np.array(SOURCE_XY) + [dx, dy]
        for angle in range(-80, 81, 20):
            for radius in [.05, .085, .12]:
                direction = np.array([np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))])
                goal = obj + radius * direction
                for stage, xy, z in [('approach', obj-.028*direction, .04), ('contact_end', goal-.017*direction, .04), ('withdraw', goal-.017*direction, .075)]:
                    points.append([*xy, z])
                    descriptors.append(dict(angle=angle, radius=radius, source_xy=obj.tolist(), stage=stage))
p = torch.tensor(points)
q = torch.tensor(e.home['manipulator'][:5]).repeat(len(p), 1)
for _ in range(30):
    q = ik.solve(p, q)
actual, _ = ik.chain.fk(q)
actual += ik.base
err = (actual-p).norm(dim=-1)
bounded = ((p >= ik.low) & (p <= ik.high)).all(1)
limits = torch.tensor(e.limits['manipulator'][:5])
within = ((q >= limits[:, 0]) & (q <= limits[:, 1])).all(1)
report = dict(points=len(points), max_position_error_mm=float(err.max()*1000), within_action_bounds=int(bounded.sum()), within_joint_limits=int(within.sum()), position_within_2mm=int((err < .002).sum()), scope='Static IK endpoints only; no path collision or physical rollout certification', failures=[dict(**d, position_error_mm=float(er*1000), within_action_bounds=bool(b), within_joint_limits=bool(l)) for d, er, b, l in zip(descriptors, err, bounded, within) if er > .002 or not b or not l])
Path(__file__).with_name('reachability.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
