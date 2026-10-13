"""Supervised variant classification from single policy-resolution images (v2 scene).

Trains the policy's CNN architecture (16/32 channels, spatial softmax) to predict
the hidden prong variant from one 128x96 image of a reset state, separately for
the manipulator wrist view, the shared fixed start view, and a gap-aligned view
placed with privileged knowledge of the gap (an upper bound for camera motion).
This measures how learnable the information is per view, not policy success.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import torch
import torch.nn as nn

from active_perception_arms import plug, plug_v2
from active_perception_arms.config import JOINTS, Experiment
from active_perception_arms.native import calibration
from active_perception_arms.robots.camera_control import LOOKAT_BOX, CameraPoseIK
from active_perception_arms.robots.kinematics import axis_to_pan_tilt
from active_perception_arms.robots.plug_control import PlugIK
from active_perception_arms.scenes import grasp_relpose, native_spec

VIEWS = ("wrist", "fixed", "gap_aligned", "gap_aligned_high")


def gap_aligned_pose(plug_pos, gap, height=0.06):
    """Camera on the line from the plug underside through the gap centre."""
    low, high = (np.array(v) for v in LOOKAT_BOX["plug"])
    target = np.array(plug_pos, dtype=float)
    target[2] -= 0.006
    y = 0.21
    scale = (y - target[1]) / (plug_v2.SCREEN_Y - target[1])
    position = target + (np.array([gap, plug_v2.SCREEN_Y, target[2]]) - target) * scale
    position[1], position[2] = y, height
    return np.clip(position, low, high), target


def render_dataset(count, seed, width=128, height=96, views=VIEWS):
    base = Experiment(
        task="plug",
        condition="active",
        control_revision="pose_v1",
        task_revision="hidden_prongs_v2",
        plug_screen="gap_v1",
        camera_action="lookat_v2",
        camera_start="lookat_v2",
        num_envs=1,
    )
    models = {v: native_spec(replace(base, plug_variant=v)).compile() for v in plug.VARIANTS}
    datas = {v: mujoco.MjData(m) for v, m in models.items()}
    renderers = {v: mujoco.Renderer(m, height, width) for v, m in models.items()}
    model = models["xm"]
    ik = PlugIK(model, dtype=torch.float64)
    camera_ik = CameraPoseIK(model, base, dtype=torch.float64)
    home = calibration("plug", camera_start="lookat_v2")
    qadr = {
        arm: np.array([model.joint(f"{arm}/{j}").qposadr[0] for j in JOINTS])
        for arm in ("manipulator", "camera_arm")
    }
    obj_adr = int(model.joint("object/free").qposadr[0])
    fixture = int(model.body("fixture/fixture").mocapid[0])
    screens = [int(model.body(f"screen_{s}/panel").mocapid[0]) for s in ("left", "right")]
    gripper = model.body("manipulator/gripper").id
    p_rel, q_rel = grasp_relpose()
    option = mujoco.MjvOption()
    option.geomgroup[:] = (1, 1, 1, 0, 0, 0)
    option.sitegroup[:] = 0
    rng = np.random.default_rng(seed)
    images = {view: np.zeros((count, 3, height, width), np.uint8) for view in views}
    labels = np.zeros(count, np.int64)
    for i in range(count):
        variant = plug.VARIANTS[i % 4]
        labels[i] = i % 4
        m, d = models[variant], datas[variant]
        xy = np.array(plug.SPAWN_CENTER) + rng.uniform(-1, 1, 2) * np.array(plug.SPAWN_HALF)
        tcp = np.r_[xy, plug.SPAWN_Z] + np.array(plug.GRIP_OFFSET)
        q = ik.reset_manip(torch.as_tensor(tcp)[None])[0].numpy()
        socket = 0.025 + rng.uniform(-0.025, 0.025, 2)
        gap = float(plug_v2.sample_gap(torch.tensor([rng.uniform()]), torch.tensor([xy]))[0])
        mujoco.mj_resetData(m, d)
        d.qpos[qadr["manipulator"]] = np.r_[q, 0.27]
        d.qpos[qadr["camera_arm"]] = home["camera_arm"]
        d.mocap_pos[fixture] = (*socket, 0)
        for mocap, center in zip(
            screens, plug_v2.screen_segment_centers(torch.tensor([gap])), strict=True
        ):
            d.mocap_pos[mocap] = center[0].numpy()
        mujoco.mj_kinematics(m, d)
        rot = d.xmat[gripper].reshape(3, 3)
        quat = np.empty(4)
        mujoco.mju_mulQuat(quat, d.xquat[gripper], q_rel)
        d.qpos[obj_adr : obj_adr + 7] = np.r_[d.xpos[gripper] + rot @ p_rel, quat]
        mujoco.mj_forward(m, d)
        for view in views:
            camera = "manipulator/wrist_cam" if view == "wrist" else "camera_arm/wrist_cam"
            # Every view starts from the parked (shared fixed) camera-arm pose.
            d.qpos[qadr["camera_arm"]] = home["camera_arm"]
            mujoco.mj_forward(m, d)
            if view.startswith("gap_aligned"):
                height = 0.12 if view == "gap_aligned_high" else 0.06
                position, target = gap_aligned_pose(d.qpos[obj_adr : obj_adr + 3], gap, height)
                pan, tilt = axis_to_pan_tilt(torch.as_tensor(target - position))
                pose = camera_ik.tensor(np.r_[position, pan.item(), tilt.item()])[None]
                qc = camera_ik.tensor(home["camera_arm"][:5])[None]
                for _ in range(40):
                    qc = camera_ik.solve(pose, qc)
                d.qpos[qadr["camera_arm"][:5]] = qc[0].numpy()
                mujoco.mj_forward(m, d)
            r = renderers[variant]
            r.update_scene(d, camera=camera, scene_option=option)
            r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            images[view][i] = r.render().transpose(2, 0, 1)
    for r in renderers.values():
        r.close()
    return images, labels


class Classifier(nn.Module):
    """Policy encoder shape: two convolutions, spatial softmax keypoints, MLP head."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ELU(),
        )
        self.head = nn.Sequential(nn.Linear(64, 128), nn.ELU(), nn.Linear(128, 4))

    def forward(self, x):
        features = self.conv(x.float() / 255.0)
        b, c, h, w = features.shape
        weights = torch.softmax(features.reshape(b, c, h * w), -1).reshape(b, c, h, w)
        ys = torch.linspace(-1, 1, h, device=x.device)
        xs = torch.linspace(-1, 1, w, device=x.device)
        keypoints = torch.stack(
            ((weights.sum(2) * xs).sum(-1), (weights.sum(3) * ys).sum(-1)), dim=-1
        )
        return self.head(keypoints.reshape(b, -1))


def train(images, labels, train_count, device, epochs, seed):
    torch.manual_seed(seed)
    x = torch.as_tensor(images)
    y = torch.as_tensor(labels)
    model = Classifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        order = torch.randperm(train_count)
        for start in range(0, train_count, 64):
            index = order[start : start + 64]
            loss = nn.functional.cross_entropy(model(x[index].to(device)), y[index].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        logits = torch.cat(
            [model(x[i : i + 256].to(device)).cpu() for i in range(train_count, len(x), 256)]
        )
    predicted = logits.argmax(-1)
    confusion = torch.zeros(4, 4, dtype=torch.long)
    for truth, guess in zip(y[train_count:].tolist(), predicted.tolist(), strict=True):
        confusion[truth, guess] += 1
    return float((predicted == y[train_count:]).float().mean()), confusion.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, default=1600)
    parser.add_argument("--test", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--views", nargs="+", default=list(VIEWS), choices=VIEWS)
    parser.add_argument("--label", default="")
    parser.add_argument("--output", type=Path, default=Path("artifacts/diagnosis/perception_probe"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    images, labels = render_dataset(
        args.train + args.test, seed=4242, width=args.width, height=args.height, views=args.views
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    result = {
        "train": args.train,
        "test": args.test,
        "epochs": args.epochs,
        "resolution": [args.width, args.height],
        "chance": 0.25,
        "class_order": list(plug.VARIANTS),
        "test_accuracy": {},
        "confusion": {},
    }
    for view in args.views:
        accuracy, confusion = train(images[view], labels, args.train, device, args.epochs, seed=0)
        result["test_accuracy"][view] = accuracy
        result["confusion"][view] = confusion
    name = f"result{('_' + args.label) if args.label else ''}.json"
    (args.output / name).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
