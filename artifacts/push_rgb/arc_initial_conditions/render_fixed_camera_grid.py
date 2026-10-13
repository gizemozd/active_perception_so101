"""Show what the actual fixed policy camera sees in nine revised resets."""
import json
import os
from pathlib import Path
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from active_perception_arms.config import Experiment
from active_perception_arms.native import NativeEnv
from active_perception_arms.sanity import CameraAudit

out=Path(__file__).resolve().parent
cfg=Experiment(task='push',condition='wrist_static',task_revision='push_v2',reward_profile='push_dense',manip_control='cartesian',occlusion='clean',reset_xy_jitter=.02,push_visibility_revision='crossbar_v3',push_target_layout='arc_v2',num_envs=1)
env=NativeEnv(cfg); audit=CameraAudit(env)
scale=4; width=cfg.width*scale; height=cfg.height*scale; caption=56;gap=16;header=86
canvas=Image.new('RGB',(3*width+4*gap,header+3*(height+caption+gap)+gap),'#101923')
d=ImageDraw.Draw(canvas); title=ImageFont.load_default(size=27); font=ImageFont.load_default(size=18)
d.text((gap,12),'Fixed camera: nine target positions',fill='white',font=title)
d.text((gap,51),'Actual 96 x 72 policy images, enlarged 4x | Crossbar present',fill='#bfd0de',font=font)
try:
 for i,seed in enumerate(range(40000,40009)):
  env.reset(seed);rgb,metrics=audit.render('fixed')
  x=gap+(i%3)*(width+gap);y=header+(i//3)*(height+caption+gap)
  canvas.paste(Image.fromarray(rgb).resize((width,height),Image.Resampling.NEAREST),(x,y))
  pixels=metrics['fixture_pixels']; verdict='VISIBLE' if pixels>=3 else 'HIDDEN'
  d.text((x+5,y+height+6),f'{i+1:02d} | seed {seed} | {verdict}',fill='#6ef59a' if pixels>=3 else '#ffb1a2',font=font)
  d.text((x+5,y+height+31),f'Green marker: {pixels} pixels',fill='#bfd0de',font=font)
finally:audit.close()
canvas.save(out/'fixed-camera-nine-resets.png')
print(out/'fixed-camera-nine-resets.png')
