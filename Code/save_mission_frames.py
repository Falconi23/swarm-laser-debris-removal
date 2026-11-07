"""
Render one frame per mission phase from a deterministic rollout of the trained policy.

Needs swarm_checkpoints/swarm_shared_latest.zip from swarm.py. Run from Code/:
  python save_mission_frames.py
Writes mission_frame_{1..4}_*.png next to this script.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from swarm import ADRSwarmEnv, StarTrackerModel

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "swarm_checkpoints", "swarm_shared_latest")

from stable_baselines3 import PPO

print("Loading model from:", CKPT + ".zip")
model = PPO.load(CKPT, device="cpu")

print("Initialising StarTracker (this may take a moment)...")
st = StarTrackerModel()

env = ADRSwarmEnv(star_tracker=st, render_mode="rgb_array")
obs, _ = env.reset(seed=42)

frames = {}       # key -> (image, step, phase, debris_hp)
prev_phase = 1.0
laser_captured   = False
destroy_captured = False

img = env.render()
frames['initial'] = (img, 0, env.phase, env.debris_hp)
print(f"[step   0] INITIAL  — phase={env.phase:.1f}  HP={env.debris_hp:.0f}")

for step_i in range(env.MAX_STEPS):
    actions = {}
    for a in env.possible_agents:
        act, _ = model.predict(obs[a], deterministic=True)
        actions[a] = act

    obs, rews, terms, truncs, infos = env.step(actions)
    cur_phase = env.phase

    # first step in which a drone is actually firing, not just in phase 2.0
    if cur_phase == 2.0 and not laser_captured and not env.debris_destroyed:
        if any(env._firing):
            img = env.render()
            frames['laser'] = (img, step_i+1, env.phase, env.debris_hp)
            laser_captured = True
            print(f"[step {step_i+1:3d}] LASER    — phase={env.phase:.1f}  HP={env.debris_hp:.0f}")

    # first step after the debris is destroyed
    if cur_phase == 2.5 and not destroy_captured:
        img = env.render()
        frames['destroyed'] = (img, step_i+1, env.phase, env.debris_hp)
        destroy_captured = True
        print(f"[step {step_i+1:3d}] DESTROY  — phase={env.phase:.1f}  HP={env.debris_hp:.0f}")

    prev_phase = cur_phase

    if all(terms.values()) or all(truncs.values()):
        img = env.render()
        status = "MISSION COMPLETE" if env.mission_done else "FAILED/TIMEOUT"
        frames['returned'] = (img, step_i+1, env.phase, env.debris_hp)
        print(f"[step {step_i+1:3d}] {status} — phase={env.phase:.1f}  HP={env.debris_hp:.0f}")
        break

env.close()

FRAME_NAMES = {
    'initial':   'mission_frame_1_initial.png',
    'laser':     'mission_frame_2_laser.png',
    'destroyed': 'mission_frame_3_destroyed.png',
    'returned':  'mission_frame_4_returned.png',
}

SUBTITLES = {
    'initial':   'Phase 1.0 — Approach: swarm departs base, forming equilateral triangle',
    'laser':     'Phase 2.0 — Laser Engagement: all three drones firing on debris',
    'destroyed': 'Phase 2.5 — Return: debris destroyed, swarm heading back to base',
    'returned':  'Phase 3.0 — Mission Complete: swarm docked at base',
}

out_dir = os.path.dirname(os.path.abspath(__file__))

for key, fname in FRAME_NAMES.items():
    if key not in frames:
        print(f"[WARN] Frame '{key}' not captured — skipping.")
        continue

    img_arr, step, phase, hp = frames[key]
    out_path = os.path.join(out_dir, fname)

    fig, ax = plt.subplots(figsize=(13, 9.5), facecolor='white')
    fig.patch.set_facecolor('white')
    ax.imshow(img_arr)
    ax.axis('off')
    fig.suptitle(SUBTITLES[key],
                 fontsize=12, fontweight='bold', color='#1a1a1a', y=0.02)
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=130, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"[OK] Saved → {fname}")

print("\n[DONE] All mission frames saved.")
