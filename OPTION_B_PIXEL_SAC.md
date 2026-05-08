# Option B — Pixel SAC (DrQ-v2 style)

Same task as Option A, but the **policy sees raw wrist-cam pixels** instead of
engineered features.  Reward is identical (Option A progress reward + reached_box_flag).

---

## Core Idea

Replace the 19-D state vector with an 84×84×3 wrist-cam image.
Stack 3 consecutive frames so the policy has temporal information (velocity
is implicit from the frame difference).  Use SB3's built-in CnnPolicy
(NatureCNN encoder → MLP actor/critic), which is structurally identical to
DrQ-v2 without the explicit random-shift augmentation.

```
obs = wrist_cam_rgb @ 84×84  →  VecFrameStack(n_stack=3)  →  (84,84,9)
         ↓
     NatureCNN encoder  →  latent 512-D
         ↓
     SAC actor / twin critics
```

## Why It Should Work

- Wrist cam already centers the bottle after the approach pose.
- Policy learns visual alignment directly — no separate detector running at inference.
- NatureCNN + SAC has solved similar pick tasks in under 2M steps in literature.
- Progress reward (Option A) eliminates farming exploits that kill pixel RL runs.

## Key Differences from Option A

| | Option A | Option B |
|---|---|---|
| Obs | 19-D state vector | 84×84 wrist cam (×3 stacked) |
| Policy | MlpPolicy | CnnPolicy (NatureCNN) |
| Buffer | 1 M transitions | 100 K (images are ~64 KB each) |
| LR | 3e-4 | 1e-4 (CNN needs slower update) |
| SDE | Yes | No (SDE + CNN doesn't mix well) |
| n_envs | 24 | 8 (more VRAM per env for forward pass) |
| Timesteps | 1 M | 2 M |

## What Stays the Same

- Reward: progress reward + reached_box_flag (Option A)
- Action space: 4-D Cartesian delta + gripper
- Contact-based grasping detection (no 0.012 m threshold bug)
- Success: contact + 2 cm lift held for 3 steps
- Same MuJoCo scene, same approach pose with j7=2.8973

## Training Command

```
python src\rl\local_pick\train_cartesian_pixel.py ^
  --timesteps 2000000 --n-envs 8 ^
  --model-path models/local_pick_option_b_pixel_2m
```

With randomization (recommended after initial verification):
```
python src\rl\local_pick\train_cartesian_pixel.py ^
  --timesteps 2000000 --n-envs 8 ^
  --model-path models/local_pick_option_b_pixel_2m ^
  --object-noise 0.02 --approach-noise 0.01 --approach-z-noise 0.01 --joint-noise 0.01
```

## Evaluation

```
python src\rl\local_pick\evaluate_cartesian_pixel.py ^
  --model-path models/local_pick_option_b_pixel_2m --episodes 5
```

## pip Install

```
pip install "stable-baselines3[extra]>=2.0"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Risk

Low–Medium.  CnnPolicy is production-quality in SB3.  The main risk is that
2 M steps may not be enough sample budget for pixel learning — if
ep_rew_mean stalls under 2.0 after 500 K steps, increase to 4 M or add
random-shift augmentation.  Option A (state-based) is the faster sanity check.
