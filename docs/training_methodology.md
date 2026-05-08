# Local Pick Policy Training Methodology

## Overview

This document describes the end-to-end process of training a visual pick-and-place policy for a 7-DOF Franka Panda robotic arm in MuJoCo simulation. The final policy is a pixel-based Soft Actor-Critic (SAC) agent trained with a curriculum of four stages, capable of grasping a bottle from random positions on a table regardless of the arm's starting wrist configuration.

---

## 1. Problem Motivation

The full pick-and-place pipeline consists of three components:
1. **Global perception** — detect the object and goal position using a scene camera
2. **Global motion planning** — RRT-based path planning from home to an approach pose above the object
3. **Local pick** — a fine-grained controller that descends, grasps, and lifts the object from the approach pose

The challenge for the local pick component is that the RRT planner delivers the arm to the approach pose with an **unpredictable wrist joint angle (j7)**, depending on the object position and IK solution chosen. The wrist-mounted camera's view of the object changes drastically with j7 — a 180° difference in j7 effectively flips the image, making a fixed-configuration policy completely fail.

Early attempts at a purely proprioceptive (joint-state) policy were abandoned in favour of a **pixel-based visual policy** that can directly observe the object through the wrist camera and learn to pick from whatever it sees, regardless of arm configuration.

---

## 2. Environment Design

### MuJoCo Scene
- Franka Panda arm (7 joints + 1 gripper actuator) mounted over a table
- A red cylindrical bottle as the pick object (RGB: [230, 38, 38])
- A goal region on the opposite side of the table
- A wrist-mounted RGB camera (`wrist_cam`) pointed downward at the object

### Observation Space
- **Policy input**: 84×84 RGB image from the wrist camera (uint8, HWC), stacked 3 frames for temporal context → effective input shape (9, 84, 84) CHW
- No joint state, no proprioception — purely visual

### Action Space
- 4-dimensional Cartesian delta: `[dx, dy, dz, gripper]`
- Cartesian deltas converted to joint targets via IK (IKFast analytical solver) each step
- `action_scale = 0.02` m per unit action
- `frame_skip = 20` physics steps per policy step

### Reward Function (`LocalPickCartesianDirectLiftEnv`)
Two-phase progress reward:
- **Phase 1 (approach)**: `score = 4.0 * (1 - tanh(5 * distance))` — reward growing as EE approaches the object (max 4.0)
- **Phase 2 (grasp + lift)**: Latches when EE is within `align_close_threshold` of the bottle. Score grows from 4 → 14:
  - +2.0 for gripper grasping at correct height
  - +8.0 × lift fraction (scaled to `min_lift_for_success`)
- **Progress reward**: Only improvements over the episode-best score yield positive reward — eliminates all farming exploits
- **Penalties**: premature gripper close, table collision, large actions
- **Success bonus**: +200 on success (lifts object `min_lift_for_success` metres with contact for `success_hold_steps` consecutive steps)

Stage 4 additions:
- `both_finger_bonus`: reward for simultaneous two-finger contact (forces centred grasp)
- `drop_penalty`: penalty for releasing the object after a meaningful lift

### Success Criteria
- Object lifted ≥ `min_lift_for_success` metres above initial position
- At least one finger (stage 1–3) or both fingers (stage 4) in contact
- Criteria met for `success_hold_steps` consecutive policy steps

---

## 3. Algorithm: Pixel SAC with DrQ-style Augmentation

**Algorithm**: Soft Actor-Critic (SAC) with `CnnPolicy` (NatureCNN backbone)

**Key hyperparameters**:
| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-4 |
| Buffer size | 50,000 |
| Batch size | 256 |
| γ (discount) | 0.99 |
| τ (target update) | 0.005 |
| Train frequency | 1 step |
| Gradient steps | 1 |
| Learning starts | 1,000 |
| Parallel envs | 8 (DummyVecEnv) |

**Observation pipeline** (SB3 wrappers, applied in order):
1. `VecFrameStack(n_stack=3)` — stack 3 frames: (84,84,3) → (84,84,9)
2. `VecTransposeImage` — HWC → CHW before VecNormalize (avoids shape mismatch in replay buffer)
3. `VecNormalize(norm_obs=False, norm_reward=True)` — reward normalisation only; raw uint8 images passed directly to CnnPolicy

**Data Augmentation (DrQ-style)**:
- Random shift: pad image by 4px on each side, take random 84×84 crop
- Applied during training only (`aug_pad=4`), disabled at eval/pipeline time (`aug_pad=0`)
- Provides implicit regularisation and improves generalisation to slightly different viewpoints

---

## 4. Training Curriculum

The policy was trained in four sequential stages, each fine-tuning from the previous stage's checkpoint. This curriculum approach was critical — training from scratch with full randomisation caused entropy collapse and zero learning.

### Stage 1 — Learn to Pick (Fixed Configuration)
**Goal**: Teach the policy to pick the object from a known, fixed arm configuration.

**Config**:
- Fixed object position: (0.5, -0.15, 0.125)
- Fixed approach: height = 0.20 m above object, j7 = 2.8973 rad (wrist at joint limit, fingers aligned with bottle axis)
- No noise of any kind

**Training**: 2,000,000 timesteps from scratch

**Result**: 10/10 success rate at eval. Policy reliably grasps and lifts the bottle but only from the exact approach configuration it was trained on. Completely fails when j7 ≠ 2.8973 (as the pipeline delivers).

**Saved as**: `local_pick_drq_2m.zip`

---

### Stage 2 — Wrist Randomisation
**Goal**: Force the policy to rely on the visual content of the image rather than a memorised arm configuration, by randomising the wrist joint angle at each episode start.

**Key insight**: j7 controls the wrist rotation, which rotates the camera. A policy that generalises across j7 values must identify the bottle visually in the image regardless of which direction the camera is pointing.

**Changes from Stage 1**:
- `random_j7=True`: j7 sampled uniformly across full joint range [−2.8973, 2.8973] each episode; solution chosen randomly from valid IK solutions (previously always chose j7 ≈ 2.8973)
- Small approach position noise: `approach_xy_noise=0.05` m, `approach_z_noise=0.05` m
- Object position still fixed

**Training**: Fine-tuned from `local_pick_drq_2m.zip` for 100,000 timesteps (converged very fast due to strong initialisation)

**Monitoring**: `ep_rew_mean` stayed above 190 throughout; `ep_len_mean` ≈ 18–20 steps. Confirmed policy retained pick ability while adapting to varied wrist angles.

**Saved as**: `local_pick_drq_stage2_100000_steps.zip`

---

### Stage 3 — Full Table Randomisation
**Goal**: Generalise to any object position on the table, not just the fixed training position.

**Changes from Stage 2**:
- `full_table_random=True`: object placed uniformly anywhere in table bounds (x: [0.43, 0.57], y: [−0.20, −0.10]) each episode
- All Stage 2 noise retained

**Training**: Fine-tuned from Stage 2 for 200,000 timesteps

**Monitoring**: Initial `ep_rew_mean ≈ 193` at 78k steps, converged quickly. Policy learns to find and pick from any table position.

**Saved as**: `local_pick_drq_stage3_200000_steps.zip`

**Pipeline test**: 50/50 eval success with full noise. Visual inspection showed successful picks from all table positions with varied wrist angles.

---

### Stage 4 — Stable Grasp Refinement
**Goal**: Improve grasp stability. Earlier stages produced technically successful but loose grasps — the object would wobble during transport or slip after being picked.

**Problem identified**: Success criteria in stages 1–3 were too loose:
- `min_lift_for_success = 0.02` m (2 cm) — trivially easy
- `success_hold_steps = 3` — only 3 consecutive steps required
- Single finger contact sufficient — policy learned to graze rather than firmly grasp

**Changes from Stage 3**:
- `min_lift_for_success = 0.05` m (5 cm) — must lift meaningfully
- `success_hold_steps = 5` — must hold for 5 consecutive steps
- `require_both_fingers = True` — both fingers must contact the object (forces centred grasp)
- `both_finger_bonus = 3.0` — additional reward for simultaneous two-finger contact
- `drop_penalty = 2.0` — penalty for releasing the object after it has been lifted
- `hold_lift_threshold = 0.04` m — minimum lift before hold/drop rewards activate

**Training**: Fine-tuned from Stage 3 for 1,000,000 timesteps

**Monitoring**:
- Initial `ep_len_mean ≈ 36` (longer episodes because stricter criteria), `ep_rew_mean ≈ 178`
- Converged at ~300k steps: `ep_rew_mean ≈ 214`, `ep_len_mean ≈ 21`
- `ep_rew_mean > 200` (above success bonus of 200) confirms near-perfect success rate
- Stabilised at ~700k steps: `ep_rew_mean ≈ 218`, marginal further improvement

**Best checkpoint**: `local_pick_drq_stage4_700000_steps.zip` (selected via end-to-end pipeline evaluation — produced 9/10 success rate)

---

## 5. Pipeline Integration

### Contact Detection Fix
The pipeline controller (`rl_pixel_pick.py`) originally detected finger-object contact by looking up MuJoCo geoms by name (`"left_finger_collision"`, `"left_fingertip_collision"`). These names did not exist in the scene XML, causing the contact sets to be empty and success to **never** be detected even when the object was physically being held.

**Fix**: Contact detection now finds all geoms belonging to the finger bodies by body ID (matching the training environment's approach), making it robust to any geom naming convention in the XML.

### Camera-Based Success Detection
In the pipeline, success detection uses a dual approach:
1. **Primary**: 120×160 wrist-cam RGB+depth render — detects red pixels, unprojects to world frame via pinhole camera model, estimates object Z position. This matches the object detection logic used in the training environment.
2. **Fallback**: Physics ground truth (`data.xpos`) if the object is not visible (occluded by the closed gripper).

### Retry Logic
The pipeline implements two levels of retry:
1. **Pick-level retry** (up to 3×): after a successful RL pick, the arm lifts to clearance height and checks if the object is still held (`object_z > 0.20` m). If dropped, returns to approach pose and retries.
2. **Trial-level retry** (up to 3×): after the full pick-and-place, checks if the object landed within 10 cm of the goal. If not, re-perceives the scene and retries the entire sequence.

### Smooth Transport
During object transport (lift to clearance and path to place), joint motion is slowed to `max_joint_step=0.01` rad/step (4× slower than approach) to prevent the object slipping from inertia during arm acceleration.

---

## 6. Key Findings and Lessons

1. **Pixel-based beats proprioceptive for this task**: A visual policy that sees the object directly is inherently robust to variations in arm configuration that would break a state-based policy.

2. **j7 (wrist rotation) is the critical variable**: The pipeline's RRT planner chooses j7 freely, producing values far from the training assumption (j7 ≈ −0.25 in pipeline vs. 2.8973 in training). This single mismatch caused 0/10 pipeline success with the original model.

3. **Curriculum training is essential**: Attempting to train with full randomisation from scratch resulted in entropy collapse at ~1M steps with near-zero success. Starting from a competent fixed-configuration policy and progressively widening the distribution allowed each stage to converge quickly.

4. **Success criteria tightness directly determines grasp quality**: Loose criteria (2 cm lift, 1 finger, 3 steps) led to policies that grazed the object rather than firmly grasping it. Tightening to 5 cm, both fingers, 5 steps produced demonstrably more stable grasps visible in the simulation viewer.

5. **Contact detection must be robust**: Named geom lookups silently fail if names don't match. Body-ID-based geom finding is more robust and matches how the training environment operates.

6. **Fine-tuning is sample-efficient**: Each curriculum stage converged in 100k–300k steps (vs. 2M for Stage 1 from scratch), demonstrating strong transfer between stages.

---

## 7. Final Pipeline Performance

Evaluated on 10 random trials with randomised object positions and goal positions:
- **9/10 trials successful** (object placed within 10 cm of goal)
- 1 failure due to RRT planning failure on retry (object bounced far from goal after drop, landed in an unreachable region)
- Average pick attempts per trial: ~1.1 (retry rarely needed)
- Pick-and-place cycle time: ~30–60 seconds per trial including planning

---

## 8. File Reference

| File | Description |
|------|-------------|
| `src/rl/local_pick/config.py` | All training hyperparameters and curriculum flags |
| `src/rl/local_pick/local_pick_env.py` | Base env: reset logic, IK approach sampling, contact detection |
| `src/rl/local_pick/local_pick_cartesian_env.py` | Cartesian delta action space, IK step |
| `src/rl/local_pick/local_pick_cartesian_direct_lift_env.py` | Two-phase reward, stable grasp criteria |
| `src/rl/local_pick/local_pick_cartesian_pixel_env.py` | 84×84 pixel observation, DrQ augmentation |
| `src/rl/local_pick/train_cartesian_pixel.py` | Training script with all curriculum flags |
| `src/rl/local_pick/evaluate_cartesian_pixel.py` | Evaluation script |
| `src/pipeline/localPick/rl_pixel_pick.py` | Pipeline controller: runs policy, camera-based success detection, retry logic |
| `src/main.py` | Full pick-and-place pipeline with trial-level retry |
| `models/local_pick_drq_2m.zip` | Stage 1 checkpoint |
| `models/local_pick_drq_stage2_100000_steps.zip` | Stage 2 checkpoint |
| `models/local_pick_drq_stage3_200000_steps.zip` | Stage 3 checkpoint |
| `models/local_pick_drq_stage4_700000_steps.zip` | Stage 4 best checkpoint (used in pipeline) |
