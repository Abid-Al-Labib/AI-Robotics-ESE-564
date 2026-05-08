# ESE 564 — Robotics AI: Local Pick RL Project Log

**Course:** ESE 564 — Robotics AI  
**Task:** Perception-based RL policy for robotic grasping of a lying bottle  
**Robot:** 7-DOF Franka Panda arm in MuJoCo  
**Algorithm:** SAC (Soft Actor-Critic) via stable-baselines3

---

## 1. Problem Statement

The full pipeline is a pick-and-place system. Perception detects the bottle on the table, the arm moves to an approach pose above it, and then **RL takes over**: the policy must descend, close the gripper around the bottle, and lift it. Once lifted, control hands back to the pipeline for transport.

**Rubric constraint:** The policy must be perception-based — it cannot observe ground-truth (GT) object positions. It must derive its understanding of the scene from camera data.

**Why RL for this subtask?** The grasp-and-lift motion is contact-rich and hard to script. Small errors in approach position and gripper orientation lead to failures that scripted controllers cannot recover from. RL can learn a robust contact-based strategy through trial and error in simulation.

---

## 2. System Architecture

### 2.1 Simulation Setup
- **Simulator:** MuJoCo 3.x
- **Robot:** Franka Panda (7-DOF arm + parallel gripper)
- **Object:** A bottle lying on its side (non-cylindrical, quaternion `[0.7071, 0.7071, 0, 0]` = 90° rotation about X axis)
- **Table bounds:** X ∈ [0.43, 0.57], Y ∈ [-0.20, -0.10], Z = 0.125 m
- **Bottle position (fixed for training):** `[0.5, -0.15, 0.125]`

### 2.2 Episode Structure
Each RL episode begins at the **approach pose** — the pipeline has already moved the arm to ~20 cm directly above the bottle. The RL policy then has up to 125 steps to descend, grasp, and lift the bottle at least 2 cm off the table.

```
[Perception Pipeline] → approach pose (IK) → [RL Policy] → grasp + lift → [Pipeline resumes]
```

### 2.3 Action Space
**4-dimensional Cartesian delta:**

| Index | Meaning | Range |
|---|---|---|
| 0 | Δx (forward/back) | [-1, 1] → scaled to ±0.02 m |
| 1 | Δy (left/right) | [-1, 1] → scaled to ±0.02 m |
| 2 | Δz (up/down) | [-1, 1] → scaled to ±0.02 m |
| 3 | gripper command | ≤0 = open, >0 = close |

Each step, the Cartesian target is converted to joint targets via IK (ikfast analytical solver). Frame skip = 20 (MuJoCo substeps per RL step).

**Why Cartesian?** Joint-space actions are hard to interpret for a manipulation task — the policy has to learn the coupling between joints and end-effector position. Cartesian actions decouple the task-relevant motion (move toward the bottle) from the low-level kinematics.

### 2.4 Observation Space (State-Based, Option A)
**19-dimensional float32 vector derived from wrist camera perception:**

| Components | Dimensions |
|---|---|
| Joint positions (q1–q7) | 7 |
| Joint velocities (dq1–dq7) | 7 |
| EE position − camera-estimated object position | 3 |
| Camera-estimated object lift above table | 1 |
| Gripper opening (m) | 1 |

The camera estimate of object position comes from a color/depth segmentation pipeline on the 120×160 wrist-cam RGB+depth image. **GT object position is never given to the policy** — this satisfies the perception-based rubric.

### 2.5 Wrist Camera
- Resolution: 120×160 (for object detection pipeline)
- MuJoCo camera: `euler="3.1416 0 0" fovy="80"` mounted on the hand body
- Points downward from the hand; rotates with joint 7

### 2.6 Approach Pose & Gripper Alignment
The IK solver (IKFast) has a free joint (joint 7, wrist rotation) that must be set to align the gripper fingers with the lying bottle's long axis.

**Problem found:** Joint 7 defaulted to -0.7853 rad, which left the gripper rotated ~45° off-axis from the bottle. Changing it required targeted IK search.

**Fix:** Set `target_j7 = 2.8973 rad` (the joint limit) and search densely in `[2.50, 2.8973]`:
```python
j7_range = np.linspace(2.50, 2.8973, 100)
solutions = self.kinematics.ik(approach_pos, target_rot, free_joint_range=j7_range)
q = min(valid, key=lambda sol: (abs(sol[-1] - target_j7), np.linalg.norm(sol - home)))
```
After this fix, the gripper fingers are correctly aligned with the bottle axis at episode start.

---

## 3. What We Tried and Why It Failed

### 3.1 Early Reward Designs (All Failed)
Multiple reward functions were tried before finding the working design:

**Attempt 1 — Direct lift reward:**
```python
lift_reward = 12.0 * lift_fraction  # if both_touching
```
*Problem:* Policy learned to push object sideways into the table edge to farm the contact signal. Never actually lifted.

**Attempt 2 — Gripper-closed gating:**
```python
gripper_closed = self._gripper_opening() <= 0.012  # was used to gate lift reward
lift_reward = 10.0 if gripper_closed and object_lift > threshold
```
*Problem:* **This was physically impossible.** The bottle is ~25 mm in diameter. With the bottle between the fingers, the maximum gripper closure is ~25 mm — the 12 mm threshold can never be met. The policy could never receive lift reward, so it never learned to lift. This bug propagated across multiple environment variants.

**Attempt 3 — Height approach reward (ungated):**
```python
height_approach_reward = 1.0 * (1.0 - np.tanh(5.0 * height_above_grasp))
```
*Problem:* This reward was always active. After the gripper closed, the policy still got rewarded for descending, so it continued pushing down with a closed gripper — shoving the bottle into the ground.

**Attempt 4 — tanh coefficient too sharp (coefficient 12):**
```python
reach_reward = 1.0 - np.tanh(12.0 * gt_distance)
```
*Problem:* At the approach height (~20 cm above the bottle), `tanh(12 × 0.20) ≈ 1.0`, so the gradient was essentially zero. The policy had no signal to descend.

### 3.2 Root Cause Summary
| Bug | Effect |
|---|---|
| `gripper_closed <= 0.012 m` threshold | Lift reward physically unreachable; policy never learns to lift |
| Ungated height_approach_reward | Policy pushes bottle into ground after closing gripper |
| Reach reward tanh coefficient too sharp | Zero gradient at approach height; policy stays put |
| Contact-only reward without lift | Policy farms contact without lifting (local optimum) |

---

## 4. Current Design — Option A (Progress Reward + reached_box_flag)

*Inspired by DeepMind MuJoCo Playground `PandaPickCubeCartesian` and the progress reward technique.*

### 4.1 Contact Detection Fix
Replace the impossible threshold with a physics-meaningful signal:
```python
gripper_grasping = (gripper_cmd > 0) and (left_touch or right_touch) and ee_at_grasp_height
```
- `gripper_cmd > 0`: policy is commanding close
- `left_touch or right_touch`: at least one finger has physical contact (MuJoCo geom contact check)
- `ee_at_grasp_height = ee_pos[2] <= initial_object_z + 0.06`: EE is near the bottle, not above it (prevents "fist-close and push down" exploit)

### 4.2 Two-Phase Reward with reached_box_flag

**Phase 1 — Approach** (before flag):
```python
score = 4.0 * (1.0 - np.tanh(5.0 * gt_distance))
# Coefficient 5 gives meaningful gradient at 20 cm (≈ 0.26), not ~0 like coeff 12
```
Maximum score = 4.0 (at gt_distance = 0).

**Phase transition** — flag latches once EE is within 5 cm of bottle:
```python
if not self._reached_box and gt_distance < 0.05:
    self._reached_box = True  # never resets within episode
```

**Phase 2 — Grasp + Lift** (after flag):
```python
lift_fraction = min(object_lift / 0.02, 1.0)   # 0.02 m = success threshold
score = 4.0                                      # full approach credit held
score += 2.0 if gripper_grasping else 0.0        # contact bonus
score += 8.0 * lift_fraction                     # lift progress, max 8.0
# Total max score = 14.0
```

### 4.3 Progress Reward Wrapper
Only pay for **improvements** over the episode-best score:
```python
progress_reward = max(score - self._best_score, 0.0)
self._best_score = max(self._best_score, score)
```
This eliminates every farming exploit by construction: you cannot gain reward by staying in any state twice. Any strategy that doesn't make progress is worth exactly zero.

### 4.4 Full Reward
```python
reward = (
    progress_reward
    - 0.5 * premature_close_penalty   # penalize closing when far from bottle
    - 0.01 * action_penalty            # L2 regularization on actions
    - 1.0 * table_penalty              # collision with table surface
)
if success:
    reward += config.success_bonus     # +200
```

### 4.5 Success Condition
Object lifted ≥ 2 cm **with at least one finger in contact**, held for 3 consecutive steps:
```python
if object_lift >= 0.02 and (left_touch or right_touch):
    self._success_hold_count += 1
else:
    self._success_hold_count = 0
return self._success_hold_count >= 3
```

### 4.6 Training Configuration

| Parameter | Value |
|---|---|
| Algorithm | SAC (Soft Actor-Critic) |
| Policy | MlpPolicy |
| Learning rate | 3e-4 |
| Buffer size | 1,000,000 |
| Batch size | 256 |
| γ (discount) | 0.99 |
| τ (soft update) | 0.005 |
| SDE | Yes (State-Dependent Exploration) |
| n_envs | 24 (SubprocVecEnv) |
| Timesteps | 1,000,000 |
| VecNormalize | Yes (obs + reward) |

**Train command:**
```
python src\rl\local_pick\train_cartesian_direct_lift.py ^
  --timesteps 1000000 --n-envs 24 ^
  --model-path models/local_pick_option_a_1m ^
  --approach-noise 0.01 --approach-z-noise 0.01 --joint-noise 0.01
```

**Eval command:**
```
python src\rl\local_pick\evaluate_cartesian.py ^
  --model-path models/local_pick_option_a_1m --episodes 5 --direct-lift
```

**Key files:**
- `src/rl/local_pick/local_pick_cartesian_direct_lift_env.py`
- `src/rl/local_pick/train_cartesian_direct_lift.py`
- `OPTION_A_PROGRESS_REWARD.md`

---

## 5. Option B — Pixel SAC (DrQ-v2 Style)

*Inspired by pick-101 (github.com/ggand0/pick-101) and DrQ-v2.*

### 5.1 Motivation
Option A uses an engineered 19-D state vector derived from perception. Option B removes the engineering step entirely: the **raw wrist-cam pixels are the observation**. The CNN learns what features matter directly from images.

This is more aligned with end-to-end perception-based learning. At inference, no object detection pipeline is needed — the policy directly maps pixels to actions.

### 5.2 Observation Space
- **84×84×3 uint8 RGB** from the wrist camera (same camera as Option A)
- **3-frame stack** via `VecFrameStack(n_stack=3)` → policy sees `(84, 84, 9)` tensor
- Frame stacking gives the policy implicit velocity information (motion between frames)

### 5.3 Why This Works Here
Because the approach pose always places the wrist cam approximately the same distance above the bottle (camera on the hand, approach = 20 cm above bottle), the visual appearance at episode start is consistent regardless of where on the table the bottle is. The CNN doesn't need to generalize over wildly different viewpoints — the pipeline handles that by setting up the approach.

### 5.4 Architecture
```
Wrist cam RGB (84×84×3)
       ↓  [×3 frame stack]
  (84, 84, 9)
       ↓
  NatureCNN encoder (3 conv layers → 512-D latent)
       ↓
  SAC actor / twin critics (MLP heads)
       ↓
  4-D Cartesian action
```
NatureCNN is the same architecture as the original DQN paper (Nature 2015), built into SB3's CnnPolicy.

### 5.5 Differences from Option A

| | Option A | Option B |
|---|---|---|
| Observation | 19-D engineered state | 84×84 wrist cam (×3 stacked) |
| Policy type | MlpPolicy | CnnPolicy (NatureCNN) |
| Buffer size | 1M | 100K (images are ~64 KB each) |
| Learning rate | 3e-4 | 1e-4 (CNN needs slower updates) |
| SDE | Yes | No (SDE + CNN unstable) |
| n_envs | 24 | 8 (more VRAM per env for CNN) |
| Timesteps | 1M | 2M |
| Reward normalization | Yes | Reward only (pixel obs must stay uint8) |

### 5.6 Why SAC instead of DrQ-v2?
Pick-101 uses DrQ-v2 (DDPG-based) from a custom library (RoboBase). We use SAC with CnnPolicy (SB3 built-in) because:
- SAC is generally more sample-efficient than DDPG (entropy bonus encourages exploration)
- SB3's CnnPolicy is production-quality and well-tested
- No external dependencies needed
- The structural difference is minimal for this task scale

### 5.7 Training Configuration

| Parameter | Value |
|---|---|
| Algorithm | SAC |
| Policy | CnnPolicy (NatureCNN) |
| Learning rate | 1e-4 |
| Buffer size | 100,000 |
| Batch size | 256 |
| γ (discount) | 0.99 |
| Frame stack | 3 |
| n_envs | 8 |
| Timesteps | 2,000,000 |
| VecNormalize | Reward only |

**Train command:**
```
python src\rl\local_pick\train_cartesian_pixel.py ^
  --timesteps 2000000 --n-envs 8 ^
  --model-path models/local_pick_option_b_pixel_2m ^
  --approach-noise 0.01 --approach-z-noise 0.01 --joint-noise 0.01
```

**Eval command:**
```
python src\rl\local_pick\evaluate_cartesian_pixel.py ^
  --model-path models/local_pick_option_b_pixel_2m --episodes 5
```

**Key files:**
- `src/rl/local_pick/local_pick_cartesian_pixel_env.py`
- `src/rl/local_pick/train_cartesian_pixel.py`
- `src/rl/local_pick/evaluate_cartesian_pixel.py`
- `OPTION_B_PIXEL_SAC.md`

---

## 6. Design Decisions Summary

### Why asymmetric reward (GT for reward, camera for obs)?
Using camera-estimated position for reward is noisy — the estimator fails when fingers occlude the bottle. GT physics gives a clean, always-accurate training signal. The policy still only sees camera-derived observations, so the perception requirement is met. This is standard practice (asymmetric actor-critic) in sim-to-real robotics.

### Why progress reward over dense reward?
Dense rewards create local optima (camping at contact, repeated approach motions, etc.). Progress reward kills all of these by construction: you can only gain reward by doing something you haven't done before this episode. This is the core technique used in DeepMind's MuJoCo Playground benchmark environments.

### Why two phases instead of one blended reward?
A single blended reward creates ambiguity — the policy can trade approach quality for lift attempts (or vice versa) in ways that stall learning. Two phases with a hard gate force the policy to master approach first before lift reward becomes available. The `reached_box` latch makes the gate one-way: once earned, the lift phase stays active.

### Why Cartesian delta over joint-space delta?
Joint-space deltas require the policy to implicitly learn the Jacobian (relationship between joint motion and EE motion). Cartesian deltas express the task directly: move toward the bottle (Δx, Δy, Δz). The IK layer handles the Jacobian for us. This dramatically reduces the credit assignment problem.

### Why not end-to-end from camera to full arm motion?
The full task (detect → approach → grasp → transport) is too long-horizon for a single RL policy to learn from scratch. Breaking it at the approach pose gives RL a well-conditioned starting state and keeps the episode length short (125 steps), which is critical for sample efficiency.

---

## 7. Reference: Key Numbers

| Quantity | Value |
|---|---|
| Success = object lift | ≥ 0.02 m |
| Success hold duration | 3 consecutive steps |
| Episode length | 125 steps |
| Frame skip | 20 MuJoCo substeps |
| Approach height | 0.20 m above bottle |
| Joint 7 at approach | 2.8973 rad (joint limit) |
| Gripper contact threshold | ≤ 0.035 m opening (allows bottle) |
| reach_box threshold | GT distance < 0.05 m |
| Phase 1 score range | 0 – 4.0 |
| Phase 2 score range | 4.0 – 14.0 |

---

## 8. Reference Repositories

- **pick-101** (github.com/ggand0/pick-101): MuJoCo + Panda + SAC (state) + DrQ-v2 (pixel), 100% at 1M (state) and 2M (pixel)
- **MuJoCo Playground** (DeepMind): Progress reward pattern, PandaPickCubeCartesian benchmark
- **stable-baselines3**: SAC, CnnPolicy, VecNormalize, VecFrameStack

---

## 9. pip Dependencies

```
pip install stable-baselines3[extra]>=2.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install mujoco
pip install gymnasium
pip install Pillow
pip install opencv-python-headless
```

Note: `cu128` is required for RTX 5070 (Blackwell architecture, CUDA 12.8+).
