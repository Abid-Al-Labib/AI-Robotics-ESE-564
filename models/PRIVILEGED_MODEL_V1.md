# Privileged Model V1

## Overview
SAC policy for local pick-and-place on a Panda 7-DOF arm in MuJoCo.
Called "privileged" because observations come directly from simulator state
(ground-truth joint positions, object position, EE position) — not from perception.
This is the baseline before transitioning to camera-based perception.

## Pipeline Role
The policy handles only the **local pick** phase:
- Starts with arm already at approach pose (~16cm above object)
- Descends, grips the bottle, lifts ~2cm above initial z
- Hands off to the scripted pipeline on success

Post-handoff pipeline:
1. Motor lift straight up to z=0.38m (clearance)
2. RRT plans from clearance to place approach (+25cm above goal)
3. Gripper opens → drop

## Results
- **8/10 trials** successful end-to-end pick-and-place
- Picks up reliably; occasional failure is grip slip before success threshold

## Model File
`privileged_model_v1.zip`

## Training Config
| Parameter | Value |
|---|---|
| Algorithm | SAC (stable-baselines3) |
| Total timesteps | 1,000,000 |
| max_episode_steps | 75 |
| action_scale | 0.02 rad/step |
| frame_skip | 20 |
| gamma | 0.99 |
| gradient_steps | 2 |
| learning_rate | 3e-4 |
| batch_size | 256 |
| buffer_size | 200,000 |

## Randomization
| Parameter | Value |
|---|---|
| Object position | Full table random (x: 0.43–0.57, y: -0.20 to -0.10) |
| Object XY noise | ±0.01m |
| Approach XY noise | ±0.02m |
| Joint noise | ±0.01 rad |

## Observation Space (20-dim)
| Component | Dims |
|---|---|
| Joint positions (joint1–7) | 7 |
| Joint velocities (joint1–7) | 7 |
| EE position − object position | 3 |
| Object lift (z − initial_z) | 1 |
| Gripper opening (finger_joint1 qpos) | 1 |
| Phase (0=open, 1=closed) | 1 |

## Action Space (8-dim)
| Component | Dims |
|---|---|
| Joint deltas (joint1–7) × action_scale | 7 |
| Gripper command (>0 = close) | 1 |

## Reward Function
```
reward = -reach_weight * reach_distance
       + lift_weight * object_lift
       + align_close_bonus      (if gripper_cmd>0 and reach_dist <= 0.015)
       + contact_bonus           (if both fingers touching and gripper_cmd>0)
       + hold_bonus              (scales with lift height, max at lift_success_z)
       - premature_close_penalty (if gripper_cmd>0 and reach_dist > 0.06)
       - action_penalty_weight * ||joint_action||^2
       - table_collision_penalty (arm hits table or center_obstacle)
       + success_bonus           (object_z >= lift_success_z)
```

| Parameter | Value |
|---|---|
| reach_weight | 1.0 |
| lift_weight | 20.0 |
| lift_success_z | 0.145m |
| success_bonus | 200.0 |
| align_close_bonus | 2.0/step |
| align_close_threshold | 0.015m |
| contact_reward | 2.0/step |
| hold_reward | 5.0/step (scaled by lift fraction) |
| hold_lift_threshold | 0.01m |
| premature_close_penalty | 5.0/step |
| premature_close_threshold | 0.06m |
| table_collision_penalty | 2.0/step |
| action_penalty_weight | 0.01 |

## Scene
- Bottle lying on side at z=0.125m (quat 0.7071 0.7071 0 0)
- Center obstacle divider at y=0, top at z=0.18m
- Finger friction: 3.0, gripper stiffness: biasprm=-800

## Known Issues / Next Steps
See `REVIEW_NOTES.md` for full list. Key items:
1. IK solution at approach differs between training (home-biased) and pipeline (current-biased) → wrist angle mismatch
2. Approach height: training 0.16m, pipeline ~0.176m (perceived z + 0.15)
3. Transition to perception-based observations (camera instead of ground-truth)
