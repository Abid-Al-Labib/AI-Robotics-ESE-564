# AI Robotics Final Project

Hybrid camera-guided planning and local reinforcement learning for MuJoCo bottle pick-and-place with a Franka Panda arm.

This repository contains:
- a MuJoCo scene with a Panda arm, bottle-like object, goal platform, and center divider obstacle
- a perception module that estimates bottle and goal positions from RGB-D images
- IKFast-based inverse kinematics
- an RRT planner in 7D joint space
- low-level arm and gripper controllers
- a local SAC policy for the contact-rich pick phase

The main runtime is `src/main.py`.

## What The Full Pipeline Does

At runtime, the system:
1. Randomizes the bottle and goal platform positions.
2. Uses the camera in `assets/mujoco/scene.xml` to render RGB and depth.
3. Estimates the bottle and goal positions from color-thresholded image detections.
4. Uses IKFast and RRT to move the arm to a pick approach pose.
5. Runs a trained SAC local-pick policy to descend, grasp, and lift the bottle.
6. Lifts to a clearance pose above the divider.
7. Uses RRT again to move to a place approach pose above the goal.
8. Opens the gripper to release the bottle.

## Repository Layout

```text
AI-Robotics-ESE-564/
  README.md
  report.tex
  requirements.txt
  REVIEW_NOTES.md
  configs/
  assets/
    mujoco/
      scene.xml
      panda.xml
      panda_assets/
  models/
    privileged_model_v1.zip
    PRIVILEGED_MODEL_V1.md
  src/
    main.py
    requirements.txt
    environment/
      mujoco_env.py
    pipeline/
      controller/
        arm_controller.py
        gripper_controls.py
        kinematics.py
        ikfast/
      perception/
        perception.py
      planner/
        rrt_planner.py
      localPick/
        rl_local_pick.py
    rl/
      local_pick/
        config.py
        local_pick_env.py
        train.py
        evaluate.py
        train_align.py
        evaluate_align.py
        test_approach.py
```

## Python Version And Environment

Use Python `3.11.x`.

For this project, you should treat Python `3.11` as required, not optional.

Why:
- the checked-in local environment used for inspection was Python `3.11.8`
- the checked-in IKFast build artifact for macOS was compiled for CPython 3.11
- MuJoCo in this project setup was not reliable on higher Python versions
- the rest of the runtime depends on using the same interpreter for MuJoCo, Stable-Baselines3, and the compiled IKFast extension

If you use a different Python version, especially a higher one, expect MuJoCo and/or the compiled IKFast module to fail.

## System Requirements

You need:
- Python `3.11`
- a working C/C++ build toolchain
- a GUI-capable environment if you want to run `src/main.py` or any rendered evaluator

Platform notes:
- macOS: install Xcode Command Line Tools with `xcode-select --install`
- Ubuntu/Debian: install `build-essential` and `python3.11-dev`
- Windows: use a Python environment with C++ build tools available, then rebuild IKFast for that interpreter

## Setup

Run all commands from the repository root unless noted otherwise.

### 1. Create a virtual environment

macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

Optional sanity check:

```bash
python --version
which python
```

You want the active interpreter to be Python 3.11 inside `.venv`.

### 1a. macOS quick path

If you are on macOS, this is the exact path you should follow:

```bash
xcode-select --install
python3.11 -m venv .venv
source .venv/bin/activate
python --version
```

The version printed here should be `Python 3.11.x`.

### 1b. Windows quick path

If you are on Windows PowerShell, use:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python --version
```

The version printed here should be `Python 3.11.x`.

### 2. Upgrade packaging tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 3. Install project dependencies

The root `requirements.txt` forwards to `src/requirements.txt`.

```bash
pip install -r requirements.txt
```

This installs:
- `mujoco`
- `gymnasium`
- `stable-baselines3`
- `torch`
- `Pillow`
- other runtime dependencies used by the simulator and RL code

### 4. Build the IKFast extension

This step is required before running the main pipeline.

`src/pipeline/controller/kinematics.py` imports the compiled IKFast module from:

```text
src/pipeline/controller/ikfast/
```

Build it with the same Python interpreter that you will use for `src/main.py` or `mjpython src/main.py`.

```bash
cd src/pipeline/controller/ikfast
python setup.py build_ext --inplace
cd ../../../..
```

What this does:
- compiles `ikfast_panda_arm.cpp`
- builds a Python extension module
- places the generated module directly in `src/pipeline/controller/ikfast/`

Expected output:
- on macOS/Linux, a file similar to `ikfast_panda_arm.cpython-311-*.so`
- on Windows, a file similar to `ikfast_panda_arm.cp311-*.pyd`

If this step fails:
- make sure your compiler toolchain is installed
- make sure you are using Python 3.11
- make sure you built with the same interpreter that will run `src/main.py`

### 4a. macOS IKFast build command

On macOS, use this exact sequence:

```bash
source .venv/bin/activate
cd src/pipeline/controller/ikfast
python setup.py build_ext --inplace
cd ../../../..
```

### 4b. Windows IKFast build command

On Windows PowerShell, use this exact sequence:

```powershell
.venv\Scripts\Activate.ps1
cd src\pipeline\controller\ikfast
python setup.py build_ext --inplace
cd ..\..\..\..
```

### 5. Optional: verify the IKFast build

From the repo root:

```bash
python -c "import sys; sys.path.append('src'); from pipeline.controller.kinematics import Kinematics; print('IKFast import OK')"
```

If this import fails, the main pipeline will also fail.

## Running The Full Pipeline

### macOS run command

On macOS, use:

```bash
mjpython src/main.py
```

Why `mjpython` on macOS:
- MuJoCo viewer-based scripts are often more reliable when launched through `mjpython`
- it helps with GUI and event-loop issues that can happen when using plain `python`
- this is the safer way to run viewer-based MuJoCo programs on macOS

After installing `mujoco`, `mjpython` should be available inside the active virtual environment. You can check with:

```bash
which mjpython
```

### Windows run command

On Windows, run:

```powershell
python src/main.py
```

Windows does not use `mjpython` in the same way as the macOS workflow in this repository. Use the Python interpreter from the active virtual environment.

### What `src/main.py` expects

- MuJoCo and the renderer installed correctly
- the IKFast extension built successfully
- the trained local-pick model present at `models/privileged_model_v1.zip`

If the model file is missing, `src/main.py` will still run the global planning stages but will skip the RL local pick phase.

### What `src/main.py` does

For each of 10 trials, the script:
- samples randomized bottle and goal positions
- resets the MuJoCo scene
- launches a passive MuJoCo viewer
- perceives the bottle and goal from the camera
- plans to a pick approach with IK + RRT
- executes the arm motion with motor targets
- runs the trained local-pick SAC controller
- if the lift succeeds, raises the object to a clearance height
- plans to the place approach pose
- opens the gripper to drop the bottle over the goal platform

### Notes about running the main script

- `src/main.py` is viewer-based. It is intended for an interactive desktop session, not a headless server.
- The script currently runs `10` randomized trials by default.
- The script prints runtime diagnostics such as perceived positions, planner progress, and local-pick success.
- On macOS, use `mjpython src/main.py` instead of plain `python src/main.py`.
- On Windows, use `python src/main.py` from the active `.venv`.

## Training The Local Pick Policy

The local RL task is isolated from the full pipeline. Training does not run the full perception + planning stack on every episode. Instead, it resets directly to a local approach pose above the bottle.

Training entry point:

```bash
python src/rl/local_pick/train.py
```

### Quick baseline training run

```bash
python src/rl/local_pick/train.py --timesteps 100000
```

This is a small sanity-check run and not the strongest configuration.

### Wider-randomization training

```bash
python src/rl/local_pick/train.py \
  --timesteps 300000 \
  --object-noise 0.02 \
  --approach-noise 0.01 \
  --joint-noise 0.01
```

### Full-table randomized training

This better matches the global runtime, where the bottle can appear across the full pick-side region.

```bash
python src/rl/local_pick/train.py \
  --timesteps 300000 \
  --full-table-random \
  --approach-noise 0.01 \
  --joint-noise 0.01 \
  --model-path models/local_pick_sac_full_table
```

### Recommended retraining command from project review notes

`REVIEW_NOTES.md` recommends a stronger retrain setup to better match runtime noise:

```bash
python src/rl/local_pick/train.py \
  --timesteps 600000 \
  --model-path models/local_pick_sac_v2 \
  --full-table-random \
  --object-noise 0.01 \
  --approach-noise 0.025 \
  --joint-noise 0.01
```

### What training script saves

Training writes the model to the `--model-path` you provide. Stable-Baselines3 will save it as a `.zip` model artifact.

Examples:
- `models/local_pick_sac.zip`
- `models/local_pick_sac_full_table.zip`
- `models/local_pick_sac_v2.zip`

## Evaluating The Local Pick Policy

Evaluation entry point:

```bash
python src/rl/local_pick/evaluate.py --model-path models/local_pick_sac.zip
```

### Headless evaluation

Use this when you do not want a viewer window:

```bash
python src/rl/local_pick/evaluate.py \
  --model-path models/privileged_model_v1.zip \
  --episodes 20 \
  --full-table-random \
  --object-noise 0.01 \
  --approach-noise 0.02 \
  --joint-noise 0.01 \
  --no-render
```

### What evaluation prints

Per episode, it reports:
- total reward
- episode length
- success flag
- average reach distance
- accumulated lift
- shaped reward components such as contact bonus and hold bonus

At the end it prints:

```text
=== Success rate: X/Y ===
```

## Training And Evaluating The Align-Only Variant

The repository also contains an align-only local RL variant. This model learns to align the gripper near the object, while gripping and lift remain more scripted.

### Train align-only model

```bash
python src/rl/local_pick/train_align.py \
  --timesteps 300000 \
  --full-table-random \
  --approach-noise 0.01 \
  --joint-noise 0.01 \
  --model-path models/local_pick_align_full_table
```

### Evaluate align-only model

```bash
python src/rl/local_pick/evaluate_align.py \
  --model-path models/local_pick_align_full_table.zip \
  --full-table-random \
  --episodes 10
```

Note: `evaluate_align.py` currently uses a human viewer and does not expose a `--no-render` flag.

## Debug And Sanity-Check Utilities

Useful helper scripts:

### Camera render check

```bash
python src/camera_test.py
```

This:
- loads the MuJoCo scene
- renders RGB and depth from `perception_cam`
- saves `camera_test.png`

### Camera pose / calibration debug

```bash
python src/debug_cam.py
```

This prints:
- camera world position
- camera rotation matrix
- field of view
- near/far settings

### Local pick approach pose visual check

```bash
mjpython src/rl/local_pick/test_approach.py
```

This opens the viewer and shows the approach pose used for the local RL task.

On non-macOS platforms, plain `python` is usually fine for this script.

## Known Model Artifact

The repository includes:

```text
models/privileged_model_v1.zip
models/PRIVILEGED_MODEL_V1.md
```

Important note:
- this is a privileged local-pick model
- it observes simulator state during the local pick phase
- it is not a fully vision-based local controller

The accompanying markdown note records:
- an `8/10` end-to-end result for the shipped checkpoint
- the training assumptions used for that artifact

## Common Problems And Fixes

### `ModuleNotFoundError: No module named 'stable_baselines3'`

You did not install the full project requirements into the active virtual environment.

Fix:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### `ImportError` for `ikfast_panda_arm`

The IKFast extension was not built, or it was built with a different Python interpreter than the one you are using now.

Fix:

```bash
source .venv/bin/activate
cd src/pipeline/controller/ikfast
python setup.py build_ext --inplace
cd ../../../..
```

Then rerun:

```bash
python -c "import sys; sys.path.append('src'); from pipeline.controller.kinematics import Kinematics; print('IKFast import OK')"
```

### Build succeeds but `src/main.py` still fails on IKFast

You likely built IKFast with one Python and ran `main.py` with another.

Check:

```bash
which python
python --version
```

Rebuild IKFast with that exact interpreter active.

### MuJoCo viewer issues on macOS

If the script starts but the viewer crashes, hangs, or behaves strangely, use:

```bash
mjpython src/main.py
```

and for viewer-based debug scripts:

```bash
mjpython src/rl/local_pick/test_approach.py
```

For this project on macOS, `mjpython` should be the default launcher for viewer-based MuJoCo scripts.

### MuJoCo not working on higher Python versions

For this repository, use Python `3.11`.

Do not assume Python `3.12+` will work just because the environment installs. In this project setup, MuJoCo was not reliable on higher versions, and even if MuJoCo installs, the IKFast build and runtime imports may still break.

### `src/main.py` opens but does not do the local pick phase

This usually means the model file was not found.

Check that this file exists:

```text
models/privileged_model_v1.zip
```

If it does not exist, the script will print that it is skipping RL local pick.

### Viewer or rendering problems

`src/main.py` and some debug/evaluation scripts need a GUI-capable environment. If you are on a headless machine:
- use `evaluate.py --no-render` for local policy evaluation
- avoid `src/main.py` unless you provide a display

### Training/evaluation mismatch

This repository has a documented mismatch between:
- the current checked-in `train.py` defaults
- the configuration recorded in `models/PRIVILEGED_MODEL_V1.md`

If you are trying to reproduce the shipped model exactly, treat `models/PRIVILEGED_MODEL_V1.md` as the authoritative note for that artifact.

## Reproducibility Notes

- The main runtime currently evaluates 10 randomized trials.
- Bottle and goal positions are randomized, but bottle orientation and obstacle placement are fixed.
- The local-pick model is privileged during deployment.
- The full runtime does not yet log a clean benchmark file automatically.

## Typical End-To-End Setup Checklist

### macOS checklist

If you just want the shortest path to getting `src/main.py` working on macOS, do this:

```bash
xcode-select --install
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
cd src/pipeline/controller/ikfast
python setup.py build_ext --inplace
cd ../../../..
mjpython src/main.py
```

### Windows checklist

If you just want the shortest path to getting `src/main.py` working on Windows PowerShell, do this:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
cd src\pipeline\controller\ikfast
python setup.py build_ext --inplace
cd ..\..\..\..
python src/main.py
```

## Deactivating The Environment

When you are done:

```bash
deactivate
```
