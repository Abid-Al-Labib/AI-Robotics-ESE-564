# ESE 564 Final Project

This project implements a robot pushing task in MuJoCo.

## Description
This repository contains the codebase for setting up a robotic simulation environment in MuJoCo, integrating perception, planning, control, and evaluation pipelines to execute and analyze a robot task.

## Installation

```bash
pip install -r requirements.txt
```

## Running the Project

The main entry point for the project is `src/main.py`.

```bash
python src/main.py
```

## Project Structure
- `configs/`: Configuration files for simulation, camera, planner, and task.
- `assets/`: Simulation assets including MuJoCo XMLs, objects, meshes, and textures.
- `src/`: Source code
  - `main.py`: Top-level pipeline runner
  - `env/`: Simulator setup and interaction wrappers.
  - `pipeline/`: Core components of the robotic system
    - `perception/`: Vision and tracking.
    - `motion/`: motion planning and controlling.
    - `pickandplace/`: local pick and place task.
    - `models/`: trained models.
  - `utils/`: Reusable helpers like geometry and logging.
- `scripts/`: Useful debug scripts.
- `outputs/`: Logs, videos, images, and experiment results.
- `notebooks/`: Jupyter notebooks used for training.
