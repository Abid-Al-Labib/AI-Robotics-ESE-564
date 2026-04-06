import os
import sys
import time
from pathlib import Path
import mujoco
import mujoco.viewer

# Add the src directory to Python path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.mujoco_env import MujocoEnv

def main():
    print("Initializing MuJoCo Simulation Pipeline...")

    # Determine paths relative to root directory
    project_root = Path(__file__).resolve().parent.parent
    xml_path = project_root / "assets" / "mujoco" / "scene.xml"
    
    # Initialize the basic environment
    env = MujocoEnv(str(xml_path))
    
    print("Environment loaded. Launching Viewer...")
    
    # Launch viewer and step through the simulation interactively
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        start_time = time.time()
        
        while viewer.is_running():
            env.step()
            viewer.sync()
            time.sleep(env.model.opt.timestep)

    print("Simulation finished.")

if __name__ == "__main__":
    main()
