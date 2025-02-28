import pybullet as p
import pybullet_data
import time

def check_pybullet_installation():
    """
    Verify PyBullet installation and robot model loading
    """
    try:
        # Connect to PyBullet
        physicsClient = p.connect(p.GUI)
        
        # Add search path for robot models
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        print(f"PyBullet data path: {pybullet_data.getDataPath()}")
        
        # Load plane
        p.loadURDF("plane.urdf")
        
        # Try to load the KUKA model
        robot = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])
        
        if robot >= 0:
            print("Successfully loaded KUKA IIWA model!")
            
            # Print available joints
            num_joints = p.getNumJoints(robot)
            print(f"\nNumber of joints: {num_joints}")
            print("\nJoint information:")
            for i in range(num_joints):
                info = p.getJointInfo(robot, i)
                print(f"Joint {i}: {info[1].decode('utf-8')}")
        
        time.sleep(5)  # Keep the window open to see the robot
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        p.disconnect()

if __name__ == "__main__":
    print("Checking PyBullet installation...")
    check_pybullet_installation() 