
# import for shapes
from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape

# for point clouds
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.const import RenderMode

# import for robot 
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper

# aux imports
import numpy as np


class MyRobot(object):
  
    def __init__(self, arm, gripper):
        self.arm = arm
        self.gripper = gripper
    
    def get_state(self):
        return{
            'arm' : self.arm.get_joint_positions(),
            'end-effector' : (self.arm.get_tip().get_position(), self.arm.get_tip().get_quaternion()),
            'gripper-state': self.gripper.get_open_amount()
        }
    
    def set_robot_arm_velocities(self, velocities, wait_for_movement=True, timeout=5.0):
        assert np.all(velocities<=1)  # keep velocity of robot less than 1 for caution 
        # self.arm.set_joint_target_velocities(velocities)
        if wait_for_movement:
            # Get initial position
            initial_positions = self.arm.get_joint_positions()
            
            # Set velocities
            self.arm.set_joint_target_velocities(velocities)
            
            # Wait until robot actually moves
            start_time = time.time()
            movement_threshold = 0.001  # Small threshold to detect movement
            
            while True:
                current_positions = self.arm.get_joint_positions()
                position_diff = np.abs(np.array(current_positions) - np.array(initial_positions))
                
                # Check if any joint has moved beyond threshold
                if np.any(position_diff > movement_threshold):
                    break
                    
                # Timeout check
                if time.time() - start_time > timeout:
                    print("Warning: Robot movement timeout")
                    break
                    
                time.sleep(0.01)  # Small delay
        else:
            self.arm.set_joint_target_velocities(velocities)

    def actuate_gripper(self, open_percentage , velocity):
        assert open_percentage >= 0 and open_percentage <=1
        assert np.all(velocity<=1)  # keep velocity of robot less than 1 for caution 
        done = self.gripper.actuate(open_percentage, velocity)
        # assert done is True





class Task():
    # TODO RETURN observations of point clouds and gripper pose
    def __init__(self, scene, max_step = 10):
        self.pr = PyRep()
        self.timestep = 0

        # init scene
        self.pr.launch(scene)
        self.pr.start()

        # inititate robot
        self.robot = self._make_robot()

        # init cameras
        self._make_cameras()

        # init object
        self.start_position = [1,1,0]
        self.obj = self._make_object(self.start_position)

        # init objective
        self.end_position = [2,2,0]        
        self.objective = lambda  : np.all(self.obj.get_position() == self.end_position)

        # init end condition 
        self.end_condition = lambda  : self.objective() or self.timestep == max_step


    def step(self, robot_velocities, gripper_open_percentage, gripper_velocity):
        self.robot.set_robot_arm_velocities(robot_velocities)
        self.robot.actuate_gripper(gripper_open_percentage, gripper_velocity)
        self.pr.step()
        self.timestep +=1
        return self._get_obs()

    def end(self):
        self.pr.stop()
        self.pr.shutdown()

    # aux function
    def _make_object(
            self,
            position,
            obj_type=PrimitiveShape.SPHERE, 
            color=[1,0, 0.1, 0.1], 
            size=[0.05, 0.05, 0.05],
        ):
            obj = Shape.create(type=obj_type,
                                color=color,
                                size= size,
                                static = False,
                                respondable = True)
            obj.set_color(color)
            obj.set_position(position)
            assert obj is not None 
            # assert obj.color is color
            # assert obj.position is position
            # assert obj.size is size
            return obj
    
    def _make_robot(self):
        arm = Panda()  # Get the panda from the scene
        gripper = PandaGripper()
        robot = MyRobot(arm,gripper)

        return robot

    def _make_cameras(self):
        VisionSensor.create(resolution=[512,512])
        self.vision_sensor = VisionSensor('Vision_sensor')
        
        # cameras config
        # Configure vision sensor for depth/point cloud capture
        self.vision_sensor.set_render_mode(RenderMode.OPENGL3)
        self.vision_sensor.set_explicit_handling(True)  # Important for manual capture control

        # Set resolution if needed
        self.vision_sensor.set_resolution([512, 512])

        # Optional: Set additional parameters
        self.vision_sensor.set_near_clipping_plane(0.01)
        self.vision_sensor.set_far_clipping_plane(10.0)

    def _get_point_clouds(self):
        point_cloud = self.vision_sensor.capture_pointcloud()
        # Shape: (width*height, 3) - may contain nan values for invalid depth points

        # Optionally also get RGB data
        rgb_data = self.vision_sensor.capture_rgb()  # Shape: (height, width, 3)

        # Create colored point cloud (XYZRGB)
        height, width, _ = rgb_data.shape
        rgb_flattened = rgb_data.reshape(height*width, 3)

        # Filter out invalid points (NaN values)
        # valid_indices = ~np.isnan(point_cloud).any(axis=1)
        # valid_points = point_cloud[valid_indices]
        # valid_colors = rgb_flattened[valid_indices]
        valid_points = point_cloud
        valid_colors = rgb_flattened

        # Combine points and colors
        # colored_point_cloud = np.hstack((valid_points, valid_colors))        
        # return colored_point_cloud
        return valid_points

    def _transform_pointclouds_to_world(self, pointclouds):
        sensor_pose = self.vision_sensor.get_matrix()
        rotation = sensor_pose[:3, :3]
        translation = sensor_pose[:3, 3]
        
        # Apply transformation
        world_points = pointclouds @ rotation.T + translation
        return world_points
    
    def _get_obs(self):
        point_clouds = self._get_point_clouds()
        w_point_clouds = self._transform_pointclouds_to_world(point_clouds)
        robot_state = self.robot.get_state()
        return {
            'point-clouds' : w_point_clouds,
            'robot-state' : robot_state,
            'object-position' : self.obj.get_position(),
            'time-step' : self.timestep,
            'objective' : self.objective(),
            'done' : self.end_condition(),
        }








if __name__  == "__main__":
    task = Task(scene='scene_panda_reach_target.ttt', max_step=100)
    import time

    for _ in range(100):
        robot_vel = np.random.random((7))
        print(robot_vel)
        gripper_open_percentage = int(np.random.random() <= 0.5)
        gripper_velocity = 0.1
        task.step(robot_vel, gripper_open_percentage, gripper_velocity)
    task.end()