# Robot Localization Project
By: Aditi Lagisetty and Avani Patil

<img width="480" height="437" alt="demo_take_1" src="https://github.com/user-attachments/assets/7c7d5882-8bc8-4f9f-852d-eec5d102f711" />

## Project Goal
The goal of this project is to localize a Neato robot in a known map using its wheel odometry and its lidar. Odometry alone cannot do this: it only measures how far the wheels have turned, so small errors add up and the robot's estimated position drifts further from the truth over time (Figure 3 shows up to 0.9 m of drift in a 46-second drive). It also says nothing about where the robot is in the map, only how far it has moved since it started. We implemented a particle filter that fixes both problems by comparing what the lidar actually sees with what it would see from many candidate poses. The filter publishes its estimate as the map → odom transform, which corrects the drift of odometry so that the rest of the ROS system knows where the robot is on the map.

## Approach
The filter keeps a cloud of N = 300 particles. Each particle is one guess of the robot's pose (x, y, θ) in the map frame, together with a weight w that says how believable that guess is. A tight cloud means the filter is confident; a spread-out cloud means it is uncertain. Table 1 lists the steps of one filter update.

| Step                  | What Happens                                                                    | Effect on Uncertainty                  |
|-----------------------|---------------------------------------------------------------------------------|----------------------------------------|
| 1. Motion update      | Move every particle by the change in odometry, plus a little random noise.      | Grows (the cloud spreads out)          |
| 2. Observation update | Weight every particle by how well the lidar scan matches the map from its pose. | Shrinks (bad guesses get low weight)   |
| 3. Pose estimate      | Average the heaviest cluster of particles into one pose and publish it.         | --                                     |
| 4. Resample           | Draw a new cloud, copying particles in proportion to their weight.              | Concentrates the cloud on good guesses |

## How to Run
**Prerequisites.** ROS 2 Jazzy, Python 3, nav2 (nav2_map_server, nav2_lifecycle_manager, nav2_msgs), and the course's Neato packages. The Neato packages are not standard ROS packages; set them up following the CompRobo course website and the comprobo26/neato_packages repository. The offline scripts also need NumPy, SciPy, scikit-learn, Pillow, Matplotlib, and PyYAML.

**Build** From your ROS2 workspace
```bash
cd ~/ros2_ws/src && git clone https://github.com/avani-patil07/robot_localization_comprobo26.git
cd ~/ros2_ws && colcon build --symlink-install && source install/setup.bash
```

**Run the filter on a recorded bag** 
```bash
ros2 launch robot_localization test_pf.py map_yaml:=$PWD/maps/mac_1st_floor_final.yaml
ros2 bag play bags/macfirst_floor_take_1 --clock
rviz2 -d rviz/pf_demo.rviz
```
In rviz, use "2D Pose Estimate" to give the filter an initial guess, or let it start from odometry. To change a parameter without editing code, pass your own YAML file (params_file:=my_params.yaml) or override a single value with --ros-args -p sigma:=0.15. For the gauntlet bag, play it without --clock and launch with use_sim_time:=false, because that bag has no /clock topic.

**Run the Unit Tests**
```bash
python3 -m pytest test/ -v
```

**Score a bag offline**
```bash
python3 scripts/replay.py bags/macfirst_floor_take_1 maps/mac_1st_floor_final.yaml -p n_particles=500
```

**Regenerate figures**
```bash
python3 scripts/make_figures.py
```
