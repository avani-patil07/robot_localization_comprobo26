# Bag files

Input bags are sensor data. Result bags are recordings of a filter running on an input bag.

Score: we judge a pose estimate by drawing the lidar scan from that pose and measuring how far the scan points land from the map's walls (median, in meters). Near 0 m means the pose is right. "Odometry only" is the same score using the robot's raw wheel odometry, which is the baseline to beat.

## Input bags

| Bag | How it was collected | Map |
|---|---|---|
| `macfirst_floor_take_1` | Provided with the assignment. A real Neato driven by hand for 93 s on the first floor of the Miller Academic Center (MAC). | `mac_1st_floor_final.yaml` |
| `macfirst_floor_take_2` | Provided with the assignment. A 46 s drive on the same floor; the robot sits still for the first 10 s. | `mac_1st_floor_final.yaml` |
| `gauntlet_run_1` | Recorded by our team in the Gazebo gauntlet simulation, 132 s of teleoperated driving. | `gauntlet.yaml` |

## Result bags

| Bag | What ran | What it shows |
|---|---|---|
| `pf_run_take_1_2026-10-18` | our filter on `take_1` | Tracks the whole drive. Scan stays about 0.03 m from the walls; odometry only drifts to 0.3–0.6 m. |
| `pf_run_take_2_2026-10-18` | our filter on `take_2` | Same: 0.00–0.06 m vs 0.28–0.91 m for odometry only. Correctly waits while the robot is still. |
| `pf_run_gauntlet_2026-10-18` | our filter on `gauntlet_run_1` | Median 0.14 m. Rough, because the map doesn't match the world (see below). |
| `amcl_run_gauntlet_2026-10-18` | nav2 AMCL on `gauntlet_run_1` | Median 0.25 m on the same data, for comparison. |

Each result bag was made by launching the filter, running `ros2 bag record`, and playing the input bag. They contain `/scan`, `/odom`, `/tf` (including the filter's `map → odom` estimate), `/map`, and the filter's `/particle_cloud`. The gauntlet runs also have `/pf_pose` or `/amcl_pose`.

## Known issues

- The starter `gauntlet.yaml` doesn't match the world we recorded in; odometry drives through an obstacle on it. We tried building our own map from the bag with `slam_toolbox`, but the walls came out doubled, so gauntlet numbers are rough.
- Gauntlet bag has no `/clock`. Play it without `--clock` and launch with `use_sim_time:=false`.
- AMCL fails on the MAC bags. It stopped after 9 updates ("Couldn't determine robot's pose associated with laser scan"). In these bags each scan arrives slightly before the matching odometry transform, AMCL drops those scans, while our filter waits and retries.

## Replaying

Watch a recorded run, with nothing else running:

```bash
ros2 bag play bags/pf_run_take_1_2026-10-18 --clock
rviz2 -d rviz/pf_demo.rviz
```

Run the filter yourself on an input bag:

```bash
ros2 launch robot_localization test_pf.py map_yaml:=$PWD/maps/mac_1st_floor_final.yaml
ros2 bag play bags/macfirst_floor_take_1 --clock      # second terminal
```

Score a bag in a few seconds without ROS:

```bash
python3 scripts/replay.py bags/macfirst_floor_take_1 maps/mac_1st_floor_final.yaml
```
