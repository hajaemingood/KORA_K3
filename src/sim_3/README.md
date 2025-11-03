# sim_3 overview

`sim_3` is a lightweight Ackermann simulator tailored for the K3 platform. It keeps the same command interfaces as the physical car (`/commands/motor/speed`, `/commands/servo/position`, `/drive`, `/scan`) so the existing control stack can run unchanged.

## Nodes

- `minimal_sim.py`: integrates `/drive` commands, publishes `/sim3/odom`, `/sim3/pose`, `/scan`, and broadcasts `odom -> base_link` plus `base_link -> laser` TF. It subscribes to `/map` and performs simple ray casting to emulate the LiDAR.
- `cmd_bridge.py`: converts `/commands/motor/speed` and `/commands/servo/position` to `/drive` (`AckermannDriveStamped`), mirroring the vehicle command path.

## Launch example

```bash
source /opt/ros/noetic/setup.bash
source ~/KORA_K3/devel/setup.bash
roslaunch sim_3 minimal.launch \
  map_file:=/root/KORA_K3/src/kora_k3/maps/f8.yaml \
  initial_x:=-2.6643 initial_y:=-0.1435 initial_yaw:=-2.4617 \
  publish_map_tf:=false
```

## Key parameters

- `initial_x`, `initial_y`, `initial_yaw`: start pose in the map frame (yaw in radians).
- `scan_topic`, `scan_frame`: LiDAR topic/frame. Defaults `/scan`, `laser`.
- `publish_map_tf`: set to `false` when AMCL publishes the `map -> odom` transform.
- LiDAR model parameters (`scan_*`, `laser_offset_*`, `scan_noise`, `max_ray_length`) can be tuned to match the Hokuyo UST-10LX or other sensors.

## Quick checks

1. Verify simulator topics: `rostopic list | grep sim3`.
2. Send a direct `/drive` command for a smoke test:
   ```bash
   rostopic pub /drive ackermann_msgs/AckermannDriveStamped \
     '{header:{stamp:now,frame_id:"base_link"}, drive:{speed:1.0, steering_angle:0.2}}'
   ```
3. Or exercise the bridge:
   ```bash
   rostopic pub /commands/motor/speed std_msgs/Float64 "data: 4000" -r 5
   rostopic pub /commands/servo/position std_msgs/Float64 "data: 0.55" -r 5
   ```
4. In RViz, confirm `/scan` visualisation and the TF tree (`map`, `odom`, `base_link`, `laser`).
