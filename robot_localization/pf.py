#!/usr/bin/env python3

""" A particle filter that localizes a Neato in a known map using odometry and lidar.

    Each update: move every particle by the odometry change (motion model), weight it by how
    well the lidar scan lines up with the map from its pose (likelihood field), estimate the
    robot pose from the heaviest cluster of particles, then resample. The estimate is published
    as the map -> odom transform, and the cloud is published on particle_cloud for rviz. """

import rclpy
from threading import Thread
from rclpy.time import Time
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav2_msgs.msg import ParticleCloud
from nav2_msgs.msg import Particle as Nav2Particle
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Pose, Point, Quaternion
from visualization_msgs.msg import Marker
from rclpy.duration import Duration
import math
import time
import numpy as np
from occupancy_field import OccupancyField
from helper_functions import TFHelper, draw_random_sample
from rclpy.qos import qos_profile_sensor_data
from angle_helpers import quaternion_from_euler

class Particle(object):
    """ Represents a hypothesis (particle) of the robot's pose consisting of x,y and theta (yaw)
        Attributes:
            x: the x-coordinate of the hypothesis relative to the map frame
            y: the y-coordinate of the hypothesis relative to the map frame
            theta: the yaw of the hypothesis relative to the map frame
            w: the particle weight (the class does not ensure that particle weights are normalized)
    """

    def __init__(self, x=0.0, y=0.0, theta=0.0, w=1.0):
        """ Construct a new Particle
            x: the x-coordinate of the hypothesis relative to the map frame
            y: the y-coordinate of the hypothesis relative to the map frame
            theta: the yaw of the hypothesis relative to the map frame
            w: the particle weight (the class does not ensure that particle weights are normalized) """
        self.w = w
        self.theta = theta
        self.x = x
        self.y = y

    def as_pose(self):
        """ A helper function to convert a particle to a geometry_msgs/Pose message """
        q = quaternion_from_euler(0, 0, self.theta)
        return Pose(position=Point(x=self.x, y=self.y, z=0.0),
                    orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]))

class ParticleFilter(Node):
    """ The class that represents a Particle Filter ROS Node
        Attributes list:
            base_frame: the name of the robot base coordinate frame (should be "base_footprint" for most robots)
            map_frame: the name of the map coordinate frame (should be "map" in most cases)
            odom_frame: the name of the odometry coordinate frame (should be "odom" in most cases)
            scan_topic: the name of the scan topic to listen to (should be "scan" in most cases)
            n_particles: the number of particles in the filter
            d_thresh: the amount of linear movement before triggering a filter update
            a_thresh: the amount of angular movement before triggering a filter update
            (the remaining tuning knobs -- noise sigmas, sigma, beam_step, random_frac, cluster_radius --
             are ROS parameters; see the declare_parameter calls in __init__ for units and meaning)
            particle_pub: a publisher for the particle cloud
            pose_pub: a publisher for the current pose estimate (for rviz)
            scan_marker_pub: a publisher for the scan projected from the pose estimate (for rviz)
            map_bounds: ((x_min, x_max), (y_min, y_max)) of the map's obstacles, used for random particles
            last_scan_timestamp: this is used to keep track of the clock when using bags
            scan_to_process: the scan that our run_loop should process next
            occupancy_field: this helper class allows you to query the map for distance to closest obstacle
            transform_helper: this helps with various transform operations (abstracting away the tf2 module)
            particle_cloud: a list of particles representing a probability distribution over robot poses
            current_odom_xy_theta: the pose of the robot in the odometry frame when the last filter update was performed.
                                   The pose is expressed as a list [x,y,theta] (where theta is the yaw)
            thread: this thread runs your main loop
    """
    def __init__(self):
        super().__init__('pf')
        self.base_frame = "base_footprint"   # the frame of the robot base
        self.map_frame = "map"          # the name of the map coordinate frame
        self.odom_frame = "odom"        # the name of the odometry coordinate frame
        self.scan_topic = "scan"        # the topic where we will get laser scans from

        # Tuning knobs, exposed as ROS parameters so they can be changed at launch without editing code,
        # e.g. `ros2 run robot_localization pf.py --ros-args -p n_particles:=500 -p sigma:=0.15`.
        # Angles are given in degrees here for readability and converted to radians below.
        self.declare_parameter('n_particles', 300)            # number of particles in the cloud
        self.declare_parameter('d_thresh', 0.2)               # (m) linear movement before a filter update
        self.declare_parameter('a_thresh_deg', 30.0)          # (deg) turning before a filter update
        self.declare_parameter('init_xy_sigma', 0.3)          # (m) std dev of initial x/y spread around the guess
        self.declare_parameter('init_theta_sigma_deg', 20.0)  # (deg) std dev of initial heading spread
        self.declare_parameter('odom_xy_noise', 0.02)         # (m) std dev of per-particle motion noise
        self.declare_parameter('odom_theta_noise_deg', 3.0)   # (deg) std dev of per-particle turn noise
        self.declare_parameter('sigma', 0.25)                 # (m) std dev of the laser likelihood Gaussian
        self.declare_parameter('beam_step', 10)               # use every beam_step-th lidar beam for weighting
        self.declare_parameter('resample_xy_sigma', 0.05)     # (m) jitter added to resampled particles
        self.declare_parameter('resample_theta_sigma_deg', 5.0)  # (deg) heading jitter added when resampling
        self.declare_parameter('random_frac', 0.05)           # fraction of particles re-drawn uniformly each resample
        self.declare_parameter('cluster_radius', 0.5)         # (m) particles this close to the best one form the estimate

        self.n_particles = self.get_parameter('n_particles').value
        self.d_thresh = self.get_parameter('d_thresh').value
        self.a_thresh = math.radians(self.get_parameter('a_thresh_deg').value)
        self.init_xy_sigma = self.get_parameter('init_xy_sigma').value
        self.init_theta_sigma = math.radians(self.get_parameter('init_theta_sigma_deg').value)
        self.odom_xy_noise = self.get_parameter('odom_xy_noise').value
        self.odom_theta_noise = math.radians(self.get_parameter('odom_theta_noise_deg').value)
        self.sigma = self.get_parameter('sigma').value
        self.beam_step = self.get_parameter('beam_step').value
        self.resample_xy_sigma = self.get_parameter('resample_xy_sigma').value
        self.resample_theta_sigma = math.radians(self.get_parameter('resample_theta_sigma_deg').value)
        self.random_frac = self.get_parameter('random_frac').value
        self.cluster_radius = self.get_parameter('cluster_radius').value

        # pose_listener responds to selection of a new approximate robot location (for instance using rviz)
        self.create_subscription(PoseWithCovarianceStamped, 'initialpose', self.update_initial_pose, 10)

        # publish the current particle cloud.  This enables viewing particles in rviz.
        self.particle_pub = self.create_publisher(ParticleCloud, "particle_cloud", qos_profile_sensor_data)

        # debug visualization: the pose estimate, and the lidar scan drawn from that pose. If the
        # estimate is right, the projected scan points land on the map's walls in rviz.
        self.pose_pub = self.create_publisher(PoseStamped, "pf_pose", 10)
        self.scan_marker_pub = self.create_publisher(Marker, "pf_scan_projection", 10)

        # laser_subscriber listens for data from the lidar
        self.create_subscription(LaserScan, self.scan_topic, self.scan_received, 10)

        # this is used to keep track of the timestamps coming from bag files
        # knowing this information helps us set the timestamp of our map -> odom
        # transform correctly
        self.last_scan_timestamp = None
        # this is the current scan that our run_loop should process
        self.scan_to_process = None
        # the particle cloud: a list of Particle objects
        self.particle_cloud = []

        self.current_odom_xy_theta = []
        self.occupancy_field = OccupancyField(self)
        self.map_bounds = self.occupancy_field.get_obstacle_bounding_box()  # ((x_min, x_max), (y_min, y_max)) of the map's obstacles
        self.transform_helper = TFHelper(self)

        # we are using a thread to work around single threaded execution bottleneck.
        # daemon=True lets the process exit on Ctrl+C; otherwise this endless loop keeps it alive.
        thread = Thread(target=self.loop_wrapper, daemon=True)
        thread.start()
        self.transform_update_timer = self.create_timer(0.05, self.pub_latest_transform)

    def pub_latest_transform(self):
        """ This function takes care of sending out the map to odom transform """
        if self.last_scan_timestamp is None:
            return
        postdated_timestamp = Time.from_msg(self.last_scan_timestamp) + Duration(seconds=0.1)
        self.transform_helper.send_last_map_to_odom_transform(self.map_frame, self.odom_frame, postdated_timestamp)

    def loop_wrapper(self):
        """ This function takes care of calling the run_loop function repeatedly.
            We are using a separate thread to run the loop_wrapper to work around
            issues with single threaded executors in ROS2 """
        while True:
            self.run_loop()
            time.sleep(0.1)

    def run_loop(self):
        """ This is the main run_loop of our particle filter.  It checks to see if
            any scans are ready and to be processed and will call several helper
            functions to complete the processing.
            
            You do not need to modify this function, but it is helpful to understand it.
        """
        if self.scan_to_process is None:
            return
        msg = self.scan_to_process

        (new_pose, delta_t) = self.transform_helper.get_matching_odom_pose(self.odom_frame,
                                                                           self.base_frame,
                                                                           msg.header.stamp)
        if new_pose is None:
            # we were unable to get the pose of the robot corresponding to the scan timestamp
            if delta_t is not None and delta_t < Duration(seconds=0.0):
                # we will never get this transform, since it is before our oldest one
                self.scan_to_process = None
            return
        
        (r, theta) = self.transform_helper.convert_scan_to_polar_in_robot_frame(msg, self.base_frame)
        # clear the current scan so that we can process the next one
        self.scan_to_process = None

        self.odom_pose = new_pose
        new_odom_xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(self.odom_pose)
        self.get_logger().debug("odom x: {0:.3f}, y: {1:.3f}, yaw: {2:.3f}".format(*new_odom_xy_theta))

        if not self.current_odom_xy_theta:
            self.current_odom_xy_theta = new_odom_xy_theta
        elif not self.particle_cloud:
            # now that we have all of the necessary transforms we can update the particle cloud
            self.initialize_particle_cloud(msg.header.stamp)
        elif self.moved_far_enough_to_update(new_odom_xy_theta):
            # we have moved far enough to do an update!
            self.update_particles_with_odom()    # update based on odometry
            self.update_particles_with_laser(r, theta)   # update based on laser scan
            self.update_robot_pose()                # update robot's pose based on particles
            self.publish_scan_projection(r, theta, msg.header.stamp)  # draw the scan from the estimate
            self.resample_particles()               # resample particles to focus on areas of high density
        # publish particles (so things like rviz can see them)
        self.publish_particles(msg.header.stamp)

    def moved_far_enough_to_update(self, new_odom_xy_theta):
        """ Return True if the robot has moved more than d_thresh in x or y, or turned more than
            a_thresh, since the last filter update. The turn uses angle_diff so that crossing
            the +-pi wraparound (e.g. 179 deg -> -179 deg) counts as a 2 deg turn, not 358 deg. """
        d_theta = self.transform_helper.angle_diff(new_odom_xy_theta[2], self.current_odom_xy_theta[2])
        return math.fabs(new_odom_xy_theta[0] - self.current_odom_xy_theta[0]) > self.d_thresh or \
               math.fabs(new_odom_xy_theta[1] - self.current_odom_xy_theta[1]) > self.d_thresh or \
               math.fabs(d_theta) > self.a_thresh

    def update_robot_pose(self):
        """ Update the estimate of the robot's pose given the updated particles.
            We use a hybrid of the mean and the mode: find the highest-weight particle, then take
            the weighted mean of only the particles within cluster_radius of it. If the cloud is
            one blob this equals the weighted mean; if it has split into separate groups, only
            the strongest group counts, so the estimate can't land between groups (or in a wall).
            The estimate is stored in self.robot_pose and used to update the map -> odom transform.
        """
        if not self.particle_cloud:
            return
        # first make sure that the particle weights are normalized
        self.normalize_particles()

        # find the single best particle (the peak of the distribution)
        best = max(self.particle_cloud, key=lambda p: p.w)

        # keep only particles near the best one, so a split cloud doesn't average
        # two clusters into a pose between them (possibly inside a wall)
        cluster = [p for p in self.particle_cloud
                   if math.hypot(p.x - best.x, p.y - best.y) < self.cluster_radius]
        cluster_w = sum(p.w for p in cluster)

        # weighted mean of x and y over the cluster (divide by cluster_w since
        # the cluster's weights alone don't sum to 1)
        x_mean = sum(p.w * p.x for p in cluster) / cluster_w
        y_mean = sum(p.w * p.y for p in cluster) / cluster_w

        # circular mean of theta: average the unit vectors then take their angle
        # (a plain average fails at the wraparound: 179 deg and -179 deg would average to 0 deg)
        sin_sum = sum(p.w * math.sin(p.theta) for p in cluster)
        cos_sum = sum(p.w * math.cos(p.theta) for p in cluster)
        theta_mean = math.atan2(sin_sum, cos_sum)

        # reuse Particle.as_pose() to build the geometry_msgs/Pose (handles the quaternion)
        self.robot_pose = Particle(x=x_mean, y=y_mean, theta=theta_mean).as_pose()

        # passes robot_pose to fix_map_to_odom_transform
        if hasattr(self, 'odom_pose'):
            self.transform_helper.fix_map_to_odom_transform(self.robot_pose,
                                                            self.odom_pose)
        else:
            self.get_logger().warn("Can't set map->odom transform since no odom data received")

        if self.last_scan_timestamp is not None:
            pose_msg = PoseStamped(pose=self.robot_pose)
            pose_msg.header.frame_id = self.map_frame
            pose_msg.header.stamp = self.last_scan_timestamp
            self.pose_pub.publish(pose_msg)

    def update_particles_with_odom(self):
        """ Motion update: move every particle by the change in odometry since the last update.
            Odometry is only trustworthy relative to the robot itself, so the odom-frame change
            is first rotated into the robot's body frame (forward, left, turn), then replayed from
            each particle's own heading, with a little Gaussian noise added per particle.
        """
        new_odom_xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(self.odom_pose)
        # compute the change in x,y,theta since our last update
        if self.current_odom_xy_theta:
            old_odom_xy_theta = self.current_odom_xy_theta
            x1, y1, theta1 = old_odom_xy_theta
            x2, y2, theta2 = new_odom_xy_theta

            # odom-frame displacement -> robot body frame (rotate by -theta1)
            dx_odom = x2 - x1
            dy_odom = y2 - y1
            forward = math.cos(theta1) * dx_odom + math.sin(theta1) * dy_odom
            left = -math.sin(theta1) * dx_odom + math.cos(theta1) * dy_odom
            d_theta = self.transform_helper.angle_diff(theta2, theta1)

            self.current_odom_xy_theta = new_odom_xy_theta
        else:
            self.current_odom_xy_theta = new_odom_xy_theta
            return

        for p in self.particle_cloud:
            # each particle gets a slightly different noisy version of the motion
            f = forward + np.random.normal(0, self.odom_xy_noise)
            l = left + np.random.normal(0, self.odom_xy_noise)
            dt = d_theta + np.random.normal(0, self.odom_theta_noise)

            # replay the body-frame motion from this particle's own heading
            p.x += f * math.cos(p.theta) - l * math.sin(p.theta)
            p.y += f * math.sin(p.theta) + l * math.cos(p.theta)
            p.theta = self.transform_helper.angle_normalize(p.theta + dt)

    def make_random_particles(self):
        """ Return one particle at a uniformly random free-space location in the map, with a
            uniformly random heading. Used to re-seed part of the cloud on every resample so the
            filter can recover if it converges to the wrong place (avoids particle depletion). """
        (x_min, x_max), (y_min, y_max) = self.map_bounds
        while True:
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            dist = self.occupancy_field.get_closest_obstacle_distance(x, y)

            if not math.isnan(dist) and dist > 0.0:
                theta = np.random.uniform(-math.pi, math.pi)
                return Particle(x=x, y=y, theta=theta, w=1.0/self.n_particles)

    def resample_particles(self):
        """ Resample the particles according to the new particle weights.
            (1 - random_frac) of the new cloud is drawn with replacement, with probability equal
            to each particle's weight, and jittered slightly so duplicates spread back out. The
            remaining random_frac is drawn uniformly over the map to guard against particle
            depletion. All weights are reset to 1/n_particles.
        """
        # make sure the distribution is normalized
        self.normalize_particles()

        n_random = int(self.random_frac * self.n_particles)
        n_sampled = self.n_particles - n_random

        weights = [p.w for p in self.particle_cloud]
        new_cloud = draw_random_sample(self.particle_cloud, weights, n_sampled)

        for p in new_cloud:
            p.x += np.random.normal(0, self.resample_xy_sigma)
            p.y += np.random.normal(0, self.resample_xy_sigma)
            p.theta = self.transform_helper.angle_normalize(p.theta + np.random.normal(0.0, self.resample_theta_sigma))
            p.w = 1.0 / self.n_particles  # reset weights to uniform distribution

        for _ in range(n_random):
            new_cloud.append(self.make_random_particles())

        self.particle_cloud = new_cloud

    def update_particles_with_laser(self, r, theta):
        """ Observation update: weight each particle by how well the scan fits the map from its pose.
            Uses the likelihood field model. For every beam_step-th valid beam, the beam's endpoint
            is projected into the map from the particle's pose, and d is the distance from that
            endpoint to the nearest obstacle. Each beam scores exp(-d^2 / 2 sigma^2); the particle's
            weight is the average beam score, cubed to sharpen the gap between good and bad poses.
            (Averaging instead of multiplying avoids underflow and tolerates a few bad beams.)
            r: the distance readings to obstacles
            theta: the angle relative to the robot frame for each corresponding reading
        """
        # keep every beam_step-th valid beam
        beams = [(dist, angle) for dist, angle in zip(r, theta)
                 if not (math.isinf(dist) or math.isnan(dist) or dist <= 0.0)]
        beams = beams[::self.beam_step]
        if not beams:  # does nothing if no valid readings
            return

        for particle in self.particle_cloud:
            total = 0.0
            for dist, angle in beams:
                # project this beam's endpoint into the map from the particle's pose/heading
                angle_map = particle.theta + angle
                x = particle.x + dist * math.cos(angle_map)
                y = particle.y + dist * math.sin(angle_map)

                obstacle_dist = self.occupancy_field.get_closest_obstacle_distance(x, y)
                if math.isnan(obstacle_dist):
                    obstacle_dist = self.sigma * 7  # penalize endpoints that land off the map

                # gaussian likelihood of this one beam
                total += math.exp(-(obstacle_dist ** 2) / (2 * self.sigma ** 2))

            # average over beams then cube to sharpen good vs bad particles
            particle.w = (total / len(beams)) ** 3

    def update_initial_pose(self, msg):
        """ Callback function to handle re-initializing the particle filter based on a pose estimate.
            These pose estimates could be generated by another ROS Node or could come from the rviz GUI """
        xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(msg.pose.pose)
        self.initialize_particle_cloud(msg.header.stamp, xy_theta)

    def initialize_particle_cloud(self, timestamp, xy_theta=None):
        """ Initialize the particle cloud.
            Arguments
            xy_theta: a triple consisting of the mean x, y, and theta (yaw) to initialize the
                      particle cloud around.  If this input is omitted, the odometry will be used """
        if xy_theta is None:
            xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(self.odom_pose)
        self.particle_cloud = []
        # create particles
        x0, y0, theta0 = xy_theta  # unpacks the guess
        for _ in range(self.n_particles):
            # draws one random sample from a gaussian
            x = np.random.normal(x0, self.init_xy_sigma)
            y = np.random.normal(y0, self.init_xy_sigma)
            theta = np.random.normal(theta0, self.init_theta_sigma)
            theta = self.transform_helper.angle_normalize(theta)
            self.particle_cloud.append(Particle(x=x, y=y, theta=theta, w=1.0))  # gives every particle same weight

        self.normalize_particles()
        self.update_robot_pose()

    def normalize_particles(self):
        """ Make sure the particle weights define a valid distribution (i.e. sum to 1.0) """
        if not self.particle_cloud:
            return
        total_weight = sum(p.w for p in self.particle_cloud)  # adds up raw weight

        if total_weight == 0.0:
            # if every particle got a weight of 0 (e.g. sigma is too small), give each particle
            # the same weight to keep the node running
            normalized_weight = 1.0 / len(self.particle_cloud)
            for p in self.particle_cloud:
                p.w = normalized_weight
            return

        # normal case: divide each particle's weight by the sum so they total 1.0
        for p in self.particle_cloud:
            p.w /= total_weight

    def publish_scan_projection(self, r, theta, timestamp):
        """ Publish the lidar scan drawn from the current pose estimate as a POINTS marker in the
            map frame. If the estimate is right, these points sit on the map's walls in rviz. """
        if not hasattr(self, 'robot_pose'):
            return
        x0, y0, theta0 = self.transform_helper.convert_pose_to_xy_and_theta(self.robot_pose)
        marker = Marker(type=Marker.POINTS, action=Marker.ADD)
        marker.header.frame_id = self.map_frame
        marker.header.stamp = timestamp
        marker.ns = "pf_scan_projection"
        marker.scale.x = marker.scale.y = 0.05
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.2, 0.2, 1.0
        for dist, angle in zip(r, theta):
            if math.isinf(dist) or math.isnan(dist) or dist <= 0.0:
                continue
            marker.points.append(Point(x=x0 + dist * math.cos(theta0 + angle),
                                       y=y0 + dist * math.sin(theta0 + angle), z=0.0))
        self.scan_marker_pub.publish(marker)

    def publish_particles(self, timestamp):
        """ Publish every particle's pose and weight as a nav2 ParticleCloud in the map frame. """
        msg = ParticleCloud()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = timestamp
        for p in self.particle_cloud:
            msg.particles.append(Nav2Particle(pose=p.as_pose(), weight=p.w))
        self.particle_pub.publish(msg)

    def scan_received(self, msg):
        """ Lidar callback: remember the scan's timestamp, and keep the scan for run_loop
            unless a previous scan is still being processed. """
        self.last_scan_timestamp = msg.header.stamp
        # we throw away scans until we are done processing the previous scan
        # self.scan_to_process is set to None in the run_loop
        if self.scan_to_process is None:
            self.scan_to_process = msg

def main(args=None):
    """ Start the particle filter node and spin until shutdown. """
    rclpy.init()
    n = ParticleFilter()
    rclpy.spin(n)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
