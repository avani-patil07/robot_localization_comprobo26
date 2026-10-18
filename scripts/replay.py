#!/usr/bin/env python3
""" Offline replay: run our particle filter on a recorded bag without starting any ROS nodes.

    Reads /scan and the odom -> base_footprint transform straight from the bag file, builds the
    map's distance-to-nearest-obstacle field from the map .yaml/.pgm, and steps the real
    ParticleFilter methods (from robot_localization/pf.py) through the data in the same order as
    run_loop. Because nothing runs in real time and the random seed is fixed, every run with the
    same settings gives the same result, which makes it possible to compare parameter settings
    fairly and much faster than playing the bag back in ROS.

    Example:
        python3 scripts/replay.py bags/macfirst_floor_take_1 maps/mac_1st_floor_final.yaml -p n_particles=500
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

PKG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(PKG_DIR, 'robot_localization'))
import rclpy.logging  # noqa: E402
import pf  # noqa: E402
from helper_functions import TFHelper  # noqa: E402

# default tuning, matching config/pf_params.yaml (angles converted to radians)
DEFAULT_PARAMS = dict(n_particles=300, d_thresh=0.2, a_thresh=math.radians(30),
                      init_xy_sigma=0.3, init_theta_sigma=math.radians(20),
                      odom_xy_noise=0.02, odom_theta_noise=math.radians(3),
                      sigma=0.25, beam_step=10,
                      resample_xy_sigma=0.05, resample_theta_sigma=math.radians(5),
                      random_frac=0.05, cluster_radius=0.5)


class MapField:
    """ Same interface as OccupancyField (get_closest_obstacle_distance, get_obstacle_bounding_box),
        built directly from a map_server .yaml/.pgm pair instead of the map_server service. """

    def __init__(self, map_yaml):
        info = yaml.safe_load(open(map_yaml))
        img = np.array(Image.open(os.path.join(os.path.dirname(map_yaml), info['image'])).convert('L'))
        # map_server's trinary rule: occupancy probability = (255 - pixel) / 255 when negate is 0
        occ_prob = (255.0 - img) / 255.0 if not info.get('negate', 0) else img / 255.0
        occupied_img = occ_prob > info['occupied_thresh']
        # flip so row index = map y cell (the image's top row is the map's highest y)
        self.occupied = np.flipud(occupied_img).T          # indexed [x cell, y cell]
        self.resolution = info['resolution']
        self.origin_x, self.origin_y = info['origin'][:2]
        self.width, self.height = self.occupied.shape
        self.closest_occ = distance_transform_edt(~self.occupied) * self.resolution
        self.image = img                                    # kept for plotting
        self.extent = (self.origin_x, self.origin_x + self.width * self.resolution,
                       self.origin_y, self.origin_y + self.height * self.resolution)

    def get_obstacle_bounding_box(self):
        xs, ys = np.nonzero(self.occupied)
        r = self.resolution
        return ((xs.min() * r + self.origin_x, xs.max() * r + self.origin_x),
                (ys.min() * r + self.origin_y, ys.max() * r + self.origin_y))

    def get_closest_obstacle_distance(self, x, y):
        i = int((x - self.origin_x) / self.resolution)
        j = int((y - self.origin_y) / self.resolution)
        if 0 <= i < self.width and 0 <= j < self.height:
            return self.closest_occ[i, j]
        return float('nan')

    def distances(self, xs, ys):
        """ Vectorized get_closest_obstacle_distance, used for scoring. """
        i = ((xs - self.origin_x) / self.resolution).astype(int)
        j = ((ys - self.origin_y) / self.resolution).astype(int)
        ok = (i >= 0) & (i < self.width) & (j >= 0) & (j < self.height)
        out = np.full(xs.shape, np.nan)
        out[ok] = self.closest_occ[i[ok], j[ok]]
        return out


def yaw_from_quaternion(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def read_bag(bag_path):
    """ Read what the filter needs from a bag.
        Returns a dict with:
            scans: list of (t, ranges array, beam angles array in the base_footprint frame)
            odom: array of rows (t, x, y, yaw) for odom -> base_footprint
            map_to_odom: array of rows (t, x, y, yaw), only for bags recorded with a filter running
            clouds: list of (t, N x 3 array of particle x, y, theta) from /particle_cloud, if recorded
            poses: dict topic -> array of rows (t, x, y, yaw) for /pf_pose and /amcl_pose, if recorded
    """
    storage = 'mcap' if any(f.endswith('.mcap') for f in os.listdir(bag_path)) else 'sqlite3'
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}

    static_yaw = {}          # child frame -> (parent frame, yaw)
    odom, map_to_odom, raw_scans, clouds = [], [], [], []
    poses = {'/pf_pose': [], '/amcl_pose': []}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic not in ('/scan', '/tf', '/tf_static', '/particle_cloud', '/pf_pose', '/amcl_pose'):
            continue
        msg = deserialize_message(data, types[topic])
        if topic in poses:
            pose = msg.pose.pose if topic == '/amcl_pose' else msg.pose
            poses[topic].append((msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                                 pose.position.x, pose.position.y, yaw_from_quaternion(pose.orientation)))
        elif topic == '/scan':
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            raw_scans.append((t, msg.header.frame_id, np.array(msg.ranges), msg.angle_min, msg.angle_max))
        elif topic == '/particle_cloud':
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            clouds.append((t, np.array([(p.pose.position.x, p.pose.position.y,
                                         yaw_from_quaternion(p.pose.orientation)) for p in msg.particles])))
        else:
            for tf in msg.transforms:
                t = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
                row = (t, tf.transform.translation.x, tf.transform.translation.y,
                       yaw_from_quaternion(tf.transform.rotation))
                if topic == '/tf_static':
                    static_yaw[tf.child_frame_id] = (tf.header.frame_id, row[3])
                elif tf.header.frame_id == 'odom' and tf.child_frame_id == 'base_footprint':
                    odom.append(row)
                elif tf.header.frame_id == 'map' and tf.child_frame_id == 'odom':
                    map_to_odom.append(row)

    def yaw_to_base(frame):
        """ Sum the static yaws from frame up to base_footprint (what convert_scan_to_polar uses). """
        yaw = 0.0
        while frame != 'base_footprint' and frame in static_yaw:
            frame, y = static_yaw[frame]
            yaw += y
        return yaw

    scans = [(t, r, np.linspace(a_min + yaw_to_base(frame), a_max + yaw_to_base(frame), len(r)))
             for t, frame, r, a_min, a_max in raw_scans]
    odom = np.array(sorted(odom))
    return dict(scans=scans, odom=odom, map_to_odom=np.array(sorted(map_to_odom)), clouds=clouds,
                poses={k: np.array(sorted(v)) for k, v in poses.items()})


def odom_at(odom, t):
    """ Interpolate the odom -> base_footprint pose at time t, or None if t is outside the data. """
    if len(odom) == 0 or t < odom[0, 0] or t > odom[-1, 0]:
        return None
    k = np.searchsorted(odom[:, 0], t)
    if k == 0:
        return tuple(odom[0, 1:])
    t0, t1 = odom[k - 1, 0], odom[k, 0]
    a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    x = odom[k - 1, 1] + a * (odom[k, 1] - odom[k - 1, 1])
    y = odom[k - 1, 2] + a * (odom[k, 2] - odom[k - 1, 2])
    d_yaw = math.atan2(math.sin(odom[k, 3] - odom[k - 1, 3]), math.cos(odom[k, 3] - odom[k - 1, 3]))
    return (x, y, odom[k - 1, 3] + a * d_yaw)


def scan_score(field, pose, ranges, angles):
    """ Median distance (m) from the scan's endpoints, drawn from pose, to the nearest map obstacle.
        0 means the scan lies exactly on the map's walls; this is our accuracy measure. """
    x, y, th = pose
    ok = np.isfinite(ranges) & (ranges > 0.05)
    d = field.distances(x + ranges[ok] * np.cos(th + angles[ok]), y + ranges[ok] * np.sin(th + angles[ok]))
    return float(np.nanmedian(d)) if np.isfinite(d).any() else float('nan')


class _OfflineTF:
    """ The parts of TFHelper the filter uses, minus tf2 (the map -> odom broadcast isn't needed offline). """
    angle_normalize = TFHelper.angle_normalize
    angle_diff = TFHelper.angle_diff
    convert_pose_to_xy_and_theta = TFHelper.convert_pose_to_xy_and_theta

    def fix_map_to_odom_transform(self, robot_pose, odom_pose):
        pass


def make_offline_filter(field, params):
    """ A ParticleFilter object wired to the offline map and helpers instead of ROS. """
    f = pf.ParticleFilter.__new__(pf.ParticleFilter)
    f._logger = rclpy.logging.get_logger('replay')
    f.transform_helper = _OfflineTF()
    f.occupancy_field = field
    f.map_bounds = field.get_obstacle_bounding_box()
    f.map_frame = 'map'
    f.last_scan_timestamp = None
    f.particle_cloud = []
    f.current_odom_xy_theta = []
    for name, value in params.items():
        setattr(f, name, value)
    return f


def replay(bag, field, params=None, seed=0, initial_guess=None, keep_clouds=False):
    """ Run the filter over a bag (the result of read_bag).
        initial_guess: (x, y, theta) to initialize around, like an rviz 2D Pose Estimate click.
                       If None, the filter initializes from odometry, as pf.py does with no click.
        Returns a list of per-update dicts: t, estimate, odom, est_score, odom_score, update_ms
        (and cloud, if keep_clouds). Scores are scan_score in meters (lower is better). """
    np.random.seed(seed)
    f = make_offline_filter(field, dict(DEFAULT_PARAMS, **(params or {})))
    to_pose = lambda xyt: pf.Particle(*xyt).as_pose()   # noqa: E731
    records = []
    for t, ranges, angles in bag['scans']:
        odom_xyt = odom_at(bag['odom'], t)
        if odom_xyt is None:
            continue
        f.odom_pose = to_pose(odom_xyt)
        # same branching as ParticleFilter.run_loop
        if not f.current_odom_xy_theta:
            f.current_odom_xy_theta = odom_xyt
            continue
        if not f.particle_cloud:
            f.initialize_particle_cloud(None, initial_guess)
            continue
        if not f.moved_far_enough_to_update(odom_xyt):
            continue
        start = time.perf_counter()
        f.update_particles_with_odom()
        f.update_particles_with_laser(list(ranges), list(angles))
        f.update_robot_pose()
        estimate = f.transform_helper.convert_pose_to_xy_and_theta(f.robot_pose)
        cloud = np.array([(p.x, p.y, p.theta, p.w) for p in f.particle_cloud]) if keep_clouds else None
        f.resample_particles()
        update_ms = (time.perf_counter() - start) * 1000
        records.append(dict(t=t, estimate=estimate, odom=odom_xyt, update_ms=update_ms,
                            est_score=scan_score(field, estimate, ranges, angles),
                            odom_score=scan_score(field, odom_xyt, ranges, angles), cloud=cloud))
    return records


def summarize(records, settle_s=0.0):
    """ Summary numbers for a replay, ignoring the first settle_s seconds of updates. """
    if not records:
        return dict(updates=0)
    t0 = records[0]['t']
    rs = [r for r in records if r['t'] - t0 >= settle_s]
    est = np.array([r['est_score'] for r in rs])
    odo = np.array([r['odom_score'] for r in rs])
    return dict(updates=len(rs),
                est_median=float(np.nanmedian(est)), odom_median=float(np.nanmedian(odo)),
                est_frac_good=float(np.mean(est < 0.1)),
                update_ms=float(np.mean([r['update_ms'] for r in rs])))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bag')
    parser.add_argument('map_yaml')
    parser.add_argument('-p', '--param', action='append', default=[],
                        help='override a parameter, e.g. -p n_particles=500 (angles in radians)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    params = {}
    for kv in args.param:
        k, v = kv.split('=')
        params[k] = type(DEFAULT_PARAMS[k])(float(v)) if isinstance(DEFAULT_PARAMS[k], int) else float(v)
    records = replay(read_bag(args.bag), MapField(args.map_yaml), params, seed=args.seed)
    for r in records:
        print("t=%7.2f  estimate (%6.2f, %6.2f, %7.1f deg)  scan-to-wall %.3f m   odom-only %.3f m   %.1f ms"
              % (r['t'] - records[0]['t'], *r['estimate'][:2], math.degrees(r['estimate'][2]),
                 r['est_score'], r['odom_score'], r['update_ms']))
    print(summarize(records))


if __name__ == '__main__':
    main()
