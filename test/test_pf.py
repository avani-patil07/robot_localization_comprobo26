""" Unit tests for each step of the particle filter, with known answers.

    These call the real ParticleFilter methods without starting ROS: the node is created with
    __new__ (skipping __init__, which needs a running map server), and the map is a fake
    occupancy field for a 4 m x 4 m square room. Run from the package root with:

        python3 -m pytest test/ -v
"""

import math
import os
import sys

import numpy as np
import pytest
import rclpy.logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'robot_localization'))
import pf  # noqa: E402
from helper_functions import TFHelper  # noqa: E402


ROOM_HALF_WIDTH = 2.0   # walls at x = +-2 and y = +-2


class FakeTFHelper:
    """ The pure-math parts of TFHelper, without the tf2 listener that needs a ROS node. """
    angle_normalize = TFHelper.angle_normalize
    angle_diff = TFHelper.angle_diff
    convert_pose_to_xy_and_theta = TFHelper.convert_pose_to_xy_and_theta


class FakeOccupancyField:
    """ Distance to the nearest wall of a square room centered on the origin (nan outside it). """

    def get_closest_obstacle_distance(self, x, y):
        if abs(x) > ROOM_HALF_WIDTH or abs(y) > ROOM_HALF_WIDTH:
            return float('nan')
        return min(ROOM_HALF_WIDTH - abs(x), ROOM_HALF_WIDTH - abs(y))


def make_filter(**overrides):
    """ Build a ParticleFilter with default tuning and fake helpers, without ROS. """
    f = pf.ParticleFilter.__new__(pf.ParticleFilter)
    f.transform_helper = FakeTFHelper()
    f.occupancy_field = FakeOccupancyField()
    f.map_bounds = ((-ROOM_HALF_WIDTH, ROOM_HALF_WIDTH), (-ROOM_HALF_WIDTH, ROOM_HALF_WIDTH))
    f.map_frame = "map"
    f._logger = rclpy.logging.get_logger('test_pf')   # Node.get_logger() reads this; normally set by Node.__init__
    f.last_scan_timestamp = None       # so update_robot_pose doesn't try to publish
    f.particle_cloud = []
    f.current_odom_xy_theta = []
    defaults = dict(n_particles=300, d_thresh=0.2, a_thresh=math.radians(30),
                    init_xy_sigma=0.3, init_theta_sigma=math.radians(20),
                    odom_xy_noise=0.0, odom_theta_noise=0.0,
                    sigma=0.25, beam_step=1,
                    resample_xy_sigma=0.0, resample_theta_sigma=0.0, random_frac=0.0,
                    cluster_radius=0.5)
    defaults.update(overrides)
    for name, value in defaults.items():
        setattr(f, name, value)
    return f


def simulate_scan(x, y, theta, n_beams=36):
    """ Ray-cast a perfect lidar scan inside the square room from pose (x, y, theta).
        Returns (ranges, beam angles relative to the robot), like convert_scan_to_polar_in_robot_frame. """
    angles = np.linspace(0, 2 * math.pi, n_beams, endpoint=False)
    ranges = []
    for a in angles:
        dx, dy = math.cos(theta + a), math.sin(theta + a)
        # distance along the ray to each of the four walls; keep the closest one in front of us
        hits = [(ROOM_HALF_WIDTH * s - x) / dx for s in (-1, 1) if abs(dx) > 1e-9] + \
               [(ROOM_HALF_WIDTH * s - y) / dy for s in (-1, 1) if abs(dy) > 1e-9]
        ranges.append(min(h for h in hits if h > 0))
    return ranges, list(angles)


# initialization

def test_init_particles_have_right_mean_and_spread():
    """ Particles drawn around a guess are centered on it with the requested spread. """
    np.random.seed(0)
    f = make_filter(n_particles=5000)
    f.update_robot_pose = lambda: None          # only testing the cloud itself here
    f.initialize_particle_cloud(timestamp=None, xy_theta=(1.0, -0.5, math.radians(45)))

    xs = np.array([p.x for p in f.particle_cloud])
    ys = np.array([p.y for p in f.particle_cloud])
    thetas = np.array([p.theta for p in f.particle_cloud])
    assert len(f.particle_cloud) == 5000
    assert xs.mean() == pytest.approx(1.0, abs=0.02)
    assert ys.mean() == pytest.approx(-0.5, abs=0.02)
    assert xs.std() == pytest.approx(0.3, rel=0.05)
    assert math.degrees(math.atan2(np.sin(thetas).mean(), np.cos(thetas).mean())) == pytest.approx(45, abs=1)
    assert sum(p.w for p in f.particle_cloud) == pytest.approx(1.0)


# motion update

def run_motion(old_odom, new_odom, particle, **overrides):
    """ Apply one motion update for odometry old_odom -> new_odom to a single particle. """
    f = make_filter(**overrides)
    f.particle_cloud = [particle]
    f.current_odom_xy_theta = old_odom
    # odom_pose is already an (x, y, theta) tuple here, so make the conversion a pass-through
    f.transform_helper.convert_pose_to_xy_and_theta = lambda pose: pose
    f.odom_pose = new_odom
    f.update_particles_with_odom()
    return particle


def test_motion_drive_forward_along_particle_heading():
    """ A particle facing 90 deg that is told 'drive 1 m forward' ends up 1 m along its own heading,
        even though the robot was facing 0 deg in the odom frame. """
    p = run_motion((0, 0, 0), (1, 0, 0), pf.Particle(0, 0, math.pi / 2))
    assert (p.x, p.y) == pytest.approx((0, 1), abs=1e-9)
    assert p.theta == pytest.approx(math.pi / 2)


def test_motion_odom_heading_differs_from_particle():
    """ Robot faces 90 deg in odom and drives 1 m; a particle facing 0 deg should move +x. """
    p = run_motion((0, 0, math.pi / 2), (0, 1, math.pi / 2), pf.Particle(0, 0, 0))
    assert (p.x, p.y) == pytest.approx((1, 0), abs=1e-9)


def test_motion_turn_across_wraparound():
    """ Odometry going 179 deg -> -179 deg is a +2 deg turn, not -358 deg. """
    p = run_motion((0, 0, math.radians(179)), (0, 0, math.radians(-179)), pf.Particle(0, 0, 0))
    assert math.degrees(p.theta) == pytest.approx(2.0)


def test_motion_turn_in_place_does_not_translate():
    p = run_motion((2, 3, 0.3), (2, 3, 0.3 + math.pi / 2), pf.Particle(5, 5, 0))
    assert (p.x, p.y) == pytest.approx((5, 5), abs=1e-9)
    assert p.theta == pytest.approx(math.pi / 2)


def test_motion_noise_spreads_identical_particles():
    """ With motion noise on, particles that start identical end up at different places. """
    np.random.seed(1)
    f = make_filter(odom_xy_noise=0.05, odom_theta_noise=math.radians(3))
    f.particle_cloud = [pf.Particle(0, 0, 0) for _ in range(200)]
    f.current_odom_xy_theta = (0, 0, 0)
    f.transform_helper.convert_pose_to_xy_and_theta = lambda pose: pose
    f.odom_pose = (1, 0, 0)
    f.update_particles_with_odom()
    xs = np.array([p.x for p in f.particle_cloud])
    assert xs.mean() == pytest.approx(1.0, abs=0.02)
    assert xs.std() > 0.02


def test_moved_far_enough_ignores_wraparound():
    """ 179 deg -> -179 deg is only 2 deg of turning, so it should not trigger an update. """
    f = make_filter()
    f.current_odom_xy_theta = (0, 0, math.radians(179))
    assert not f.moved_far_enough_to_update((0, 0, math.radians(-179)))
    assert f.moved_far_enough_to_update((0, 0, math.radians(179 - 45)))
    assert f.moved_far_enough_to_update((0.3, 0, math.radians(179)))


# laser (observation) update

@pytest.mark.parametrize("offset", [0.1, 0.2, 0.3])
def test_laser_true_pose_scores_higher_than_offset_poses(offset):
    """ A particle at the true pose outweighs particles shifted 10-30 cm or rotated. """
    true_pose = (0.5, -0.3, math.radians(20))
    r, theta = simulate_scan(*true_pose)
    f = make_filter()
    x, y, th = true_pose
    f.particle_cloud = [pf.Particle(x, y, th),
                        pf.Particle(x + offset, y, th),
                        pf.Particle(x, y - offset, th),
                        pf.Particle(x, y, th + offset)]      # offset rad, about 6-17 deg
    f.update_particles_with_laser(r, theta)
    weights = [p.w for p in f.particle_cloud]
    assert weights[0] == pytest.approx(1.0)                  # every beam lands exactly on a wall
    assert all(weights[0] > w for w in weights[1:])


def test_laser_skips_invalid_beams():
    """ inf / nan / zero ranges are ignored instead of crashing or zeroing the weight. """
    r, theta = simulate_scan(0, 0, 0)
    r[0], r[1], r[2] = float('inf'), float('nan'), 0.0
    f = make_filter()
    f.particle_cloud = [pf.Particle(0, 0, 0)]
    f.update_particles_with_laser(r, theta)
    assert f.particle_cloud[0].w == pytest.approx(1.0)


# pose estimate

def test_circular_mean_of_179_and_minus_179_is_180():
    f = make_filter()
    f.particle_cloud = [pf.Particle(0, 0, math.radians(179), 0.5),
                        pf.Particle(0, 0, math.radians(-179), 0.5)]
    f.update_robot_pose()
    _, _, theta = TFHelper.convert_pose_to_xy_and_theta(None, f.robot_pose)
    assert abs(math.degrees(theta)) == pytest.approx(180.0)


def test_pose_is_weighted_mean_for_single_cluster():
    f = make_filter()
    f.particle_cloud = [pf.Particle(0.0, 0, 0, 0.75), pf.Particle(0.2, 0, 0, 0.25)]
    f.update_robot_pose()
    assert f.robot_pose.position.x == pytest.approx(0.05)


def test_pose_uses_heaviest_cluster_when_cloud_splits():
    """ Two groups 3 m apart: the estimate is the heavier group, not the midpoint between them. """
    f = make_filter()
    f.particle_cloud = [pf.Particle(-1.5, 0, 0, 0.2) for _ in range(3)] + \
                       [pf.Particle(1.5, 0, 0, 0.1) for _ in range(4)]
    f.update_robot_pose()
    assert f.robot_pose.position.x == pytest.approx(-1.5)


# normalize + resample

def test_normalize_sums_to_one_and_handles_all_zero():
    f = make_filter()
    f.particle_cloud = [pf.Particle(w=2.0), pf.Particle(w=6.0)]
    f.normalize_particles()
    assert [p.w for p in f.particle_cloud] == pytest.approx([0.25, 0.75])
    f.particle_cloud = [pf.Particle(w=0.0) for _ in range(4)]
    f.normalize_particles()
    assert [p.w for p in f.particle_cloud] == pytest.approx([0.25] * 4)


def test_resample_picks_particles_in_proportion_to_weight():
    """ With weights [0.7, 0.1, 0.1, 0.1], the first particle is picked about 70% of the time. """
    np.random.seed(2)
    f = make_filter(n_particles=10000)
    f.particle_cloud = [pf.Particle(x=i, w=w) for i, w in enumerate([0.7, 0.1, 0.1, 0.1])]
    f.resample_particles()
    xs = np.array([p.x for p in f.particle_cloud])
    assert len(f.particle_cloud) == 10000
    assert np.mean(xs == 0) == pytest.approx(0.7, abs=0.02)
    assert all(p.w == pytest.approx(1 / 10000) for p in f.particle_cloud)


def test_resample_copies_are_independent():
    """ Resampled duplicates must be separate objects, or jittering one would move the others. """
    f = make_filter(n_particles=10)
    f.particle_cloud = [pf.Particle(0, 0, 0, 1.0)]
    f.resample_particles()
    f.particle_cloud[0].x = 99
    assert f.particle_cloud[1].x == 0


def test_resample_injects_random_particles_inside_map():
    np.random.seed(3)
    f = make_filter(n_particles=100, random_frac=0.1)
    f.particle_cloud = [pf.Particle(0, 0, 0, 1.0)]
    f.resample_particles()
    moved = [p for p in f.particle_cloud if (p.x, p.y) != (0, 0)]
    assert len(f.particle_cloud) == 100
    assert len(moved) == 10
    assert all(abs(p.x) < ROOM_HALF_WIDTH and abs(p.y) < ROOM_HALF_WIDTH for p in moved)
