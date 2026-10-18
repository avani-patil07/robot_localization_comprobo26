#!/usr/bin/env python3
""" Run the evaluation experiments and draw every figure used in the README.

    Uses the offline replay (scripts/replay.py) for the parameter sweeps, so each setting is run
    on identical data with fixed random seeds, and reads the live-recorded bags in bags/ for the
    AMCL comparison. Writes PNG/GIF files to docs/images/ and prints the numbers quoted in the
    README. Takes a few minutes. Run from the package root:

        python3 scripts/make_figures.py
"""

import math
import os
import sys
from multiprocessing import Pool

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay import MapField, read_bag, replay, scan_score, summarize  # noqa: E402

PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT = os.path.join(PKG, 'docs', 'images')
SEEDS = [0, 1, 2]

# reference palette (light mode): categorical slots in fixed order, plus text/surface tokens
FILTER, ODOM, AMCL = '#2a78d6', '#eb6834', '#1baf7a'
SURFACE, INK, INK_2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
                     'axes.edgecolor': GRID, 'axes.labelcolor': INK_2, 'xtick.color': INK_2,
                     'ytick.color': INK_2, 'text.color': INK, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'lines.linewidth': 2, 'font.size': 10, 'axes.titlesize': 11, 'legend.frameon': False})

RUNS = {  # name -> (bag, map)
    'take_1': ('bags/macfirst_floor_take_1', 'maps/mac_1st_floor_final.yaml'),
    'take_2': ('bags/macfirst_floor_take_2', 'maps/mac_1st_floor_final.yaml'),
    'gauntlet': ('bags/gauntlet_run_1', 'maps/gauntlet.yaml'),
}
DATA = {}   # filled before the worker pool forks, so workers share it


def load():
    for name, (bag, map_yaml) in RUNS.items():
        DATA[name] = (read_bag(os.path.join(PKG, bag)), MapField(os.path.join(PKG, map_yaml)))


def _run(job):
    """ Pool worker: (run name, params, seed, initial_guess offset) -> summary + convergence time. """
    name, params, seed, offset = job
    bag, field = DATA[name]
    guess = None
    if offset is not None:
        # start from the robot's true starting pose (odometry, which starts aligned with these maps),
        # shifted by offset = (dx, dy, dtheta), like a careless rviz 2D Pose Estimate click
        x, y, th = bag['odom'][0, 1:]
        guess = (x + offset[0], y + offset[1], th + offset[2])
    records = replay(bag, field, params, seed=seed, initial_guess=guess)
    return dict(summarize(records), converge_s=convergence_time(records))


def convergence_time(records, good=0.1, streak=3):
    """ Seconds until the scan-to-wall score first stays below `good` m for `streak` updates (inf if never). """
    run = 0
    for r in records:
        run = run + 1 if r['est_score'] < good else 0
        if run == streak:
            return r['t'] - records[0]['t']
    return math.inf


def sweep(pool, names, param, values, extra=None, offset=None, seeds=SEEDS):
    """ Mean summary over seeds for each value of one parameter, per run name. """
    jobs = [(n, dict(extra or {}, **{param: v}), s, offset) for n in names for v in values for s in seeds]
    results = pool.map(_run, jobs)
    table = {}
    for (n, p, s, _), res in zip(jobs, results):
        table.setdefault((n, p[param]), []).append(res)
    out = {}
    for key, rs in table.items():
        out[key] = {k: float(np.mean([r[k] for r in rs])) for k in ('est_median', 'est_frac_good', 'update_ms')}
        out[key]['converged'] = sum(math.isfinite(r['converge_s']) for r in rs)
        finite = [r['converge_s'] for r in rs if math.isfinite(r['converge_s'])]
        out[key]['converge_s'] = float(np.mean(finite)) if finite else math.inf
    return out


def crop(ax, field, xs, ys, margin=1.5):
    ax.imshow(field.image, cmap='gray', extent=field.extent, vmin=0, vmax=255, alpha=0.55, zorder=0)
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xlabel('x in map frame (m)')
    ax.set_ylabel('y in map frame (m)')


def fig_trajectories():
    """ Filter estimate vs raw odometry on the map, plus scan-to-wall score over time, for each bag. """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), gridspec_kw={'height_ratios': [2.2, 1]})
    numbers = {}
    for col, name in enumerate(RUNS):
        bag, field = DATA[name]
        recs = replay(bag, field, seed=0)
        est = np.array([r['estimate'] for r in recs])
        odo = np.array([r['odom'] for r in recs])
        ax = axes[0, col]
        crop(ax, field, np.r_[est[:, 0], odo[:, 0]], np.r_[est[:, 1], odo[:, 1]])
        ax.plot(odo[:, 0], odo[:, 1], color=ODOM, ls='--', label='odometry only')
        ax.plot(est[:, 0], est[:, 1], color=FILTER, label='particle filter')
        ax.plot(*est[0, :2], 'o', color=INK, ms=6, zorder=5)
        ax.annotate('start', est[0, :2], textcoords='offset points', xytext=(6, 6), color=INK_2)
        ax.set_title(name.replace('_', ' '))
        if col == 0:
            ax.legend(loc='best')
        t = np.array([r['t'] for r in recs]) - recs[0]['t']
        ax = axes[1, col]
        ax.plot(t, [r['odom_score'] for r in recs], color=ODOM, ls='--', label='odometry only')
        ax.plot(t, [r['est_score'] for r in recs], color=FILTER, label='particle filter')
        ax.set_xlabel('time since first update (s)')
        if col == 0:
            ax.set_ylabel('scan-to-wall distance (m)\nlower is better')
        ax.set_ylim(bottom=0)
        numbers[name] = summarize(recs)
    fig.suptitle('Particle filter vs. odometry alone on each bag (default parameters)', x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'trajectories.png'), dpi=110)
    plt.close(fig)
    return numbers


def fig_amcl():
    """ Our live filter vs nav2 AMCL on the same gauntlet bag, both recorded with ros2 bag record. """
    pf_bag = read_bag(os.path.join(PKG, 'bags/pf_run_gauntlet_2026-10-18'))
    amcl_bag = read_bag(os.path.join(PKG, 'bags/amcl_run_gauntlet_2026-10-18'))
    field = DATA['gauntlet'][1]

    def scored(bag, topic):
        scan_t = np.array([s[0] for s in bag['scans']])
        rows = []
        for t, x, y, th in bag['poses'][topic]:
            k = int(np.argmin(abs(scan_t - t)))
            _, ranges, angles = bag['scans'][k]
            rows.append((t, x, y, th, scan_score(field, (x, y, th), ranges, angles)))
        return np.array(rows)

    ours, amcl = scored(pf_bag, '/pf_pose'), scored(amcl_bag, '/amcl_pose')
    odo = pf_bag['odom']
    t0 = min(ours[0, 0], amcl[0, 0])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={'width_ratios': [1, 1.4]})
    crop(a1, field, np.r_[ours[:, 1], amcl[:, 1]], np.r_[ours[:, 2], amcl[:, 2]], margin=0.8)
    a1.plot(odo[:, 1], odo[:, 2], color=ODOM, ls='--', lw=1.5, label='odometry')
    a1.plot(amcl[:, 1], amcl[:, 2], color=AMCL, label='nav2 AMCL')
    a1.plot(ours[:, 1], ours[:, 2], color=FILTER, label='our particle filter')
    a1.legend(loc='upper left', fontsize=9)
    a1.set_title('Estimated path in the gauntlet')
    a2.plot(amcl[:, 0] - t0, amcl[:, 4], 'o-', color=AMCL, ms=4, label='nav2 AMCL')
    a2.plot(ours[:, 0] - t0, ours[:, 4], 'o-', color=FILTER, ms=4, label='our particle filter')
    a2.set_xlabel('time (s)')
    a2.set_ylabel('scan-to-wall distance (m), lower is better')
    a2.set_ylim(bottom=0)
    a2.legend()
    a2.set_title('How well each estimate lines the scan up with the map')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'amcl_comparison.png'), dpi=110)
    plt.close(fig)
    return dict(ours_median=float(np.median(ours[:, 4])), amcl_median=float(np.median(amcl[:, 4])),
                ours_updates=len(ours), amcl_updates=len(amcl),
                mean_gap_m=float(np.mean([np.hypot(*(ours[np.argmin(abs(ours[:, 0] - t)), 1:3] - (x, y)))
                                          for t, x, y, _, _ in amcl])))


def two_panel_sweep(res, names, values, param_label, fname, title, log_x=False, show_time=True):
    """ Accuracy (left) and update time (right) vs one parameter: two charts, never two y-axes.
        With show_time=False only the accuracy chart is drawn. """
    fig, axes = plt.subplots(1, 2 if show_time else 1, figsize=(11, 3.8) if show_time else (5.8, 3.8))
    axes = np.atleast_1d(axes)
    a1 = axes[0]
    # every line here is our filter (blue); the bag is told apart by line style and marker, not color
    styles = {'take_1': dict(ls='-', marker='o'), 'take_2': dict(ls='--', marker='s')}
    for n in names:
        a1.plot(values, [res[(n, v)]['est_median'] for v in values], color=FILTER, ms=6,
                label=n.replace('_', ' '), **styles[n])
        if show_time:
            axes[1].plot(values, [res[(n, v)]['update_ms'] for v in values], color=FILTER, ms=6,
                         label=n.replace('_', ' '), **styles[n])
    a1.set_ylabel('median scan-to-wall distance (m)')
    if show_time:
        axes[1].set_ylabel('time per filter update (ms)')
    for a in axes:
        a.set_xlabel(param_label)
        a.set_ylim(bottom=0)
        if log_x:
            a.set_xscale('log')
            a.set_xticks(values)
            a.set_xticklabels([str(v) for v in values])
            a.minorticks_off()
    a1.legend()
    fig.suptitle(title, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=110)
    plt.close(fig)


def fig_recovery(res, fracs, seeds):
    """ How much of the run the filter spends well-localized after a bad initial guess,
        for different amounts of random-particle injection. """
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    names = ['take_1', 'take_2']
    width = 0.35
    x = np.arange(len(fracs))
    ax.set_axisbelow(True)
    for i, (n, hatch) in enumerate(zip(names, ('', '//'))):
        vals = [res[(n, f)]['est_frac_good'] * 100 for f in fracs]
        ax.bar(x + (i - 0.5) * width, vals, width - 0.03, color=FILTER if not hatch else SURFACE,
               edgecolor=FILTER, hatch=hatch, linewidth=1.5, label=n.replace('_', ' '))
        for xi, v in zip(x, vals):
            ax.annotate('%d%%' % v, (xi + (i - 0.5) * width, v), textcoords='offset points',
                        xytext=(0, 3), ha='center', color=INK_2, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(['%d%%' % (f * 100) for f in fracs])
    ax.set_xlabel('share of particles re-drawn at random each resample (random_frac)')
    ax.set_ylabel('updates within 0.1 m of the walls (%)')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left')
    ax.set_title('Recovering from an initial guess 2 m and 90° off (mean of %d seeds)' % len(seeds),
                 loc='left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'recovery.png'), dpi=110)
    plt.close(fig)


def gif_convergence(name='take_1', offset=(0.6, -0.4, math.radians(30)), seed=0):
    """ Animated particle cloud converging from a wrong initial guess, following the robot. """
    bag, field = DATA[name]
    x, y, th = bag['odom'][0, 1:]
    recs = replay(bag, field, seed=seed, initial_guess=(x + offset[0], y + offset[1], th + offset[2]),
                  keep_clouds=True)[:60]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(field.image, cmap='gray', extent=field.extent, vmin=0, vmax=255, alpha=0.6)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    cloud = ax.scatter([], [], s=6, color=FILTER, alpha=0.6, linewidths=0, label='particles')
    est, = ax.plot([], [], 'o', color=INK, ms=7, label='pose estimate')
    odo, = ax.plot([], [], 'o', color=ODOM, ms=7, label='odometry')
    ax.legend(loc='lower left', fontsize=8, frameon=True, facecolor=SURFACE, edgecolor=GRID)
    label = ax.text(0.02, 0.97, '', transform=ax.transAxes, va='top', fontsize=9, color=INK,
                    bbox=dict(facecolor=SURFACE, edgecolor=GRID, boxstyle='round,pad=0.3'))

    def frame(i):
        r = recs[i]
        cloud.set_offsets(r['cloud'][:, :2])
        est.set_data([r['estimate'][0]], [r['estimate'][1]])
        odo.set_data([r['odom'][0]], [r['odom'][1]])
        ax.set_xlim(r['estimate'][0] - 3, r['estimate'][0] + 3)
        ax.set_ylim(r['estimate'][1] - 3, r['estimate'][1] + 3)
        label.set_text('update %d   scan-to-wall %.2f m' % (i + 1, r['est_score']))
        return cloud, est, odo, label

    anim = FuncAnimation(fig, frame, frames=len(recs), blit=False)
    fig.tight_layout()
    anim.save(os.path.join(OUT, 'convergence.gif'), writer=PillowWriter(fps=4), dpi=70)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    load()
    print('trajectories:', fig_trajectories())
    print('amcl:', fig_amcl())
    gif_convergence()
    # timing sweeps run one at a time so update_ms isn't inflated by runs competing for the CPU
    with Pool(1) as serial, Pool() as pool:
        n_values = [25, 50, 100, 200, 300, 500, 1000]
        res = sweep(serial, ['take_1', 'take_2'], 'n_particles', n_values)
        two_panel_sweep(res, ['take_1', 'take_2'], n_values, 'number of particles (log scale)',
                        'sweep_particles.png', 'More particles: accuracy vs. cost', log_x=True)
        print('n_particles:', {k: {m: round(v, 3) for m, v in d.items()} for k, d in res.items()})

        beam_values = [1, 3, 10, 30, 60]
        res = sweep(serial, ['take_1', 'take_2'], 'beam_step', beam_values)
        two_panel_sweep(res, ['take_1', 'take_2'], beam_values, 'beam_step: use every Nth lidar beam (log scale)',
                        'sweep_beam_step.png', 'Fewer beams: accuracy vs. cost', log_x=True)
        print('beam_step:', {k: {m: round(v, 3) for m, v in d.items()} for k, d in res.items()})

        sigma_values = [0.05, 0.1, 0.25, 0.5, 1.0]
        res = sweep(pool, ['take_1', 'take_2'], 'sigma', sigma_values)
        two_panel_sweep(res, ['take_1', 'take_2'], sigma_values, 'sigma of the laser likelihood (m)',
                        'sweep_sigma.png', 'How strict the laser model is', show_time=False)
        print('sigma:', {k: {m: round(v, 3) for m, v in d.items()} for k, d in res.items()})

        fracs = [0.0, 0.05, 0.15, 0.3]
        seeds = list(range(6))
        res = sweep(pool, ['take_1', 'take_2'], 'random_frac', fracs, offset=(2.0, 0.0, math.radians(90)),
                    seeds=seeds)
        fig_recovery(res, fracs, seeds)
        print('recovery:', {k: {m: round(v, 3) for m, v in d.items()} for k, d in res.items()})


if __name__ == '__main__':
    main()
