"""Closed-path geometry and curvature-limited speed planning."""

import numpy as np

from fastsim.track import resample_closed


def curvature(path, offset=6):
    previous = np.roll(path, offset, axis=0)
    following = np.roll(path, -offset, axis=0)
    incoming = path - previous
    outgoing = following - path
    lengths = (np.linalg.norm(incoming, axis=1)
               * np.linalg.norm(outgoing, axis=1)
               * np.linalg.norm(previous - following, axis=1))
    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    return np.divide(2 * cross, lengths, out=np.zeros_like(cross), where=lengths > 0)


def speed_profile(path, car, lateral_accel, acceleration=2.0, deceleration=2.0):
    speeds = np.minimum(car.v_max, np.sqrt(lateral_accel / np.maximum(np.abs(curvature(path)), 1e-6)))
    distances = np.linalg.norm(np.roll(path, -1, axis=0) - path, axis=1)
    for _ in range(2):
        for index in range(len(speeds)):
            following = (index + 1) % len(speeds)
            speeds[following] = min(speeds[following],
                                    np.sqrt(speeds[index] ** 2 + 2 * acceleration * distances[index]))
        for index in range(len(speeds) - 1, -1, -1):
            following = (index + 1) % len(speeds)
            speeds[index] = min(speeds[index],
                                np.sqrt(speeds[following] ** 2 + 2 * deceleration * distances[index]))
    return speeds


def plan(track, car):
    # Preserve the planner settings used by the saved checkpoint.
    path = resample_closed(track.centerline, 0.05)
    speeds = speed_profile(path, car, lateral_accel=min(5.0, 0.8 * car.a_lat_max))
    return path, speeds
