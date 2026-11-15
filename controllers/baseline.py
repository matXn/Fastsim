"""沿中心线以 1 m/s 行驶的纯追踪基线。"""
import numpy as np

from fastsim import CarParams, PathFollower, State, Track


def plan(track: Track, car: CarParams) -> tuple[np.ndarray, np.ndarray]:
    """返回 (path, speed):path 形状 (n,2) 路线点,speed 形状 (n,) 每点车速。"""
    path = track.centerline.copy()
    speed = np.full(len(path), 1.0)
    return path, speed


class Controller:
    def __init__(self, track: Track, car_params: CarParams):
        path, speed = plan(track, car_params)
        self.follower = PathFollower(path, speed, car_params)

    def __call__(self, state: State) -> tuple[float, float]:
        return self.follower(state)
